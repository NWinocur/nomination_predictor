"""
This module provides functionality to extract judicial vacancy data from HTML pages
using Google Cloud's Document AI service and save the results to a CSV file.
"""
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import (
    documentai_v1 as documentai,  # older version allows access to documentai.Document.DocumentLayout.Block()
)
from loguru import logger
import pandas as pd

from nomination_predictor.dataset import fetch_html


def _extract_text_from_layout(layout: documentai.Document.Page.Layout, text: str) -> str:
    """Extract text from a layout element."""
    if not layout.text_anchor.text_segments:
        return ""
    return "".join(
        text[int(segment.start_index):int(segment.end_index)]
        for segment in layout.text_anchor.text_segments
    ).strip()


def _extract_table_data(document: documentai.Document) -> List[Dict[str, Any]]:
    """Extract table data from a Document AI document.
    
    Args:
        document: The Document AI document object
        
    Returns:
        List of dictionaries containing table data
    """
    tables = []
    
    try:
        for page in document.pages:
            for table in page.tables:
                # Extract headers
                headers = []
                if table.header_rows:
                    for cell in table.header_rows[0].cells:
                        cell_text = _extract_text_from_layout(cell.layout, document.text)
                        headers.append(cell_text)
                
                # Extract rows
                rows = []
                for row in table.body_rows:
                    row_data = []
                    for cell in row.cells:
                        cell_text = _extract_text_from_layout(cell.layout, document.text)
                        row_data.append(cell_text)
                    rows.append(row_data)
                
                if rows:
                    tables.append({
                        "headers": headers if headers else [f"Column_{i+1}" for i in range(len(rows[0]))],
                        "rows": rows
                    })
    except Exception as e:
        logger.error(f"Error extracting table data: {str(e)}")
    
    return tables


def _extract_table_data(document: documentai.Document) -> List[Dict[str, Any]]:
    """Extract table data from a Document AI document.
    
    Handles both PDF (page-based) and HTML (document_layout-based) documents.
    
    Args:
        document: The Document AI document object
        
    Returns:
        List of dictionaries containing table data
    """
    tables = []
    
    def process_table(table, doc_text: str) -> Dict[str, Any]:
        """Helper to process a single table and return its data."""
        # Extract headers
        headers = []
        if hasattr(table, 'header_rows') and table.header_rows:
            for cell in table.header_rows[0].cells:
                cell_text = _extract_text_from_layout(cell.layout, doc_text)
                headers.append(cell_text)
        
        # Extract rows
        rows = []
        for row in getattr(table, 'body_rows', []):
            row_data = []
            for cell in row.cells:
                cell_text = _extract_text_from_layout(cell.layout, doc_text)
                row_data.append(cell_text)
            if row_data:  # Only add non-empty rows
                rows.append(row_data)
        
        return {"headers": headers, "rows": rows}
    
    try:
        # Check for HTML document structure (document_layout)
        if hasattr(document, 'document_layout') and hasattr(document.document_layout, 'blocks'):
            # Process HTML document structure
            for block in document.document_layout.blocks:
                if hasattr(block, 'table_block') and block.table_block:
                    table_data = process_table(block.table_block, document.text)
                    if table_data['rows']:
                        tables.append(table_data)
        
        # Check for PDF document structure (pages)
        if hasattr(document, 'pages'):
            # Process PDF document structure
            for page in document.pages:
                for table in getattr(page, 'tables', []):
                    table_data = process_table(table, document.text)
                    if table_data['rows']:
                        tables.append(table_data)
        
        # If no tables found in either structure, log a warning
        if not tables:
            doc_structure = []
            if hasattr(document, 'document_layout'):
                doc_structure.append('document_layout')
            if hasattr(document, 'pages'):
                doc_structure.append('pages')
            logger.warning(
                f"No tables found in document. Detected structure: {', '.join(doc_structure) or 'unknown'}"
            )
            
    except Exception as e:
        logger.error(f"Error extracting table data: {str(e)}")
        if hasattr(e, '__traceback__'):
            logger.debug(f"Traceback: {e.__traceback__}")
    
    return tables



def process_vacancies_to_csv(url: str, output_csv_path: str) -> bool:
    """Process judicial vacancies from HTML page and save to CSV.
    
    Args:
        url: URL of the judicial vacancies page
        output_csv_path: Path to save the CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Fetch HTML content
        logger.info(f"Fetching content from {url}")
        html_content = fetch_html(url)
        if not html_content:
            logger.error("Failed to fetch HTML content")
            return False
            
        # Initialize Document AI client
        project_id = os.getenv("GCP_PROJECT_ID")
        location = "us"  # Format is 'us' or 'eu'
        processor_id = os.getenv("GCP_PROCESSOR_ID")
        
        if not all([project_id, processor_id]):
            logger.error("Missing required environment variables")
            return False
            
        client = documentai.DocumentProcessorServiceClient()
        name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
        
        # Process the document
        logger.info("Processing document with Document AI")
        raw_document = documentai.RawDocument(
            content=html_content.encode(),
            mime_type="text/html"
        )
        
        request = documentai.ProcessRequest(
            name=name,
            raw_document=raw_document
        )
        
        result = client.process_document(request=request)
        document = result.document
        
        # Extract tables
        tables = _extract_table_data(document)
        
        if not tables:
            logger.warning("No tables found in document")
            return False
            
        # Combine all tables into one DataFrame
        all_rows = []
        for table in tables:
            if table["rows"]:
                if table["headers"]:
                    all_rows.append(table["headers"])
                all_rows.extend(table["rows"])
        
        if not all_rows:
            logger.warning("No valid table data found")
            return False
            
        # Create DataFrame (use first row as headers if available)
        if len(all_rows) > 1 and all_rows[0]:
            df = pd.DataFrame(all_rows[1:], columns=all_rows[0])
        else:
            df = pd.DataFrame(all_rows)
        
        # Clean up the DataFrame
        df = df.dropna(how='all').reset_index(drop=True)
        
        # Ensure output directory exists
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV with pipe separator
        df.to_csv(output_path, sep='|', index=False)
        logger.success(f"Successfully saved {len(df)} rows to {output_path}")
        return True
        
    except DefaultCredentialsError:
        logger.error("Google Cloud credentials not found. Ensure GOOGLE_APPLICATION_CREDENTIALS is set.")
        return False
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return False


def main() -> None:
    """Main function to process judicial vacancies and save to CSV."""
    # Example URL - replace with actual URL
    url = "https://www.uscourts.gov/judges-judgeships/judicial-vacancies"
    
    # Output to data/interim directory
    output_dir = Path(__file__).parent.parent / "data" / "interim"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "judicial_vacancies_2025.csv"
    
    success = process_vacancies_to_csv(url, str(output_path))
    if success:
        logger.info(f"Successfully processed vacancies to {output_path}")
    else:
        logger.error("Failed to process vacancies")


if __name__ == "__main__":
    main()