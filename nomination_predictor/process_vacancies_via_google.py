"""
This module provides functionality to extract judicial vacancy data from HTML pages
using Google Cloud's Document AI service and save the results to a CSV file.

TODO: if we're able to parse HTML purely locally, get rid of this file.  If we need GCP Doc AI, update this file to handle new no-PDF de-scope until we've proven out HTML-only parsing.

"""
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
from google.cloud import (
    documentai_v1 as documentai,  # older version allows access to documentai.Document.DocumentLayout.Block()
)
from loguru import logger
import pandas as pd
import requests


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



def _create_dataframe_from_tables(tables: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert extracted tables into a pandas DataFrame.
    
    Args:
        tables: List of tables with headers and rows
        
    Returns:
        DataFrame containing all table data
        
    Raises:
        ValueError: If there's an issue with the table structure
    """
    if not tables:
        raise ValueError("No tables provided")
    
    all_rows = []
    for table in tables:
        if not table.get("rows"):
            continue
            
        # Add headers if they exist
        if table.get("headers"):
            all_rows.append(table["headers"])
        all_rows.extend(table["rows"])
    
    if not all_rows:
        raise ValueError("No valid table data found")
    
    # Verify all rows have the same number of columns
    if len(all_rows) > 1:
        num_columns = len(all_rows[0])
        for i, row in enumerate(all_rows[1:], 1):
            if len(row) != num_columns:
                logger.error(f"Row {i} has {len(row)} columns, expected {num_columns}")
                logger.debug(f"Problematic row: {row}")
                raise ValueError(f"Column count mismatch in row {i}")
    
    # Use first row as headers if available
    if len(all_rows) > 1 and all_rows[0]:
        return pd.DataFrame(all_rows[1:], columns=all_rows[0])
    return pd.DataFrame(all_rows)

def _process_document_with_dai(
    content: bytes,
    mime_type: str,
    project_id: str,
    location: str,
    processor_id: str
) -> documentai.Document:
    """Process document using Document AI.
    
    Args:
        content: Document content as bytes
        mime_type: Document MIME type
        project_id: GCP project ID
        location: Processor location
        processor_id: Processor ID
        
    Returns:
        Processed Document object
    """
    client = documentai.DocumentProcessorServiceClient()
    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
    
    logger.debug(f"Processing document with processor: {name}")
    raw_document = documentai.RawDocument(
        content=content,
        mime_type=mime_type
    )
    
    request = documentai.ProcessRequest(
        name=name,
        raw_document=raw_document
    )
    
    result = client.process_document(request=request)
    return result.document

def _validate_environment() -> tuple[str, str, str]:
    """Validate and return required environment variables.
    
    Returns:
        Tuple of (project_id, location, processor_id)
        
    Raises:
        ValueError: If required environment variables are missing or not strings
    """
    load_dotenv()
    project_id = os.getenv("GCP_PROJECT_ID")
    processor_id = os.getenv("GCP_PROCESSOR_ID")
    location = "us"  # Default location
    
    # Check if any value is None or not a string
    if (project_id is None or 
        processor_id is None or 
        not isinstance(project_id, str) or 
        not isinstance(processor_id, str) or
        not isinstance(location, str)):
        
        missing = []
        if project_id is None or not isinstance(project_id, str):
            missing.append("GCP_PROJECT_ID")
        if processor_id is None or not isinstance(processor_id, str):
            missing.append("GCP_PROCESSOR_ID")
        if not isinstance(location, str):  # location has a default, so we only check type
            missing.append("location")
            
        raise ValueError(
            f"Missing or invalid environment variables: {', '.join(missing)}"
        )
    
    return project_id, location, processor_id

def process_vacancies_to_csv(content: bytes, output_csv_path: str, mime_type: str = "text/html") -> bool:
    """Process judicial vacancies from content and save to CSV.
    
    Args:
        content: The document content as bytes
        output_csv_path: Path to save the CSV file
        mime_type: The MIME type of the document ("text/html" or "application/pdf")
        
    Returns:
        bool: True if successful, False otherwise
    """
    if mime_type not in ["text/html", "application/pdf"]:
        logger.error(f"Unsupported MIME type: {mime_type}. Must be 'text/html' or 'application/pdf'")
        return False

    try:
        # Validate environment and get configuration
        project_id, location, processor_id = _validate_environment()
        
        # Process document
        logger.info(f"Processing {mime_type} document with Document AI")
        document = _process_document_with_dai(
            content=content,
            mime_type=mime_type,
            project_id=project_id,
            location=location,
            processor_id=processor_id
        )
        
        # Extract and process tables
        tables = _extract_table_data(document)
        if not tables:
            logger.warning("No tables found in document")
            return False
        
        # Create and save DataFrame
        df = _create_dataframe_from_tables(tables)
        
        # Clean up the DataFrame
        df = df.dropna(how='all').reset_index(drop=True)
        
        # Ensure output directory exists
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV with pipe separator
        df.to_csv(output_path, sep='|', index=False)
        logger.success(f"Successfully saved {len(df)} rows to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return False


def download_file(url: str, timeout: int = 30) -> Tuple[Optional[bytes], Optional[str]]:
    """Download file from URL and return its content and detected MIME type.
    
    Args:
        url: URL of the file to download
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (content, mime_type) or (None, None) if download fails
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        # Get content type from headers
        content_type = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
        
        # If content type is generic, try to determine from URL
        if not content_type or content_type in ['application/octet-stream', 'text/plain']:
            parsed = urlparse(url)
            ext = Path(parsed.path).suffix.lower()
            content_type = mimetypes.guess_type(url)[0] or content_type
        
        # Map common content types to our supported types
        if 'pdf' in content_type:
            content_type = 'application/pdf'
        elif 'html' in content_type or 'text/' in content_type:
            content_type = 'text/html'
            
        return response.content, content_type
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        return None, None

def process_vacancies_from_url(url: str, output_csv_path: str) -> bool:
    """Download and process judicial vacancies from a URL.
    
    Args:
        url: URL of the document to process (HTML or PDF)
        output_csv_path: Path to save the CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Downloading document from {url}")
    content, mime_type = download_file(url)
    
    if not content:
        logger.error("Failed to download document")
        return False
    
    if not mime_type or mime_type not in ["text/html", "application/pdf"]:
        logger.warning(f"Unknown or unsupported content type: {mime_type}. Will attempt to process as HTML")
        mime_type = "text/html"
    
    return process_vacancies_to_csv(
        content=content,
        output_csv_path=output_csv_path,
        mime_type=mime_type
    )

def main() -> None:
    """Main function to process judicial vacancies from a URL and save to CSV."""
    # Example URLs - uncomment the one you want to use
    url = "https://www.uscourts.gov/judges-judgeships/judicial-vacancies/archive-judicial-vacancies/2003/01/vacancies"
    # url = "https://www.uscourts.gov/sites/default/files/example.pdf"  # Example PDF URL
    
    # Output to data/interim directory
    output_dir = Path(__file__).parent.parent / "data" / "interim"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename based on URL
    url_path = url.rstrip('/').split('/')[-1] or "vacancies"
    output_path = output_dir / f"{url_path}.csv"
    
    success = process_vacancies_from_url(url, str(output_path))
    if success:
        logger.info(f"Successfully processed vacancies to {output_path}")
    else:
        logger.error("Failed to process vacancies")

if __name__ == "__main__":
    main()