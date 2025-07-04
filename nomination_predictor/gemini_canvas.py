import os

from dotenv import load_dotenv
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import documentai_v1 as documentai
import pandas as pd


def fetch_html(url: str) -> str:
    """
    Fetches the raw HTML content from a given URL.

    Args:
        url: The URL to fetch.

    Returns:
        The HTML content as a string, or an empty string if an error occurs.
    """
    import requests
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return ""

def _get_text_from_cell_blocks(cell: documentai.Document.DocumentLayout.DocumentLayoutBlock, document: documentai.Document) -> str:
    """
    Helper function to extract text from the blocks within a table cell
    from the documentLayout structure.
    """
    # The text is directly available in the nested textBlock.
    return " ".join(
        [block.text_block.text for block in cell.blocks if block.text_block]
    ).strip().replace('\n', ' ')


def _find_and_process_tables_from_layout(block: documentai.Document.DocumentLayout.DocumentLayoutBlock, document: documentai.Document, found_tables: list):
    """
    Recursively searches for tableBlocks in the documentLayout and processes them.
    """
    if block.table_block:
        table_data = {
            "headers": [],
            "body_rows": []
        }
        # Extract header
        if block.table_block.header_rows:
            header_row = block.table_block.header_rows[0]
            table_data["headers"] = [_get_text_from_cell_blocks(cell, document) for cell in header_row.cells]

        # Extract body rows
        for row in block.table_block.body_rows:
            row_data = [_get_text_from_cell_blocks(cell, document) for cell in row.cells]
            table_data["body_rows"].append(row_data)
        
        found_tables.append(table_data)

    # Recursively search in nested blocks
    if block.blocks:
        for sub_block in block.blocks:
            _find_and_process_tables_from_layout(sub_block, document, found_tables)


def process_vacancies_to_csv(url: str, output_csv_path: str, project_id: str, processor_id: str):
    """
    Fetches HTML from a URL, sends it to a specified Document AI processor,
    extracts tables, prints the result, and saves it to a CSV file.

    Args:
        url: The URL of the judicial vacancies page to process.
        output_csv_path: The file path to save the resulting CSV.
        project_id: Your Google Cloud project ID.
        processor_id: The ID of the Document AI processor to use.
    """
    location = "us" # Or "eu", depending on your processor's location

    print(f"Fetching HTML from: {url}")
    html_content = fetch_html(url)
    if not html_content:
        print("Could not retrieve HTML content. Aborting.")
        return

    try:
        print("Initializing Document AI client...")
        client = documentai.DocumentProcessorServiceClient()
        processor_path = client.processor_path(project_id, location, processor_id)

        raw_document = documentai.RawDocument(
            content=html_content.encode("utf-8"),
            mime_type="text/html",
        )
        request = documentai.ProcessRequest(name=processor_path, raw_document=raw_document)

        print(f"Sending request to processor '{processor_id}'...")
        result = client.process_document(request=request)
        document = result.document
        print("Successfully received response.")

    except DefaultCredentialsError:
        print("❌ Authentication Failed. Please check your GOOGLE_APPLICATION_CREDENTIALS path in the .env file.")
        return
    except Exception as e:
        print(f"An error occurred during Document AI processing: {e}")
        return

    # 4. Parse the response using the documentLayout structure.
    # Note: If using a Form Parser for PDFs, you would parse document.pages instead.
    print("\n--- Extracted Table Data ---")
    all_found_tables = []
    if document.document_layout and document.document_layout.blocks:
        for block in document.document_layout.blocks:
            _find_and_process_tables_from_layout(block, document, all_found_tables)

    if not all_found_tables:
        print("No tables were found in the document layout.")
        return
        
    # For this example, we'll process the first table found.
    # The HTML page you provided has two tables; the first is the main one.
    target_table = all_found_tables[0]
    headers = target_table["headers"]
    all_rows_data = target_table["body_rows"]

    print(f"Headers: {headers}")
    for row in all_rows_data:
        print(row)

    # 5. Save the data to a CSV file
    if all_rows_data:
        df = pd.DataFrame(all_rows_data, columns=headers)
        separator = '|'
        df.to_csv(output_csv_path, sep=separator, index=False, encoding='utf-8')
        print(f"\n✅ Successfully saved data to '{output_csv_path}' using '{separator}' as the separator.")


if __name__ == '__main__':
    # Load environment variables from .env file
    # Your .env file should contain:
    # GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/key.json
    # GCP_PROJECT_ID="your-project-id"
    # GCP_LAYOUT_PROCESSOR_ID="your-layout-parser-id-for-html"
    # GCP_FORM_PROCESSOR_ID="your-form-parser-id-for-pdf"
    load_dotenv()
    
    project_id = os.getenv("GCP_PROJECT_ID")
    
    # --- Logic to choose processor ---
    # For this run, we are processing an HTML URL, so we use the Layout Parser.
    # In the future, you could add logic here to detect file type and choose the appropriate processor.
    processor_id = os.getenv("GCP_LAYOUT_PROCESSOR_ID")
    
    target_url = "https://www.uscourts.gov/judges-judgeships/judicial-vacancies/archive-judicial-vacancies/2025/07/vacancies"
    output_filename = "judicial_vacancies_2025.csv"
    
    if not all([project_id, processor_id, os.getenv("GOOGLE_APPLICATION_CREDENTIALS")]):
        print("❌ Error: Ensure GOOGLE_APPLICATION_CREDENTIALS, GCP_PROJECT_ID, and a processor ID are set in your .env file.")
    else:
        process_vacancies_to_csv(
            url=target_url, 
            output_csv_path=output_filename,
            project_id=project_id,
            processor_id=processor_id
        )
