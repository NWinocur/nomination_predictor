"""This is a Google Gemini 2.5 Pro derived Python script intended as an initial experiment with Google Cloud's Document AI.

This script was initially authored by prompting Gemini as follows:

Would you kindly draft for me a function which does the following?
- calls my `fetch_html(url: str) -> str:` function to download the following html file from the Internet:
https://www.uscourts.gov/judges-judgeships/judicial-vacancies/archive-judicial-vacancies/2025/07/vacancies
- temporarily stores that html in whichever data type is considered best-practice to transmit to Google Cloud's Document AI service
- sends it to Google Cloud's Document AI service, with the request that the service return the data found in the page's table of judicial vacancies, in a tabular format well-suited to storing as a pandas dataframe
- prints and/or logs to the terminal that tabular data
- saves that data to a CSV file (using whichever character is considered best-practice to use as a comma-alternative, because for this data, characters such as , and . and - and / are ubiquitous throughout) """

import os

from dotenv import load_dotenv
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import documentai
import pandas as pd


def fetch_html(url: str) -> str:
    """
    Fetches the raw HTML content from a given URL.

    Args:
        url: The URL to fetch.

    Returns:
        The HTML content as a string, or an empty string if an error occurs.
    """
    # This is a basic implementation. In a real project, you might have
    # more robust error handling, headers, etc.
    import requests
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return ""

def _get_text_from_layout(layout: documentai.Document.Page.Layout, document: documentai.Document) -> str:
    """
    Helper function to extract text from a layout segment in the Document AI response.
    """
    text = ""
    if layout.text_anchor and layout.text_anchor.text_segments:
        for segment in layout.text_anchor.text_segments:
            start, end = int(segment.start_index), int(segment.end_index)
            text += document.text[start:end]
    return text.strip().replace('\n', ' ')

def process_vacancies_to_csv(url: str, output_csv_path: str):
    """
    Fetches HTML from a URL, sends it to Document AI to extract tables,
    prints the result, and saves it to a CSV file.

    Args:
        url: The URL of the judicial vacancies page to process.
        output_csv_path: The file path to save the resulting CSV.
    """
    # 1. Load credentials and configuration from .env file
    load_dotenv()
    project_id = os.getenv("GCP_PROJECT_ID")
    processor_id = os.getenv("GCP_PROCESSOR_ID")
    location = "us"  # Or "eu", depending on your processor's location

    if not all([project_id, processor_id, os.getenv("GOOGLE_APPLICATION_CREDENTIALS")]):
        print("❌ Error: Ensure GOOGLE_APPLICATION_CREDENTIALS, GCP_PROJECT_ID, and GCP_PROCESSOR_ID are set in your .env file.")
        return

    # 2. Fetch the HTML content
    print(f"Fetching HTML from: {url}")
    html_content = fetch_html(url)
    if not html_content:
        print("Could not retrieve HTML content. Aborting.")
        return

    # 3. Send the document to Document AI
    try:
        print("Initializing Document AI client...")
        client = documentai.DocumentProcessorServiceClient()
        processor_path = client.processor_path(project_id, location, processor_id)

        # Best practice is to send the raw bytes of the HTML
        raw_document = documentai.RawDocument(
            content=html_content.encode("utf-8"),
            mime_type="text/html",  # Specify the content type
        )

        request = documentai.ProcessRequest(name=processor_path, raw_document=raw_document)

        print("Sending request to Document AI for processing...")
        result = client.process_document(request=request)
        document = result.document
        print("Successfully received response.")

    except DefaultCredentialsError:
        print("❌ Authentication Failed. Please check your GOOGLE_APPLICATION_CREDENTIALS path in the .env file.")
        return
    except Exception as e:
        print(f"An error occurred during Document AI processing: {e}")
        return

    # 4. Parse the response and print the table
    print("\n--- Extracted Table Data ---")
    all_rows_data = []
    
    # Assuming the first table on the first page is the one we want
    if document.pages and document.pages[0].tables:
        table = document.pages[0].tables[0]

        # Extract header
        header_row = table.header_rows[0]
        headers = [_get_text_from_layout(cell.layout, document) for cell in header_row.cells]
        print(f"Headers: {headers}")

        # Extract body rows
        for row in table.body_rows:
            row_data = [_get_text_from_layout(cell.layout, document) for cell in row.cells]
            all_rows_data.append(row_data)
            print(row_data) # Log each row to the terminal
    else:
        print("No tables were found in the document.")
        return

    # 5. Save the data to a CSV file
    if all_rows_data:
        df = pd.DataFrame(all_rows_data, columns=headers)
        
        # Using a pipe '|' is a common and safe alternative to a comma
        separator = '|'
        
        df.to_csv(output_csv_path, sep=separator, index=False, encoding='utf-8')
        print(f"\n✅ Successfully saved data to '{output_csv_path}' using '{separator}' as the separator.")


if __name__ == '__main__':
    # URL for the well-formatted 2025 data
    target_url = "https://www.uscourts.gov/judges-judgeships/judicial-vacancies/archive-judicial-vacancies/2025/07/vacancies"
    output_filename = "judicial_vacancies_2025.csv"
    
    process_vacancies_to_csv(url=target_url, output_csv_path=output_filename)

