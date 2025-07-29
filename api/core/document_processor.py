# api/core/document_processor.py
import io
import fitz  # PyMuPDF
import httpx
from typing import List

async def download_document(url: str) -> bytes:
    """Asynchronously downloads a document from a URL."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            return response.content
        except httpx.RequestError as e:
            raise ValueError(f"Error downloading document from {url}: {e}")

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extracts text from PDF content using PyMuPDF."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {e}")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """Splits text into smaller chunks and prepends 'search_document: ' to each chunk."""
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
    return [f"search_document: {chunk}" for chunk in chunks]