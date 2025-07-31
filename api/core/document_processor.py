import io
import os
import fitz  # PyMuPDF
import docx  # python-docx
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List
from urllib.parse import urlparse

# Create a thread pool for CPU-bound parsing tasks
pdf_executor = ThreadPoolExecutor(max_workers=os.cpu_count())

async def download_document(url: str) -> bytes:
    """Asynchronously downloads a document from a URL."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            return response.content
        except httpx.RequestError as e:
            raise ValueError(f"Error downloading document from {url}: {e}")

# --- Private, specific text extractors ---

def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts text from PDF bytes using PyMuPDF."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

def _extract_text_from_docx(docx_bytes: bytes) -> str:
    """Extracts text from DOCX bytes using python-docx."""
    with io.BytesIO(docx_bytes) as docx_file:
        doc = docx.Document(docx_file)
        return "\n\n".join(para.text for para in doc.paragraphs if para.text)

def _extract_text_from_txt(txt_bytes: bytes) -> str:
    """Extracts text from TXT bytes."""
    # Decode with utf-8, ignoring errors for robustness
    return txt_bytes.decode('utf-8', errors='ignore')

# --- Main synchronous dispatcher function ---

def extract_text_from_document_bytes(document_bytes: bytes, file_type: str) -> str:
    """
    Dispatches to the correct text extractor based on the file type.
    This function is designed to be run in a thread pool.
    """
    file_type = file_type.lower()
    
    if file_type == '.pdf':
        return _extract_text_from_pdf(document_bytes)
    elif file_type == '.docx':
        return _extract_text_from_docx(document_bytes)
    elif file_type == '.txt':
        return _extract_text_from_txt(document_bytes)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

# --- Main asynchronous entry point for processing ---

async def process_document(url: str, document_bytes: bytes) -> str:
    """
    Asynchronously processes document bytes to extract text based on the URL's file type.
    """
    # 1. Determine file type from URL
    try:
        path = urlparse(url).path
        file_type = os.path.splitext(path)[1]
        if not file_type:
            raise ValueError("Could not determine file type from URL.")
    except Exception as e:
        raise ValueError(f"Invalid document URL provided: {e}")

    # 2. Run the synchronous, CPU-bound extraction in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        pdf_executor, 
        extract_text_from_document_bytes, 
        document_bytes, 
        file_type
    )

# --- Chunking function (remains the same) ---

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 90) -> List[str]:
    """
    Splits text into smaller chunks using a hierarchical strategy.
    Tries to split by paragraph, then by line, then by character.
    """
    if not text:
        return []

    # Start with a list of the whole text
    docs = [text]
    
    # Define separators in order of preference
    separators = ["\n\n", "\n", " ", ""]
    
    for sep in separators:
        new_docs = []
        for doc in docs:
            if len(doc) > chunk_size:
                # Split the doc and add the pieces to new_docs
                splits = doc.split(sep)
                
                # Re-combine small pieces to form chunks of the right size
                current_chunk = ""
                for i, part in enumerate(splits):
                    # If adding the next part (and separator) would be too long
                    if len(current_chunk) + len(part) + len(sep) > chunk_size and current_chunk:
                        new_docs.append(current_chunk)
                        # Start the next chunk with an overlap
                        current_chunk = current_chunk[-(overlap if overlap < len(current_chunk) else 0):] + sep + part
                    else:
                        current_chunk += (sep if current_chunk else "") + part
                
                # Add the last remaining chunk
                if current_chunk:
                    new_docs.append(current_chunk)
            else:
                # If the doc is already small enough, keep it
                new_docs.append(doc)
        docs = new_docs

    # Final pass to ensure no empty strings and add the prefix
    final_chunks = [f"search_document: {doc.strip()}" for doc in docs if doc.strip()]
    
    print(f"Successfully chunked document into {len(final_chunks)} pieces.")
    return final_chunks