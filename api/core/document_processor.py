# api/core/document_processor.py
import io
import os
import re
import fitz  # PyMuPDF
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List
from urllib.parse import urlparse
import easyocr # Use the GPU-accelerated library

# --- Initialize the OCR reader once to load the model into GPU memory ---
print("Initializing GPU-accelerated OCR Reader...")
# This will automatically use the RTX 4060 if PyTorch with CUDA is installed.
OCR_READER = easyocr.Reader(['en'], gpu=True)
print("✅ OCR Reader loaded successfully on GPU.")

# Use a simple thread pool. The heavy work is on the GPU, not the CPU.
cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count())

async def download_document(url: str) -> bytes:
    """Downloads a document from a URL."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            return response.content
        except httpx.RequestError as e:
            raise ValueError(f"Error downloading document from {url}: {e}")

def _extract_text_from_pdf_sync(pdf_bytes: bytes) -> str:
    """
    Synchronous function to extract text. It attempts a fast digital extraction
    and falls back to high-speed GPU OCR if the document is scanned.
    """
    # 1. Try the fast method for digitally native PDFs first.
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            digital_text = "".join(page.get_text() for page in doc)
        # If we get a reasonable amount of text, we're done.
        if len(digital_text.strip()) > 100:
            print("INFO: Successfully extracted text from digitally native PDF.")
            return digital_text
    except Exception as e:
        print(f"Digital extraction failed: {e}. Proceeding to OCR.")
        digital_text = ""

    # 2. If digital text is minimal, it's a scanned PDF. Use GPU OCR.
    print("WARN: Digital text extraction yielded minimal results. Switching to GPU-based OCR.")
    full_ocr_text = []
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                print(f"Processing page {page_num + 1}/{len(doc)} with GPU OCR...")
                # Render the page at a high resolution for better OCR quality.
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                
                # easyocr reads the text from the image bytes.
                # detail=0 and paragraph=True are optimized for speed and structure.
                result = OCR_READER.readtext(img_bytes, detail=0, paragraph=True)
                full_ocr_text.extend(result)
    except Exception as ocr_error:
        raise IOError(f"A critical error occurred during GPU OCR processing: {ocr_error}")

    return "\n".join(full_ocr_text)

async def process_document(url: str, document_bytes: bytes) -> str:
    """
    Asynchronously processes a document, offloading the extraction to a thread pool.
    Returns the extracted text.
    """
    path = urlparse(url).path
    file_type = os.path.splitext(path)[1].lower()
    if not file_type:
        raise ValueError("Could not determine file type from URL.")

    if file_type != '.pdf':
        return document_bytes.decode('utf-8', errors='ignore')

    loop = asyncio.get_event_loop()
    # Run the synchronous extraction function in the executor.
    return await loop.run_in_executor(
        cpu_executor, 
        _extract_text_from_pdf_sync, 
        document_bytes
    )

def fast_sliding_window_chunker(
    text: str, chunk_size: int, chunk_overlap: int
) -> List[str]:
    """
    Extremely fast and simple text chunker using a sliding window.
    This is the only chunking strategy you need.
    """
    if not text or not text.strip():
        return []

    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    text_len = len(text)
    i = 0
    while i < text_len:
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        step = chunk_size - chunk_overlap
        i += step

    prefixed_chunks = [f"search_document: {chunk.strip()}" for chunk in chunks if chunk.strip()]
    print(f"Successfully created {len(prefixed_chunks)} fast chunks.")
    return prefixed_chunks