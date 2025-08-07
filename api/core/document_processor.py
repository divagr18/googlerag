# api/core/document_processor.py

import subprocess
import io
import os
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from typing import AsyncIterator, List, Tuple, Dict, Optional
from urllib.parse import urlparse
import numpy as np
import fitz  # PyMuPDF
import httpx
from .embedding_manager import OptimizedEmbeddingManager
import docx
import pandas as pd
from PIL import Image
import pytesseract
from httpx import AsyncHTTPTransport
from pptx import Presentation
from .query_expander import DomainQueryExpander
import tempfile
import shutil

# --- DEV MODE MODIFICATION START ---
# Define the URL prefix and the local path for the specific file.
PRINCIPIA_URL_PREFIX = "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf"
# Using a raw string (r"...") is important for Windows paths with backslashes.
LOCAL_PRINCIPIA_PATH = r"C:\Users\Keshav\Downloads\principia_newton (2).pdf"
# --- DEV MODE MODIFICATION END ---


async def download_with_aria2c(url: str, destination_path: str):
    """
    Uses the external aria2c command-line tool to download a file.
    Runs the blocking command in a thread pool to avoid freezing the event loop.
    """
    print(f"ARIA2C: Starting high-speed download for {url}")
    start_time = time.perf_counter()
    
    command = [
        "aria2c",
        url,
        "--dir", os.path.dirname(destination_path),
        "--out", os.path.basename(destination_path),
        "-x", "8",
        "-s", "8",
        "--quiet=true",
        "--auto-file-renaming=false"
    ]

    def run_blocking_download():
        """This function will be executed in a separate thread."""
        try:
            # Use the standard, blocking subprocess.run
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return result
        except FileNotFoundError:
            # This error is caught and re-raised with a more helpful message
            raise RuntimeError("`aria2c` is not installed or not in the system's PATH.")
        except subprocess.CalledProcessError as e:
            # This catches errors from aria2c itself (e.g., download failed)
            raise IOError(f"aria2c download failed with code {e.returncode}: {e.stderr}")

    try:
        loop = asyncio.get_running_loop()
        # Run the blocking function in the default executor (a ThreadPool)
        await loop.run_in_executor(None, run_blocking_download)
        
        duration = time.perf_counter() - start_time
        print(f"ARIA2C: Download complete in {duration:.2f}s. File saved to temporary location.")

    except Exception as e:
        # Clean up partially downloaded file if it exists
        if os.path.exists(destination_path):
            os.remove(destination_path)
        raise e

async def stream_document(url: str) -> AsyncIterator[bytes]:
    """
    Streams a document. If it's the Principia URL, it uses a local file with a
    simulated delay. Otherwise, it downloads to a temporary location using aria2c.
    """
    # --- DEV MODE MODIFICATION START ---
    if url.startswith(PRINCIPIA_URL_PREFIX):
        if os.path.exists(LOCAL_PRINCIPIA_PATH):
            print(f"DEV_MODE: Matched Principia URL. Using local file: {LOCAL_PRINCIPIA_PATH}")
            print("DEV_MODE: Simulating a 1-second delay...")
            print("DEV_MODE: Simulation complete. Streaming local file content.")
            
            with open(LOCAL_PRINCIPIA_PATH, "rb") as f:
                content = f.read()
            
            chunk_size = 8192
            for i in range(0, len(content), chunk_size):
                yield content[i:i+chunk_size]
            return # Exit the function after streaming the local file
        else:
            print(f"DEV_MODE WARNING: Local file not found at '{LOCAL_PRINCIPIA_PATH}'. Falling back to web download.")
    # --- DEV MODE MODIFICATION END ---

    # Fallback to original logic if the URL is not the special one OR the local file doesn't exist.
    temp_dir = tempfile.mkdtemp()
    original_filename = os.path.basename(urlparse(url).path) or "downloaded_file"
    destination_path = os.path.join(temp_dir, original_filename)

    try:
        await download_with_aria2c(url, destination_path)
        with open(destination_path, "rb") as f:
            content = f.read()
        
        chunk_size = 8192
        for i in range(0, len(content), chunk_size):
            yield content[i:i+chunk_size]

    except Exception as e:
        print(f"FATAL: An error occurred during the download or streaming process: {e}")
        raise
    finally:
        print(f"CLEANUP: Removing temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

def _extract_page_text_from_bytes(args: Tuple[bytes, int]) -> Tuple[str, int]:
    pdf_bytes, page_no = args
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return doc.load_page(page_no).get_text(), page_no + 1
    except Exception as e:
        print(f"Error extracting page {page_no}: {e}")
        return "", page_no + 1

async def _extract_text_from_pdf_stream(pdf_stream: io.BytesIO) -> List[Tuple[str, int]]:
    pdf_bytes = pdf_stream.getvalue()
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            page_count = doc.page_count
    except Exception as e:
        raise ValueError(f"Failed to open PDF from memory stream: {e}")
    if page_count == 0:
        return []
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(cpu_executor, _extract_page_text_from_bytes, (pdf_bytes, i))
        for i in range(page_count)
    ]
    return await asyncio.gather(*tasks)

def _extract_text_from_docx_stream(docx_stream: io.BytesIO) -> List[Tuple[str, int]]:
    try:
        document = docx.Document(docx_stream)
        full_text = "\n".join([para.text for para in document.paragraphs])
        return [(full_text, 1)]
    except Exception as e:
        raise ValueError(f"Failed to process DOCX file: {e}")

def _extract_text_from_xlsx_stream(xlsx_stream: io.BytesIO) -> List[Tuple[str, int]]:
    try:
        sheets = pd.read_excel(xlsx_stream, sheet_name=None)
        full_text = [f"Sheet: {name}\n{df.to_csv(index=False)}" for name, df in sheets.items()]
        return [("\n\n".join(full_text), 1)]
    except Exception as e:
        raise ValueError(f"Failed to process XLSX file: {e}")

def _extract_text_from_image_stream(image_stream: io.BytesIO) -> List[Tuple[str, int]]:
    try:
        image = Image.open(image_stream)
        text = pytesseract.image_to_string(image)
        return [(text, 1)]
    except Exception as e:
        raise ValueError(f"Failed to process image file with OCR: {e}")

def _extract_text_from_pptx_stream(pptx_stream: io.BytesIO) -> List[Tuple[str, int]]:
    try:
        presentation = Presentation(pptx_stream)
        pages_data = []
        for i, slide in enumerate(presentation.slides):
            slide_texts = [run.text for shape in slide.shapes if shape.has_text_frame for para in shape.text_frame.paragraphs for run in para.runs]
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
                slide_texts.append("\n--- Notes ---\n" + slide.notes_slide.notes_text_frame.text)
            pages_data.append(("\n".join(slide_texts), i + 1))
        return pages_data
    except Exception as e:
        raise ValueError(f"Failed to process PPTX file: {e}")

async def process_document_stream(url: str, document_iterator: AsyncIterator[bytes]) -> List[Tuple[str, int]]:
    path = urlparse(url).path
    file_type = os.path.splitext(path)[1].lower() or ".txt"
    document_bytes_io = io.BytesIO(b"".join([chunk async for chunk in document_iterator]))
    loop = asyncio.get_running_loop()
    
    image_formats = ['.png', '.jpeg', '.jpg', '.bmp', '.tiff', '.gif']
    if file_type == '.pdf': return await _extract_text_from_pdf_stream(document_bytes_io)
    if file_type == '.docx': return await loop.run_in_executor(cpu_executor, _extract_text_from_docx_stream, document_bytes_io)
    if file_type == '.pptx': return await loop.run_in_executor(cpu_executor, _extract_text_from_pptx_stream, document_bytes_io)
    if file_type == '.xlsx': return await loop.run_in_executor(cpu_executor, _extract_text_from_xlsx_stream, document_bytes_io)
    if file_type in image_formats: return await loop.run_in_executor(cpu_executor, _extract_text_from_image_stream, document_bytes_io)
    if file_type in ['.txt', '.md', '.csv']: return [(document_bytes_io.read().decode('utf-8', errors='ignore'), 1)]
    raise ValueError(f"Unsupported file type: '{file_type}'.")

def smart_paragraph_split(text: str, page_num: int) -> List[Tuple[str, int]]:
    MAX_PARA_LENGTH = 4096
    paragraphs = text.split('\n\n')
    result: List[Tuple[str, int]] = []
    for para in paragraphs:
        para = para.strip()
        if len(para) > 50:
            if len(para) > MAX_PARA_LENGTH:
                for i in range(0, len(para), MAX_PARA_LENGTH):
                    sub_para = para[i:i + MAX_PARA_LENGTH].strip()
                    if len(sub_para) > 50:
                        result.append((sub_para, page_num))
            else:
                result.append((para, page_num))
    return result

async def optimized_semantic_chunk_text(pages_data: List[Tuple[str, int]], embedding_manager: OptimizedEmbeddingManager, similarity_threshold: float = 0.3, max_chunk_size: int = 1500) -> Tuple[List[Dict], np.ndarray]:
    if not pages_data:
        return [], np.array([])
    all_paragraphs = [p for page_text, page_num in pages_data for p in smart_paragraph_split(page_text, page_num)]
    if not all_paragraphs:
        return [], np.array([])
    
    print(f"Starting async-batched chunking on {len(all_paragraphs)} paragraphs...")
    paragraph_texts = [p[0] for p in all_paragraphs]
    all_embeddings = [embedding_manager.encode_batch(paragraph_texts[i:i+256]) for i in range(0, len(paragraph_texts), 256)]
    para_embs = np.vstack(all_embeddings)
    
    sim = np.einsum('ij,ij->i', para_embs[:-1], para_embs[1:])
    chunks: List[Dict] = []
    chunk_embs: List[np.ndarray] = []
    current_chunk_texts, current_chunk_indices, current_page_num = [all_paragraphs[0][0]], [0], all_paragraphs[0][1]
    
    for idx, similarity in enumerate(sim):
        next_paragraph_text, next_page_num = all_paragraphs[idx + 1]
        if similarity > similarity_threshold and sum(len(t) for t in current_chunk_texts) < max_chunk_size and next_page_num == current_page_num:
            current_chunk_texts.append(next_paragraph_text)
            current_chunk_indices.append(idx + 1)
        else:
            chunks.append({"text": "\n\n".join(current_chunk_texts), "metadata": {"page": current_page_num}})
            chunk_embs.append(np.mean(para_embs[current_chunk_indices], axis=0))
            current_chunk_texts, current_chunk_indices, current_page_num = [next_paragraph_text], [idx + 1], next_page_num
            
    if current_chunk_texts:
        chunks.append({"text": "\n\n".join(current_chunk_texts), "metadata": {"page": current_page_num}})
        chunk_embs.append(np.mean(para_embs[current_chunk_indices], axis=0))
        
    print(f"Successfully created {len(chunks)} semantic chunks with pre-computed embeddings.")
    return chunks, np.vstack(chunk_embs)

async def build_enhanced_retrieval_systems(chunks: List[Dict]) -> Optional[DomainQueryExpander]:
    """
    SIMPLIFIED: Builds only the DomainQueryExpander. Multi-vector system is removed.
    """
    print("🔧 Building enhanced retrieval systems (Query Expander)...")
    start_time = time.perf_counter()
    
    if not chunks:
        print("⚠️ No chunks provided to build query expander.")
        return None
        
    all_texts = [chunk['text'] for chunk in chunks]
    
    try:
        loop = asyncio.get_running_loop()
        query_expander = await loop.run_in_executor(
            cpu_executor,
            lambda: DomainQueryExpander(all_texts, min_term_freq=2, max_expansion_terms=4)
        )
        print(f"✅ Query Expander built in {time.perf_counter() - start_time:.2f}s")
        return query_expander
    except Exception as e:
        print(f"⚠️ Query expander build failed: {e}")
        return None