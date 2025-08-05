# api/core/document_processor.py

import io
import os
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, List, Tuple, Dict
from urllib.parse import urlparse
import numpy as np
import fitz  # PyMuPDF
import httpx
from .embedding_manager import OptimizedEmbeddingManager
# --- NEW: Import the docx library ---
import docx
from httpx import AsyncHTTPTransport

# Thread pool for CPU-bound tasks like PDF page parsing
cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

async def stream_document(url: str) -> AsyncIterator[bytes]:
    transport = AsyncHTTPTransport(retries=3)
    limits = httpx.Limits(max_connections=4)
    async with httpx.AsyncClient(limits=limits, transport=transport) as client:
        try:
            async with client.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.RequestError as e:
            raise ValueError(f"Error streaming document from {url} after retries: {e}")

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
    pages_data = await asyncio.gather(*tasks)
    return pages_data

# --- NEW: Helper function to extract text from a DOCX file stream ---
def _extract_text_from_docx_stream(docx_stream: io.BytesIO) -> List[Tuple[str, int]]:
    """
    Extracts text from a DOCX file stream. All text is assigned to page 1.
    """
    try:
        document = docx.Document(docx_stream)
        full_text = "\n".join([para.text for para in document.paragraphs])
        # DOCX files are not paginated, so we return all text as a single "page".
        return [(full_text, 1)]
    except Exception as e:
        raise ValueError(f"Failed to process DOCX file: {e}")

async def process_document_stream(
    url: str,
    document_iterator: AsyncIterator[bytes]
) -> List[Tuple[str, int]]:
    """
    Processes a document from a stream, returning a list of (text, page_number) tuples.
    Handles PDF, DOCX, and plain text. Rejects other types.
    """
    path = urlparse(url).path
    # Use a default extension of .txt if none is found
    file_type = os.path.splitext(path)[1].lower() or ".txt"

    document_bytes_io = io.BytesIO()
    async for chunk in document_iterator:
        document_bytes_io.write(chunk)
    document_bytes_io.seek(0)

    # --- UPDATED: Added logic for .docx and explicit rejection of other types ---
    if file_type == '.pdf':
        return await _extract_text_from_pdf_stream(document_bytes_io)
    elif file_type == '.docx':
        # Run the synchronous docx parsing in a thread to avoid blocking
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            cpu_executor, _extract_text_from_docx_stream, document_bytes_io
        )
    elif file_type in ['.txt', '.md', '.csv']:
        # For plain text, return all content with page number 1
        text = document_bytes_io.read().decode('utf-8', errors='ignore')
        return [(text, 1)]
    else:
        # Explicitly reject unsupported file types
        raise ValueError(f"Unsupported file type: '{file_type}'. Please provide a PDF, DOCX, or TXT file.")

# ... (rest of the file is unchanged) ...
def smart_paragraph_split(text: str, page_num: int) -> List[Tuple[str, int]]:
    paragraphs = text.split('\n\n')
    result: List[Tuple[str, int]] = []
    for para in paragraphs:
        para = para.strip()
        if len(para) > 50:
            result.append((para, page_num))
    return result

async def optimized_semantic_chunk_text(
    pages_data: List[Tuple[str, int]],
    embedding_manager: OptimizedEmbeddingManager,
    similarity_threshold: float = 0.3,
    max_chunk_size: int = 1500,
) -> Tuple[List[Dict], np.ndarray]:
    if not pages_data:
        return [], np.array([])
    all_paragraphs = []
    for page_text, page_num in pages_data:
        all_paragraphs.extend(smart_paragraph_split(page_text, page_num))
    if not all_paragraphs:
        return [], np.array([])
    print(f"Starting async-batched chunking on {len(all_paragraphs)} paragraphs...")
    paragraph_texts = [p[0] for p in all_paragraphs]
    all_embeddings: List[np.ndarray] = []
    batch_size = 256
    for i in range(0, len(paragraph_texts), batch_size):
        batch = paragraph_texts[i:i + batch_size]
        emb = embedding_manager.encode_batch(batch)
        all_embeddings.append(emb)
        await asyncio.sleep(0)
    para_embs = np.vstack(all_embeddings)
    sim = np.einsum('ij,ij->i', para_embs[:-1], para_embs[1:])
    chunks: List[Dict] = []
    chunk_embs: List[np.ndarray] = []
    current_chunk_texts = [all_paragraphs[0][0]]
    current_chunk_indices = [0]
    current_page_num = all_paragraphs[0][1]
    for idx, similarity in enumerate(sim):
        next_paragraph_text, next_page_num = all_paragraphs[idx + 1]
        if similarity > similarity_threshold and sum(len(t) for t in current_chunk_texts) < max_chunk_size and next_page_num == current_page_num:
            current_chunk_texts.append(next_paragraph_text)
            current_chunk_indices.append(idx + 1)
        else:
            chunk_text = "\n\n".join(current_chunk_texts)
            avg_emb = np.mean(para_embs[current_chunk_indices], axis=0)
            chunks.append({"text": chunk_text, "metadata": {"page": current_page_num}})
            chunk_embs.append(avg_emb)
            current_chunk_texts = [next_paragraph_text]
            current_chunk_indices = [idx + 1]
            current_page_num = next_page_num
    if current_chunk_texts:
        chunk_text = "\n\n".join(current_chunk_texts)
        avg_emb = np.mean(para_embs[current_chunk_indices], axis=0)
        chunks.append({"text": chunk_text, "metadata": {"page": current_page_num}})
        chunk_embs.append(avg_emb)
    print(f"Successfully created {len(chunks)} semantic chunks with pre-computed embeddings.")
    return chunks, np.vstack(chunk_embs)