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

# Thread pool for CPU-bound tasks like PDF page parsing
cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

async def stream_document(url: str) -> AsyncIterator[bytes]:
    """
    Streams a document's content chunk by chunk instead of downloading it all at once.
    """
    limits = httpx.Limits(max_connections=4)
    async with httpx.AsyncClient(limits=limits) as client:
        try:
            async with client.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.RequestError as e:
            raise ValueError(f"Error streaming document from {url}: {e}")

# --- UPDATED: In-Memory PDF Page Extraction Helper ---
def _extract_page_text_from_bytes(args: Tuple[bytes, int]) -> Tuple[str, int]:
    """
    Opens a PDF from bytes in memory and extracts text and page number from a single page.
    """
    pdf_bytes, page_no = args
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            # Return text and the original page number (1-based for readability)
            return doc.load_page(page_no).get_text(), page_no + 1
    except Exception as e:
        print(f"Error extracting page {page_no}: {e}")
        return "", page_no + 1

# --- UPDATED: In-Memory Parallel PDF Extraction ---
async def _extract_text_from_pdf_stream(pdf_stream: io.BytesIO) -> List[Tuple[str, int]]:
    """
    Extracts text and page numbers from a PDF stream in parallel.
    Returns a list of (page_text, page_number) tuples.
    """
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
    # pages_data is now a list of (text, page_num)
    pages_data = await asyncio.gather(*tasks)
    return pages_data

async def process_document_stream(
    url: str,
    document_iterator: AsyncIterator[bytes]
) -> List[Tuple[str, int]]:
    """
    Processes a document from a stream, returning a list of (text, page_number) tuples.
    For non-PDFs, all text is assigned to page 1.
    """
    path = urlparse(url).path
    file_type = os.path.splitext(path)[1].lower() or ".pdf"

    document_bytes_io = io.BytesIO()
    async for chunk in document_iterator:
        document_bytes_io.write(chunk)
    document_bytes_io.seek(0)

    if file_type == '.pdf':
        return await _extract_text_from_pdf_stream(document_bytes_io)
    else:
        # For plain text, return all content with page number 1
        text = document_bytes_io.read().decode('utf-8', errors='ignore')
        return [(text, 1)]

def smart_paragraph_split(text: str, page_num: int) -> List[Tuple[str, int]]:
    """Splits text into paragraphs and retains the page number for each."""
    paragraphs = text.split('\n\n')
    result: List[Tuple[str, int]] = []
    # Simplified logic for clarity, can be enhanced as before
    for para in paragraphs:
        para = para.strip()
        if len(para) > 50: # Simple filter for meaningful paragraphs
            result.append((para, page_num))
    return result

# --- UPDATED: Async-Native Batching to handle metadata ---
async def optimized_semantic_chunk_text(
    pages_data: List[Tuple[str, int]],
    embedding_manager: OptimizedEmbeddingManager,
    similarity_threshold: float = 0.3,
    max_chunk_size: int = 1500,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Chunks text and returns a list of chunk dictionaries (with metadata) and their embeddings.
    """
    if not pages_data:
        return [], np.array([])

    # Flatten pages into paragraphs with associated page numbers
    all_paragraphs = []
    for page_text, page_num in pages_data:
        all_paragraphs.extend(smart_paragraph_split(page_text, page_num))

    if not all_paragraphs:
        return [], np.array([])

    print(f"Starting async-batched chunking on {len(all_paragraphs)} paragraphs...")
    
    paragraph_texts = [p[0] for p in all_paragraphs]
    
    # Batch encode all paragraph texts
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
    # Start with the page number of the first paragraph
    current_page_num = all_paragraphs[0][1]

    for idx, similarity in enumerate(sim):
        next_paragraph_text, next_page_num = all_paragraphs[idx + 1]
        
        # Combine if similarity is high, chunk size is not exceeded, AND they are on the same page
        if similarity > similarity_threshold and sum(len(t) for t in current_chunk_texts) < max_chunk_size and next_page_num == current_page_num:
            current_chunk_texts.append(next_paragraph_text)
            current_chunk_indices.append(idx + 1)
        else:
            # Finalize the current chunk
            chunk_text = "\n\n".join(current_chunk_texts)
            avg_emb = np.mean(para_embs[current_chunk_indices], axis=0)
            chunks.append({"text": chunk_text, "metadata": {"page": current_page_num}})
            chunk_embs.append(avg_emb)
            
            # Start a new chunk
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