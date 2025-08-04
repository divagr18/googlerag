# api/core/document_processor.py

import io
import os
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, List, Tuple
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

# --- NEW: In-Memory PDF Page Extraction Helper ---
def _extract_page_text_from_bytes(args: Tuple[bytes, int]) -> str:
    """
    Opens a PDF from bytes in memory and extracts text from a single page.
    This is thread-safe as each thread gets its own copy of the bytes.
    """
    pdf_bytes, page_no = args
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return doc.load_page(page_no).get_text()
    except Exception as e:
        # Log or handle page-specific extraction errors if necessary
        print(f"Error extracting page {page_no}: {e}")
        return ""

# --- REVISED: In-Memory Parallel PDF Extraction ---
async def _extract_text_from_pdf_stream(pdf_stream: io.BytesIO) -> str:
    """
    Extracts text from a PDF stream in parallel using a thread pool,
    operating entirely in memory to avoid disk I/O bottlenecks.
    """
    pdf_bytes = pdf_stream.getvalue()
    
    # Open the document once from memory to get the page count
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            page_count = doc.page_count
    except Exception as e:
        raise ValueError(f"Failed to open PDF from memory stream: {e}")

    if page_count == 0:
        return ""

    loop = asyncio.get_running_loop()
    # Create tasks that pass the raw bytes and page number to each worker
    tasks = [
        loop.run_in_executor(cpu_executor, _extract_page_text_from_bytes, (pdf_bytes, i))
        for i in range(page_count)
    ]
    pages_text = await asyncio.gather(*tasks)
    
    return "".join(pages_text)

async def process_document_stream(
    url: str,
    document_iterator: AsyncIterator[bytes]
) -> str:
    """
    Processes a document from an async iterator of bytes, assembling it in memory
    and then passing it to the appropriate extractor.
    """
    path = urlparse(url).path
    file_type = os.path.splitext(path)[1].lower()
    if not file_type:
        # Fallback for URLs without extensions
        file_type = ".pdf" # Assume PDF if no extension is found in URL path

    # Assemble streamed chunks into an in-memory byte stream
    document_bytes_io = io.BytesIO()
    async for chunk in document_iterator:
        document_bytes_io.write(chunk)
    document_bytes_io.seek(0)

    if file_type == '.pdf':
        # Use the new, faster in-memory parallel extraction
        return await _extract_text_from_pdf_stream(document_bytes_io)
    else:
        # For other file types, just decode
        return document_bytes_io.read().decode('utf-8', errors='ignore')

def smart_paragraph_split(text: str) -> List[str]:
    paragraphs = text.split('\n\n')
    result: List[str] = []
    for para in paragraphs:
        para = para.strip()
        if len(para) < 50:
            if result and len(result[-1]) < 500:
                result[-1] += ' ' + para
            continue
        if len(para) > 2000:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current, current_len = [], 0
            for sent in sentences:
                if current_len + len(sent) > 1500 and current:
                    result.append(' '.join(current))
                    current, current_len = [sent], len(sent)
                else:
                    current.append(sent); current_len += len(sent)
            if current:
                result.append(' '.join(current))
        else:
            result.append(para)
    return [p for p in result if p]

# --- Async-Native Batching for Ultra-Fast Chunking ---
async def optimized_semantic_chunk_text(
    text: str,
    embedding_manager: OptimizedEmbeddingManager,
    similarity_threshold: float = 0.3,
    max_chunk_size: int = 1500,
) -> Tuple[List[str], np.ndarray]:
    """
    Blazing-fast chunking that returns both text chunks and their pre-computed embeddings.
    """
    if not text.strip():
        return [], np.array([])
    paragraphs = smart_paragraph_split(text)
    if not paragraphs:
        return [], np.array([])

    print(f"Starting async-batched chunking on {len(paragraphs)} paragraphs...")

    all_embeddings: List[np.ndarray] = []
    batch_size = 256
    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i + batch_size]
        emb = embedding_manager.encode_batch(batch)
        all_embeddings.append(emb)
        await asyncio.sleep(0)
    para_embs = np.vstack(all_embeddings)

    sim = np.einsum('ij,ij->i', para_embs[:-1], para_embs[1:])

    chunks: List[str] = []
    chunk_embs: List[np.ndarray] = []
    cur_texts = [paragraphs[0]]
    cur_idxs = [0]

    for idx, similarity in enumerate(sim):
        if similarity > similarity_threshold and sum(len(t) for t in cur_texts) < max_chunk_size:
            cur_texts.append(paragraphs[idx+1])
            cur_idxs.append(idx+1)
        else:
            chunks.append("\n\n".join(cur_texts))
            avg_emb = np.mean(para_embs[cur_idxs], axis=0)
            chunk_embs.append(avg_emb)
            cur_texts, cur_idxs = [paragraphs[idx+1]], [idx+1]

    if cur_texts:
        chunks.append("\n\n".join(cur_texts))
        avg_emb = np.mean(para_embs[cur_idxs], axis=0)
        chunk_embs.append(avg_emb)

    print(f"Successfully created {len(chunks)} semantic chunks with pre-computed embeddings.")
    return chunks, np.vstack(chunk_embs)