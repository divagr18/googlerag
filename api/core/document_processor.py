# api/core/document_processor.py
import io
import os
import re
import fitz
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List
from urllib.parse import urlparse
import numpy as np
from .embedding_manager import OptimizedEmbeddingManager

cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count())

# --- Document Downloading and Parsing (Unchanged) ---
# ... (all functions from download_document to smart_paragraph_split are the same) ...
async def download_document(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, follow_redirects=True, timeout=60.0)
            response.raise_for_status()
            return response.content
        except httpx.RequestError as e:
            raise ValueError(f"Error downloading document from {url}: {e}")

def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

def extract_text_from_document_bytes(document_bytes: bytes, file_type: str) -> str:
    if file_type == '.pdf':
        return _extract_text_from_pdf(document_bytes)
    else:
        return document_bytes.decode('utf-8', errors='ignore')

async def process_document(url: str, document_bytes: bytes) -> str:
    path = urlparse(url).path
    file_type = os.path.splitext(path)[1].lower()
    if not file_type:
        raise ValueError("Could not determine file type from URL.")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        cpu_executor, extract_text_from_document_bytes, document_bytes, file_type
    )

def smart_paragraph_split(text: str) -> List[str]:
    paragraphs = text.split('\n\n')
    result = []
    for para in paragraphs:
        para = para.strip()
        if len(para) < 50:
            if result and len(result[-1]) < 500: result[-1] += ' ' + para
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
            if current: result.append(' '.join(current))
        else:
            result.append(para)
    return [p for p in result if p]


# --- NEW: Async-Native Batching for Ultra-Fast Chunking ---
async def optimized_semantic_chunk_text(
    text: str, 
    embedding_manager: OptimizedEmbeddingManager, 
    similarity_threshold: float = 0.6,
    min_chunk_size: int = 200,
    max_chunk_size: int = 2000
) -> List[str]:
    """
    Blazing-fast chunking by processing embeddings in async-friendly batches.
    """
    if not text.strip(): return []
    
    paragraphs = smart_paragraph_split(text)
    if not paragraphs: return []
    
    print(f"Starting async-batched chunking on {len(paragraphs)} paragraphs...")

    # 1. Embed paragraphs in smaller, async-friendly batches.
    all_embeddings = []
    batch_size = 256  # A good batch size for a 4060
    for i in range(0, len(paragraphs), batch_size):
        batch_texts = paragraphs[i:i + batch_size]
        # This is still a sync call, but it's much shorter.
        batch_embeddings = embedding_manager.encode_batch(batch_texts)
        all_embeddings.append(batch_embeddings)
        # CRITICAL: Yield control to the event loop after each batch.
        await asyncio.sleep(0)
    
    embeddings = np.vstack(all_embeddings)
    
    # 2. The numpy similarity calculation is already very fast, so we keep it.
    similarities = np.einsum('ij,ij->i', embeddings[:-1], embeddings[1:])
    
    # 3. Grouping logic is unchanged as it's already fast.
    chunks = []
    current_chunk = [paragraphs[0]]
    for i, similarity in enumerate(similarities):
        if similarity > similarity_threshold and sum(len(p) for p in current_chunk) < max_chunk_size:
            current_chunk.append(paragraphs[i+1])
        else:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraphs[i+1]]
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # 4. Post-processing is unchanged.
    final_chunks = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        if len(chunk) < min_chunk_size and i < len(chunks) - 1:
            final_chunks.append(chunk + "\n\n" + chunks[i+1])
            i += 2
        else:
            final_chunks.append(chunk)
            i += 1

    print(f"Successfully created {len(final_chunks)} semantic chunks.")
    return final_chunks