# api/core/document_processor.py
import io
import os
import re
import fitz
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, List, Tuple
from urllib.parse import urlparse
import numpy as np
from .embedding_manager import OptimizedEmbeddingManager

cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count())
async def stream_document(url: str) -> AsyncIterator[bytes]:
    """
    Streams a document's content chunk by chunk instead of downloading it all at once.
    """
    async with httpx.AsyncClient() as client:
        try:
            async with client.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.RequestError as e:
            raise ValueError(f"Error streaming document from {url}: {e}")

# --- MODIFIED: PDF Extraction from a Stream ---
def _extract_text_from_pdf_stream(pdf_stream: io.BytesIO) -> str:
    """
    Extracts text from a PDF provided as a byte stream.
    This is a synchronous function to be run in a thread pool.
    """
    # PyMuPDF can open a file-like object (our in-memory stream)
    with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

# --- MODIFIED: Main Processing function to handle the stream ---
async def process_document_stream(url: str, document_iterator: AsyncIterator[bytes]) -> str:
    """
    Processes a document from an async iterator of bytes, assembling it in memory
    and then passing it to the text extractor.
    """
    path = urlparse(url).path
    file_type = os.path.splitext(path)[1].lower()
    if not file_type:
        raise ValueError("Could not determine file type from URL.")

    # Assemble the streamed chunks into an in-memory byte stream
    document_bytes_io = io.BytesIO()
    async for chunk in document_iterator:
        document_bytes_io.write(chunk)
    
    # The stream is now complete, move the cursor to the beginning
    document_bytes_io.seek(0)

    loop = asyncio.get_event_loop()
    if file_type == '.pdf':
        # Run the synchronous stream-based PDF parser in a thread
        return await loop.run_in_executor(
            cpu_executor, _extract_text_from_pdf_stream, document_bytes_io
        )
    else:
        # For other file types, just decode
        return document_bytes_io.read().decode('utf-8', errors='ignore')
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
) -> Tuple[List[str], np.ndarray]: # <--- MODIFIED RETURN TYPE
    """
    Blazing-fast chunking that returns both text chunks and their pre-computed embeddings.
    """
    if not text.strip(): return [], np.array([])
    
    paragraphs = smart_paragraph_split(text)
    if not paragraphs: return [], np.array([])
    
    print(f"Starting async-batched chunking on {len(paragraphs)} paragraphs...")

    all_embeddings = []
    batch_size = 256
    for i in range(0, len(paragraphs), batch_size):
        batch_texts = paragraphs[i:i + batch_size]
        batch_embeddings = embedding_manager.encode_batch(batch_texts)
        all_embeddings.append(batch_embeddings)
        await asyncio.sleep(0)
    
    paragraph_embeddings = np.vstack(all_embeddings)
    
    similarities = np.einsum('ij,ij->i', paragraph_embeddings[:-1], paragraph_embeddings[1:])
    
    # --- MODIFIED: Group chunks AND their embeddings simultaneously ---
    chunks = []
    chunk_embeddings = []
    current_chunk_texts = [paragraphs[0]]
    current_chunk_indices = [0]

    for i, similarity in enumerate(similarities):
        if similarity > similarity_threshold and sum(len(p) for p in current_chunk_texts) < max_chunk_size:
            current_chunk_texts.append(paragraphs[i+1])
            current_chunk_indices.append(i+1)
        else:
            # Finalize the current chunk
            chunks.append("\n\n".join(current_chunk_texts))
            # Average the embeddings for this chunk
            avg_embedding = np.mean(paragraph_embeddings[current_chunk_indices], axis=0)
            chunk_embeddings.append(avg_embedding)
            
            # Start a new chunk
            current_chunk_texts = [paragraphs[i+1]]
            current_chunk_indices = [i+1]

    if current_chunk_texts:
        chunks.append("\n\n".join(current_chunk_texts))
        avg_embedding = np.mean(paragraph_embeddings[current_chunk_indices], axis=0)
        chunk_embeddings.append(avg_embedding)

    # Post-processing (merging small chunks) is complex with pre-computed embeddings.
    # For performance, we can accept slightly smaller chunks or simplify the logic.
    # Let's skip the merge for now as the performance gain from avoiding re-embedding is much larger.
    
    print(f"Successfully created {len(chunks)} semantic chunks with pre-computed embeddings.")
    return chunks, np.array(chunk_embeddings)