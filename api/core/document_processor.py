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
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from .embedding_manager import OptimizedEmbeddingManager # Import the manager

# A single thread pool for all CPU-bound parsing tasks
cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count())

async def download_document(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, follow_redirects=True, timeout=30.0)
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
    # Add other file types like .docx if needed
    else:
        # Fallback for .txt or other text-based formats
        return document_bytes.decode('utf-8', errors='ignore')

async def process_document(url: str, document_bytes: bytes) -> str:
    path = urlparse(url).path
    file_type = os.path.splitext(path)[1].lower()
    if not file_type:
        raise ValueError("Could not determine file type from URL.")

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        cpu_executor, 
        extract_text_from_document_bytes, 
        document_bytes, 
        file_type
    )

# --- UNIFIED SEMANTIC CHUNKING STRATEGY ---

def smart_paragraph_split(text: str) -> List[str]:
    """Intelligent paragraph splitting optimized for legal/financial docs"""
    paragraphs = text.split('\n\n')
    result = []
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
                    current.append(sent)
                    current_len += len(sent)
            if current:
                result.append(' '.join(current))
        else:
            result.append(para)
    return result

def _cluster_and_chunk(
    paragraphs: List[str], 
    embedding_model: OptimizedEmbeddingManager, 
    target_chunk_size: int
) -> List[str]:
    """
    Synchronous helper function for clustering. To be run in an executor.
    """
    if not paragraphs:
        return []

    print(f"Starting semantic chunking on {len(paragraphs)} paragraphs...")
    
    # Embed all paragraphs using the centralized manager
    paragraph_embeddings = embedding_model.encode_batch(paragraphs)
    
    # Cluster paragraphs to form semantic chunks
    total_chars = sum(len(p) for p in paragraphs)
    n_clusters = max(1, round(total_chars / target_chunk_size))
    n_clusters = min(n_clusters, len(paragraphs))

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init='auto', batch_size=256)
    kmeans.fit(paragraph_embeddings)
    
    clustered_paragraphs = [[] for _ in range(n_clusters)]
    for i, label in enumerate(kmeans.labels_):
        clustered_paragraphs[label].append(paragraphs[i])
        
    chunks = ['\n\n'.join(cluster).strip() for cluster in clustered_paragraphs if cluster]
    
    final_chunks = [f"search_document: {chunk}" for chunk in chunks if chunk]
    print(f"Successfully created {len(final_chunks)} semantic chunks.")
    return final_chunks

async def optimized_semantic_chunk_text(
    text: str, 
    embedding_model: OptimizedEmbeddingManager, 
    target_chunk_size: int = 1500
) -> List[str]:
    """
    This is the single, unified chunking strategy.
    It runs CPU-bound ML clustering in a thread pool to avoid blocking.
    """
    if not text.strip():
        return []
    
    paragraphs = smart_paragraph_split(text)
    if not paragraphs:
        return []

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        cpu_executor,
        _cluster_and_chunk,
        paragraphs,
        embedding_model,
        target_chunk_size
    )