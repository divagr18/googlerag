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
    """
    FIX: Runs synchronous, CPU-bound parsing in a thread pool to prevent
    blocking the async event loop.
    """
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
    # Split by double newlines, then filter for meaningful paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
    return paragraphs

def cluster_paragraphs(paragraphs: List[str], embeddings: np.ndarray, target_chunk_size: int) -> List[str]:
    if not paragraphs:
        return []
    
    total_chars = sum(len(p) for p in paragraphs)
    n_clusters = max(1, round(total_chars / target_chunk_size))
    
    # Ensure n_clusters is not more than the number of paragraphs
    n_clusters = min(n_clusters, len(paragraphs))

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init='auto', batch_size=256)
    kmeans.fit(embeddings)
    
    # Group paragraphs by their assigned cluster
    clustered_paragraphs = [[] for _ in range(n_clusters)]
    for i, label in enumerate(kmeans.labels_):
        clustered_paragraphs[label].append(paragraphs[i])
        
    # Join paragraphs in each cluster to form the final chunks
    chunks = ['\n\n'.join(cluster).strip() for cluster in clustered_paragraphs if cluster]
    return chunks

async def optimized_semantic_chunk_text(text: str, embedding_model, target_chunk_size: int = 1500) -> List[str]:
    """
    This is the single, unified chunking strategy for the entire application.
    It uses ML-based clustering for semantic coherence.
    """
    if not text.strip():
        return []
    
    paragraphs = smart_paragraph_split(text)
    if not paragraphs:
        return []

    print(f"Starting semantic chunking on {len(paragraphs)} paragraphs...")
    
    # Embed all paragraphs in one go for efficiency
    paragraph_embeddings = embedding_model.encode(
        paragraphs,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    
    # Cluster paragraphs to form semantic chunks
    chunks = cluster_paragraphs(paragraphs, paragraph_embeddings, target_chunk_size)
    
    final_chunks = [f"search_document: {chunk}" for chunk in chunks if chunk]
    print(f"Successfully created {len(final_chunks)} semantic chunks.")
    return final_chunks