# api/core/vector_store.py
import numpy as np
import faiss
import asyncio
from typing import List, Dict, Tuple
from .embedding_manager import OptimizedEmbeddingManager
from rank_bm25 import BM25Okapi
import re
from concurrent.futures import ThreadPoolExecutor

cpu_executor = ThreadPoolExecutor()

def _tokenize_doc(doc: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z]{3,}\b", doc.lower())

class RequestKnowledgeBase:
    def __init__(self, embedding_manager: OptimizedEmbeddingManager):
        self.manager = embedding_manager
        self.chunks: List[str] = []
        self.faiss_index: faiss.IndexFlatIP = None
        self.bm25_index: BM25Okapi = None
        self.cache: Dict[str, List[str]] = {}

    async def _build_bm25_parallel(self, chunks: List[str]):
        print("Building BM25 index (in parallel)...")
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(cpu_executor, _tokenize_doc, chunk) for chunk in chunks]
        tokenized_corpus = await asyncio.gather(*tasks)
        self.bm25_index = BM25Okapi(tokenized_corpus)

    async def _build_faiss_async(self, chunks: List[str]):
        """Builds FAISS index with async-friendly embedding batches."""
        print("Building FAISS index (with async batching)...")
        
        # Embed chunks in smaller batches, yielding control in between.
        all_embeddings = []
        batch_size = 256
        for i in range(0, len(chunks), batch_size):
            batch_texts = chunks[i:i + batch_size]
            batch_embeddings = self.manager.encode_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
            await asyncio.sleep(0) # Yield control
            
        embeddings = np.vstack(all_embeddings)
        if embeddings.dtype == np.float16:
            embeddings = embeddings.astype(np.float32)
        
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)

    async def build(self, chunks: List[str]):
        if not chunks: raise ValueError("Cannot build knowledge base from empty chunks.")
        self.chunks = chunks
        print(f"Building KB with {len(chunks)} chunks...")
        
        # Run both builds concurrently. Both are now internally parallel/async.
        await asyncio.gather(
            self._build_bm25_parallel(chunks),
            self._build_faiss_async(chunks)
        )
        print(f"✅ KB ready ({self.faiss_index.ntotal} vectors)")

    async def search(self, query: str, k: int = 5, fusion_weights: Tuple[float, float] = None) -> List[str]:
        cache_key = f"{query}_{k}_{fusion_weights}"
        if cache_key in self.cache: return self.cache[cache_key]
        if self.faiss_index is None: raise ValueError("Knowledge base not built yet.")
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._search_sync, query, k, fusion_weights)
        if len(self.cache) > 50: self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = results
        return results

    def _search_sync(self, query: str, k: int, fusion_weights: Tuple[float, float] = None) -> List[str]:
        search_k = min(k * 2, len(self.chunks), 20)
        bm25_results = self._bm25_search(query, search_k)
        faiss_results = self._faiss_search(query, search_k)
        fused = self._fuse_results(bm25_results, faiss_results, fusion_weights)
        return fused[:k]

    def _bm25_search(self, query: str, k: int) -> Dict[int, float]:
        tokens = _tokenize_doc(query)
        if not tokens: return {}
        scores = self.bm25_index.get_scores(tokens)
        if len(scores) <= k: return {i: s for i, s in enumerate(scores) if s > 0}
        top = np.argpartition(scores, -k)[-k:]
        return {i: scores[i] for i in top if scores[i] > 0}

    def _faiss_search(self, query: str, k: int) -> Dict[int, float]:
        emb = self.manager.encode_batch([query])
        if emb.dtype == np.float16: emb = emb.astype(np.float32)
        distances, indices = self.faiss_index.search(emb, k)
        return {idx: float(dist) for idx, dist in zip(indices[0], distances[0]) if idx != -1 and dist > 0}

    def _fuse_results(self, bm25: Dict[int, float], faiss: Dict[int, float], fusion_weights: Tuple[float, float] = None) -> List[str]:
        if bm25: max_b = max(bm25.values()) or 1.0; bm25 = {i: v / max_b for i, v in bm25.items()}
        if faiss: max_f = max(faiss.values()) or 1.0; faiss = {i: v / max_f for i, v in faiss.items()}
        
        # Use dynamic weights if provided, otherwise default
        if fusion_weights is None:
            weights = (0.4, 0.6)  # Default weights (BM25, FAISS)
        else:
            weights = fusion_weights
            
        indices = set(bm25) | set(faiss)
        scored = [(i, weights[0] * bm25.get(i, 0) + weights[1] * faiss.get(i, 0)) for i in indices]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [self.chunks[i] for i, score in scored if score > 0]