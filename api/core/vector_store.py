# api/core/vector_store.py
import numpy as np
import faiss
import asyncio
from typing import List, Dict
from .embedding_manager import OptimizedEmbeddingManager
from rank_bm25 import BM25Okapi

class RequestKnowledgeBase:
    """
    CPU-optimized knowledge base with intelligent caching and memory management.
    All blocking operations are run in a thread pool.
    """
    def __init__(self, embedding_manager: OptimizedEmbeddingManager):
        self.manager = embedding_manager
        self.chunks: List[str] = []
        self.faiss_index: faiss.IndexFlatIP = None
        self.bm25_index: BM25Okapi = None
        self.cache: Dict[str, List[str]] = {}

    async def build(self, chunks: List[str]):
        """Asynchronously builds a CPU-based FAISS index in a thread pool."""
        if not chunks:
            raise ValueError("Cannot build knowledge base from empty chunks.")
        
        self.chunks = chunks
        print(f"Building CPU-optimized KB with {len(chunks)} chunks...")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._build_sync, chunks)

        print(f"✅ CPU-optimized KB ready ({self.faiss_index.ntotal} vectors)")

    def _build_sync(self, chunks: List[str]):
        """Synchronous helper for building indexes."""
        # Build BM25 index
        print("Building BM25 index...")
        tokenized_corpus = [self._fast_tokenize(doc) for doc in chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # Build CPU FAISS index
        print("Building CPU FAISS index...")
        embeddings = self.manager.encode_batch(chunks)
        
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)
        del embeddings

    async def search(self, query: str, k: int = 5) -> List[str]:
        """Asynchronously performs a search and uses an efficient FIFO cache."""
        cache_key = f"{query}_{k}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.faiss_index is None:
            raise ValueError("Knowledge base not built yet.")

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._search_sync, query, k)

        # FIFO cache eviction
        if len(self.cache) > 50:
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = results
        return results

    def _search_sync(self, query: str, k: int) -> List[str]:
        """Synchronous helper for searching."""
        search_k = min(k * 2, len(self.chunks), 20)
        bm25_results = self._bm25_search(query, search_k)
        faiss_results = self._faiss_search(query, search_k)
        
        fused = self._fuse_results(bm25_results, faiss_results)
        return fused[:k]

    def _fast_tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

    def _bm25_search(self, query: str, k: int) -> Dict[int, float]:
        tokens = self._fast_tokenize(query)
        if not tokens: return {}
        scores = self.bm25_index.get_scores(tokens)
        if len(scores) <= k: return {i: s for i, s in enumerate(scores) if s > 0}
        top = np.argpartition(scores, -k)[-k:]
        return {i: scores[i] for i in top if scores[i] > 0}

    def _faiss_search(self, query: str, k: int) -> Dict[int, float]:
        emb = self.manager.encode_batch([query])
        distances, indices = self.faiss_index.search(emb, k)
        return {idx: float(dist) for idx, dist in zip(indices[0], distances[0]) if idx != -1 and dist > 0}

    def _fuse_results(self, bm25: Dict[int, float], faiss: Dict[int, float]) -> List[str]:
        # Normalize
        if bm25: max_b = max(bm25.values()) or 1.0; bm25 = {i: v / max_b for i, v in bm25.items()}
        if faiss: max_f = max(faiss.values()) or 1.0; faiss = {i: v / max_f for i, v in faiss.items()}
        
        # Weighting
        weights = (0.4, 0.6)
        indices = set(bm25) | set(faiss)
        scored = []
        for i in indices:
            score = weights[0] * bm25.get(i, 0) + weights[1] * faiss.get(i, 0)
            if score > 0: scored.append((i, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [self.chunks[i] for i, _ in scored]

    def __del__(self):
        pass