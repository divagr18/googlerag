import numpy as np
import faiss
import asyncio
from typing import List, Dict, Tuple
from .embedding_manager import OptimizedEmbeddingManager
from rank_bm25 import BM25Okapi
import re
from concurrent.futures import ThreadPoolExecutor

# Thread pool for CPU-bound tasks
cpu_executor = ThreadPoolExecutor()
# Semaphore to cap concurrent searches
search_sem = asyncio.Semaphore(40)


def _tokenize_doc(doc: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z]{3,}\b", doc.lower())

class RequestKnowledgeBase:
    def __init__(
        self,
        embedding_manager: OptimizedEmbeddingManager,
        use_gpu: bool = True
    ):
        self.manager = embedding_manager
        # --- UPDATED: self.chunks now stores dictionaries ---
        self.chunks: List[Dict] = []
        self.faiss_index = None
        self.bm25_index: BM25Okapi = None
        self.cache: Dict[str, List[Tuple[Dict, float]]] = {}
        self.embed_cache: Dict[str, np.ndarray] = {}
        self.use_gpu = use_gpu

    async def _build_bm25_parallel(self, chunks: List[Dict]):
        loop = asyncio.get_event_loop()
        # Tokenize the 'text' part of each chunk dictionary
        tasks = [loop.run_in_executor(cpu_executor, _tokenize_doc, c['text']) for c in chunks]
        tokenized = await asyncio.gather(*tasks)
        self.bm25_index = BM25Okapi(tokenized)

    async def _build_faiss_async(self, embeddings: np.ndarray):
        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                print("⚠️ FAISS GPU resources unavailable, falling back to CPU index.")
                self.faiss_index = index
        else:
            self.faiss_index = index

    async def build(self, chunks: List[Dict], precomputed_embeddings: np.ndarray):
        if not chunks:
            raise ValueError("Cannot build KB from empty chunks")
        self.chunks = chunks
        await asyncio.gather(
            self._build_bm25_parallel(chunks),
            self._build_faiss_async(precomputed_embeddings)
        )

    # --- UPDATED: Search now returns a list of (chunk_dict, score) tuples ---
    async def search(
        self,
        query: str,
        k: int = 5,
        fusion_weights: Tuple[float, float] = (0.4, 0.6)
    ) -> List[Tuple[Dict, float]]:
        key = f"{query}_{k}_{fusion_weights}"
        if key in self.cache:
            return self.cache[key]

        async with search_sem:
            if query in self.embed_cache:
                q_emb = self.embed_cache[query]
            else:
                q_emb = self.manager.encode_batch([query])
                q_emb = q_emb.astype(np.float32)
                q_emb /= np.maximum(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-12)
                self.embed_cache[query] = q_emb

            tok = _tokenize_doc(query)
            bm25_res = {}
            if tok and self.bm25_index:
                scores = self.bm25_index.get_scores(tok)
                top_idx = np.argpartition(scores, -k)[-k:]
                bm25_res = {i: scores[i] for i in top_idx if scores[i] > 0}

            faiss_k = min(k * 2, len(self.chunks))
            D, I = self.faiss_index.search(q_emb, faiss_k)
            faiss_res = {int(idx): float(score) for idx, score in zip(I[0], D[0]) if idx != -1 and score > 0}

            max_b = max(bm25_res.values()) if bm25_res else 1
            max_f = max(faiss_res.values()) if faiss_res else 1
            combined = {}
            for idx in set(bm25_res) | set(faiss_res):
                b = bm25_res.get(idx, 0) / max_b
                f = faiss_res.get(idx, 0) / max_f
                combined[idx] = fusion_weights[0] * b + fusion_weights[1] * f

            sorted_idx = sorted(combined.items(), key=lambda x: -x[1])
            # Return the chunk dictionary and its score
            results = [(self.chunks[i], combined[i]) for i, _ in sorted_idx[:k]]

            if len(self.cache) > 128:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = results
            return results