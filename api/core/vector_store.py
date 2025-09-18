# api/core/vector_store.py

import numpy as np
import faiss
import asyncio
from typing import List, Dict, Optional, Tuple
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
        self, embedding_manager: OptimizedEmbeddingManager, use_gpu: bool = True
    ):
        self.manager = embedding_manager
        self.chunks: List[Dict] = []
        self.faiss_index = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.cache: Dict[str, List[Tuple[Dict, float]]] = {}
        self.embed_cache: Dict[str, np.ndarray] = {}
        self.use_gpu = use_gpu

    async def _build_bm25_parallel(self, chunks: List[Dict]):
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(cpu_executor, _tokenize_doc, c["text"]) for c in chunks
        ]
        tokenized = await asyncio.gather(*tasks)
        non_empty_docs = [doc for doc in tokenized if len(doc) > 0]
        if non_empty_docs:
            self.bm25_index = BM25Okapi(non_empty_docs)
        else:
            self.bm25_index = None

    async def _build_faiss_async(self, embeddings: np.ndarray):
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

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
            self._build_faiss_async(precomputed_embeddings),
        )

    async def search(
        self, query: str, k: int = 5, fusion_weights: Tuple[float, float] = (0.4, 0.6)
    ) -> List[Tuple[Dict, float]]:
        if not self.chunks or not self.faiss_index or not self.bm25_index:
            return []

        effective_k = min(k, len(self.chunks))
        key = f"{query}_{effective_k}_{fusion_weights}"
        if key in self.cache:
            return self.cache[key]

        async with search_sem:
            if query in self.embed_cache:
                q_emb = self.embed_cache[query]
            else:
                q_emb = self.manager.encode_batch([query])
                q_emb = q_emb.astype(np.float32)
                faiss.normalize_L2(q_emb)
                self.embed_cache[query] = q_emb

            tok = _tokenize_doc(query)
            bm25_res = {}
            if tok:
                scores = self.bm25_index.get_scores(tok)
                top_idx = np.argpartition(scores, -effective_k)[-effective_k:]
                bm25_res = {i: scores[i] for i in top_idx if scores[i] > 0}

            faiss_k = min(effective_k * 2, len(self.chunks))
            D, I = self.faiss_index.search(q_emb, faiss_k)
            faiss_res = {
                int(idx): float(score)
                for idx, score in zip(I[0], D[0])
                if idx != -1 and score > 0
            }

            max_b = max(bm25_res.values()) if bm25_res else 1
            max_f = max(faiss_res.values()) if faiss_res else 1
            combined = {}
            for idx in set(bm25_res) | set(faiss_res):
                b = bm25_res.get(idx, 0) / max_b
                f = faiss_res.get(idx, 0) / max_f
                combined[idx] = fusion_weights[0] * b + fusion_weights[1] * f

            sorted_idx = sorted(combined.items(), key=lambda x: -x[1])
            results = [
                (self.chunks[i], combined[i]) for i, _ in sorted_idx[:effective_k]
            ]

            if len(self.cache) > 128:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = results
            return results
