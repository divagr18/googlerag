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
        use_gpu: bool = True,
        index_type: str = "ivf_pq",  # options: "ivf_pq" or "hnsw"
        nlist: int = 512,
        m: int = 64,
        hnsw_m: int = 32
    ):
        self.manager = embedding_manager
        self.chunks: List[str] = []
        self.faiss_index = None
        self.bm25_index: BM25Okapi = None
        self.cache: Dict[str, List[str]] = {}
        self.embed_cache: Dict[str, np.ndarray] = {}
        self.use_gpu = use_gpu
        self.index_type = index_type
        # IVF-PQ params
        self.nlist = nlist
        self.m = m
        # HNSW params
        self.hnsw_m = hnsw_m

    async def _build_bm25_parallel(self, chunks: List[str]):
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(cpu_executor, _tokenize_doc, c) for c in chunks]
        tokenized = await asyncio.gather(*tasks)
        self.bm25_index = BM25Okapi(tokenized)

    async def _build_faiss_async(self, chunks: List[str], embeddings: np.ndarray):
        # Normalize & cast
        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)

        dim = embeddings.shape[1]
        num_vecs = embeddings.shape[0]

        # If too few points to train IVF-PQ, fallback to FlatIP
        min_required = self.nlist * 40  # typical rule-of-thumb: 40 vectors per list
        if self.index_type == "ivf_pq" and num_vecs < min_required:
            print(f"⚠️ Only {num_vecs} vectors (<{min_required}), using FlatIP instead of IVF-PQ.")
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
        else:
            if self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(dim, self.hnsw_m)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 50
                index.add(embeddings)
            else:
                # IVF+PQ
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFPQ(quantizer, dim, self.nlist, self.m, 8)
                index.train(embeddings)
                index.add(embeddings)

        # Attempt GPU offload, else keep CPU
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                print("⚠️ FAISS GPU resources unavailable, falling back to CPU index.")
                self.faiss_index = index
        else:
            self.faiss_index = index


    async def build(self, chunks: List[str], precomputed_embeddings: np.ndarray):
        if not chunks:
            raise ValueError("Cannot build KB from empty chunks")
        self.chunks = chunks
        # Build BM25 and FAISS in parallel
        await asyncio.gather(
            self._build_bm25_parallel(chunks),
            self._build_faiss_async(chunks, precomputed_embeddings)
        )

    async def search(
        self,
        query: str,
        k: int = 5,
        fusion_weights: Tuple[float, float] = (0.4, 0.6),
        dynamic_k: bool = True,
        similarity_threshold: float = 0.1
    ) -> List[str]:
        key = f"{query}_{k}_{fusion_weights}_{dynamic_k}"
        if key in self.cache:
            return self.cache[key]

        async with search_sem:
            # Encode or fetch from cache
            if query in self.embed_cache:
                q_emb = self.embed_cache[query]
            else:
                q_emb = self.manager.encode_batch([query])
                q_emb = q_emb.astype(np.float32)
                q_emb /= np.maximum(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-12)
                self.embed_cache[query] = q_emb

            # BM25 search
            tok = _tokenize_doc(query)
            bm25_res = {}
            if tok and self.bm25_index:
                scores = self.bm25_index.get_scores(tok)
                top_idx = np.argpartition(scores, -k)[-k:]
                bm25_res = {i: scores[i] for i in top_idx if scores[i] > 0}

            # FAISS search
            faiss_k = min(k * 2, len(self.chunks))
            D, I = self.faiss_index.search(q_emb, faiss_k)
            faiss_res = {int(idx): float(score) for idx, score in zip(I[0], D[0]) if idx != -1 and score > 0}

            # Fuse scores
            max_b = max(bm25_res.values()) if bm25_res else 1
            max_f = max(faiss_res.values()) if faiss_res else 1
            combined = {}
            for idx in set(bm25_res) | set(faiss_res):
                b = bm25_res.get(idx, 0) / max_b
                f = faiss_res.get(idx, 0) / max_f
                combined[idx] = fusion_weights[0] * b + fusion_weights[1] * f

            # Optional dynamic k: stop early if top score is high
            sorted_idx = sorted(combined.items(), key=lambda x: -x[1])
            if dynamic_k and sorted_idx and sorted_idx[0][1] > similarity_threshold:
                top_n = 1
            else:
                top_n = k
            results = [self.chunks[i] for i, _ in sorted_idx[:top_n]]

            # Cache and return
            if len(self.cache) > 128:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = results
            return results

