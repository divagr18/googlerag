import numpy as np
import faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import torch
import gc

class RequestKnowledgeBase:
    """
    RTX 4060 optimized knowledge base with intelligent caching and memory management
    """
    def __init__(self, embedding_model):
        self.model = embedding_model
        self.chunks: List[str] = []
        self.faiss_index = None
        self.bm25_index = None
        
        self.use_gpu = torch.cuda.is_available()
        self.gpu_resources = None
        self.cache = {}
        
        if self.use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()
            self.gpu_resources.setTempMemory(256 * 1024 * 1024)
            self.gpu_resources.setCacheMemory(512 * 1024 * 1024)
            torch.cuda.empty_cache()

    def build(self, chunks: List[str]):
        if not chunks:
            raise ValueError("Cannot build knowledge base from empty chunks.")
        
        self.chunks = chunks
        print(f"Building RTX 4060 optimized KB with {len(chunks)} chunks...")

        print("Building BM25 index...")
        tokenized_corpus = [self._fast_tokenize(doc) for doc in chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        print("Building GPU-accelerated FAISS index...")
        embeddings = self._batch_encode_optimized(chunks)
        dimension = embeddings.shape[1]
        
        if self.use_gpu and len(chunks) > 100:
            self.faiss_index = self._build_gpu_hnsw_index(embeddings, dimension)
        elif self.use_gpu:
            self.faiss_index = self._build_gpu_flat_index(embeddings, dimension)
        else:
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(embeddings)
            
        del embeddings
        if self.use_gpu:
            torch.cuda.empty_cache()
        print(f"✅ RTX 4060 optimized KB ready ({self.faiss_index.ntotal} vectors)")

    def _batch_encode_optimized(self, texts: List[str]) -> np.ndarray:
        batch_size = 24 if len(texts) > 100 else 16
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch, batch_size=len(batch), show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
                device='cuda' if self.use_gpu else 'cpu'
            )
            all_embeddings.append(batch_embeddings)
            if self.use_gpu and i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
        return np.vstack(all_embeddings)

    def _build_gpu_flat_index(self, embeddings: np.ndarray, dimension: int) -> faiss.GpuIndex:
        config = faiss.GpuIndexFlatConfig()
        config.useFloat16 = True
        config.device = 0
        gpu_index = faiss.GpuIndexFlatIP(self.gpu_resources, dimension, config)
        gpu_index.add(embeddings)
        return gpu_index

    def _build_gpu_hnsw_index(self, embeddings: np.ndarray, dimension: int) -> faiss.Index:
        hnsw_index = faiss.IndexHNSWFlat(dimension, 32)
        hnsw_index.hnsw.efConstruction = 64
        hnsw_index.add(embeddings)
        if self.use_gpu and len(embeddings) > 500:
            try:
                return faiss.index_cpu_to_gpu(self.gpu_resources, 0, hnsw_index)
            except:
                return hnsw_index
        return hnsw_index

    def search(self, query: str, k: int = 5) -> List[str]:
        cache_key = f"{query}_{k}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if self.faiss_index is None or self.bm25_index is None:
            raise ValueError("Knowledge base not built yet.")

        search_k = min(k * 2, len(self.chunks), 20)
        bm25_results = self._bm25_search_fast(query, search_k)
        faiss_results = self._faiss_search_fast(query, search_k)
        
        fused_results = self._fast_fusion(bm25_results, faiss_results, query)
        final_results = fused_results[:k]
        
        # Corrected: Use FIFO cache eviction instead of clearing the whole cache
        if len(self.cache) > 50:
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = final_results
        
        return final_results

    def _fast_tokenize(self, text: str) -> List[str]:
        import re
        text = re.sub(r'^search_document:\s*', '', text)
        return re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    def _bm25_search_fast(self, query: str, k: int) -> Dict[int, float]:
        query_tokens = self._fast_tokenize(query)
        if not query_tokens: return {}
        scores = self.bm25_index.get_scores(query_tokens)
        if len(scores) <= k:
            return {i: score for i, score in enumerate(scores) if score > 0}
        top_indices = np.argpartition(scores, -k)[-k:]
        return {idx: scores[idx] for idx in top_indices if scores[idx] > 0}

    def _faiss_search_fast(self, query: str, k: int) -> Dict[int, float]:
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
            device='cuda' if self.use_gpu else 'cpu'
        )
        distances, indices = self.faiss_index.search(query_embedding, k)
        return {idx: float(dist) for idx, dist in zip(indices[0], distances[0]) if idx != -1 and dist > 0.1}

    def _fast_fusion(self, bm25_results: Dict[int, float], faiss_results: Dict[int, float], query: str) -> List[str]:
        if bm25_results:
            max_bm25 = max(bm25_results.values())
            if max_bm25 > 0: bm25_results = {k: v/max_bm25 for k, v in bm25_results.items()}
        if faiss_results:
            max_faiss = max(faiss_results.values())
            if max_faiss > 0: faiss_results = {k: v/max_faiss for k, v in faiss_results.items()}

        w1, w2 = (0.7, 0.3) if any(kw in query.lower() for kw in ['rate', 'fee', 'amount', 'percent']) else (0.4, 0.6)

        all_indices = set(bm25_results.keys()) | set(faiss_results.keys())
        scored_results = []
        for idx in all_indices:
            score = (w1 * bm25_results.get(idx, 0) + w2 * faiss_results.get(idx, 0))
            if score > 0.1: scored_results.append((idx, score))
        
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [self.chunks[idx] for idx, _ in scored_results]

    def __del__(self):
        if self.use_gpu:
            torch.cuda.empty_cache()