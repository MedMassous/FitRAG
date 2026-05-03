"""
retriever.py — Hybrid Retrieval Engine
Combines FAISS semantic search with BM25 keyword search.
Includes a lightweight re-ranker based on query-chunk overlap scoring.
"""

import re
import math
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np

from ingestion import ChromaIndex, EmbeddingModel, load_index_and_chunks

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

TOP_K_SEMANTIC = 10
TOP_K_BM25     = 10
TOP_K_FINAL    = 6
ALPHA          = 0.6 


# ─────────────────────────────────────────────
# BM25 (Okapi) Implementation
# ─────────────────────────────────────────────

class BM25:
    """Lightweight BM25 built over the stored chunk corpus."""

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.corpus = corpus
        self.N = len(corpus)

        self.tokenised = [self._tok(doc) for doc in corpus]
        self.avgdl = sum(len(d) for d in self.tokenised) / max(self.N, 1)

        self.tf:  List[Counter] = [Counter(d) for d in self.tokenised]
        self.df:  Counter = Counter()
        for doc_tokens in self.tokenised:
            for term in set(doc_tokens):
                self.df[term] += 1

    @staticmethod
    def _tok(text: str) -> List[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def score(self, query: str) -> np.ndarray:
        q_terms = self._tok(query)
        scores  = np.zeros(self.N, dtype=np.float32)

        for term in q_terms:
            if term not in self.df:
                continue
            idf = math.log((self.N - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1)
            for i, tf_doc in enumerate(self.tf):
                freq = tf_doc[term]
                dl   = len(self.tokenised[i])
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (freq * (self.k1 + 1)) / denom

        return scores

    def top_k(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        scores  = self.score(query)
        indices = np.argsort(scores)[::-1][:k]
        return scores[indices], indices


# ─────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    semantic_ids: List[int],
    bm25_ids:     List[int],
    alpha: float = ALPHA,
    k_rrf: int   = 60,
) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = defaultdict(float)

    for rank, idx in enumerate(semantic_ids):
        scores[idx] += alpha * (1.0 / (k_rrf + rank + 1))

    for rank, idx in enumerate(bm25_ids):
        scores[idx] += (1 - alpha) * (1.0 / (k_rrf + rank + 1))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────
# Re-ranker
# ─────────────────────────────────────────────

def rerank(
    query:      str,
    candidates: List[Dict],
    top_k:      int = TOP_K_FINAL,
) -> List[Dict]:
    fitness_keywords = {
        "workout", "exercise", "training", "sets", "reps", "rest",
        "cardio", "strength", "muscle", "program", "split", "overload",
        "squat", "deadlift", "bench", "pull", "push", "leg", "upper",
        "lower", "core", "recovery", "nutrition", "protein", "calories",
    }
    q_tokens = set(re.findall(r"\b[a-z]{2,}\b", query.lower()))

    scored = []
    for chunk in candidates:
        text     = chunk["text"].lower()
        c_tokens = set(re.findall(r"\b[a-z]{2,}\b", text))

        overlap     = len(q_tokens & c_tokens) / (len(q_tokens) + 1)
        fit_boost   = len(c_tokens & fitness_keywords) * 0.02
        length_norm = min(len(text) / 400, 1.0)

        chunk["rerank_score"] = overlap + fit_boost + 0.1 * length_norm
        scored.append(chunk)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]


# ─────────────────────────────────────────────
# Main Retriever
# ─────────────────────────────────────────────

class FitnessRetriever:
    """Hybrid retriever: ChromaDB semantic + BM25 keyword + RRF fusion + re-rank."""

    def __init__(self):
        self.chroma_idx, self.chunks = load_index_and_chunks()
        self.embedder = EmbeddingModel.get()
        self._bm25: Optional[BM25] = None

        if not self.chunks:
            log.warning("No chunks loaded — ingest PDFs first.")

    @property
    def bm25(self) -> BM25:
        """Lazy-init BM25 (built once, invalidated on refresh)."""
        if self._bm25 is None:
            self._bm25 = BM25([c["text"] for c in self.chunks])
        return self._bm25

    def retrieve(
        self,
        query:         str,
        top_k:         int          = TOP_K_FINAL,
        source_filter: Optional[str] = None,
    ) -> List[Dict]:
        if not self.chunks:
            return []

        # 1. Semantic search via ChromaDB collection.query()
        q_vec = self.embedder.embed([query])
        sem_scores, sem_ids = self.chroma_idx.search(q_vec, k=TOP_K_SEMANTIC)

        sem_ids_valid = [
            (int(i), float(s))
            for i, s in zip(sem_ids, sem_scores)
            if i >= 0 and i < len(self.chunks)
        ]

        bm25_scores, bm25_ids = self.bm25.top_k(query, k=TOP_K_BM25)
        bm25_ids_list = [int(i) for i in bm25_ids]

        sem_id_list = [i for i, _ in sem_ids_valid]
        fused       = reciprocal_rank_fusion(sem_id_list, bm25_ids_list, alpha=ALPHA)

        sem_score_map  = {i: s for i, s in sem_ids_valid}
        bm25_score_map = {int(idx): float(bm25_scores[pos]) for pos, idx in enumerate(bm25_ids_list)}

        seen     = set()
        hydrated = []
        for idx, fused_score in fused:
            if idx in seen:
                continue
            seen.add(idx)
            chunk = dict(self.chunks[idx])   

            chunk["semantic_score"] = sem_score_map.get(idx, 0.0)
            chunk["bm25_score"]     = bm25_score_map.get(idx, 0.0)
            chunk["fused_score"]    = fused_score
            hydrated.append(chunk)

        if source_filter:
            hydrated = [c for c in hydrated if c["source"] == source_filter]

        results = rerank(query, hydrated, top_k=top_k)

        log.info(
            f"Retrieved {len(results)} chunks "
            f"(semantic pool: {len(sem_ids_valid)}, bm25 pool: {len(bm25_ids_list)})"
        )
        return results

    def retrieve_by_file(self, filename: str, top_k: int = 20) -> List[Dict]:
        return [c for c in self.chunks if c["source"] == filename][:top_k]

    def available_sources(self) -> List[str]:
        return sorted({c["source"] for c in self.chunks})

    def refresh(self):
        self.chroma_idx, self.chunks = load_index_and_chunks()
        self._bm25 = None
        log.info(f"Retriever refreshed — {len(self.chunks)} chunks loaded")
