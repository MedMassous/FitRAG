"""
ingestion.py — PDF Ingestion Pipeline
Handles PDF parsing, chunking, embedding, and FAISS vector DB storage.
Supports incremental indexing (new PDFs added without full reindex).
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import fitz 
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────
# Configuration — read from env vars with sensible defaults
# ─────────────────────────────────────────────
VECTOR_DB_PATH = Path("data/faiss_index")
METADATA_PATH  = Path("data/metadata.json")
CHUNKS_PATH    = Path("data/chunks.json")
HASH_PATH      = Path("data/processed_hashes.json")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "64"))
EMBED_DIM        = 384

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _ensure_dirs():
    for p in [VECTOR_DB_PATH, METADATA_PATH.parent]:
        p.mkdir(parents=True, exist_ok=True)


def _file_hash(path: str) -> str:
    """SHA256 of file bytes — used to skip already-indexed PDFs."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path, default):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default


def _save_json(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
# PDF Extraction
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    pages = []
    filename = Path(pdf_path).name

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if len(text) > 30:
            pages.append({
                "page": page_num,
                "text": text,
                "source": filename,
                "total_pages": len(doc),
            })

    doc.close()
    log.info(f"Extracted {len(pages)} pages from '{filename}'")
    return pages


# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────

def chunk_pages(pages: List[Dict]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "text":        split,
                "source":      page["source"],
                "page":        page["page"],
                "chunk_index": i,
                "chunk_id":    f"{page['source']}::p{page['page']}::c{i}",
            })
    log.info(f"Generated {len(chunks)} chunks from {len(pages)} pages")
    return chunks


# ─────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────

class EmbeddingModel:
    """Singleton wrapper around SentenceTransformer."""
    _instance: Optional["EmbeddingModel"] = None

    def __init__(self):
        log.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        self.model = SentenceTransformer(EMBED_MODEL_NAME)

    @classmethod
    def get(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
        )
        return np.array(vecs, dtype=np.float32)


# ─────────────────────────────────────────────
# FAISS Index Manager
# ─────────────────────────────────────────────

class FAISSIndex:
    def __init__(self):
        self.index_file = VECTOR_DB_PATH / "index.faiss"
        self.index: Optional[faiss.Index] = None
        self._load_or_create()

    def _load_or_create(self):
        if self.index_file.exists():
            log.info("Loading existing FAISS index…")
            self.index = faiss.read_index(str(self.index_file))
        else:
            log.info("Creating new FAISS index…")
            self.index = faiss.IndexFlatIP(EMBED_DIM)

    def add(self, vectors: np.ndarray):
        self.index.add(vectors)
        self._save()

    def search(self, query_vec: np.ndarray, k: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        scores, indices = self.index.search(query_vec, k)
        return scores[0], indices[0]

    def _save(self):
        faiss.write_index(self.index, str(self.index_file))

    @property
    def total(self) -> int:
        return self.index.ntotal


# ─────────────────────────────────────────────
# Ingestion Orchestrator
# ─────────────────────────────────────────────

class IngestionPipeline:
    def __init__(self):
        _ensure_dirs()
        self.faiss_idx  = FAISSIndex()
        self.embedder   = EmbeddingModel.get()
        self.chunks     = _load_json(CHUNKS_PATH, [])
        self.metadata   = _load_json(METADATA_PATH, {})
        self.hashes     = _load_json(HASH_PATH, {})

    def ingest(self, pdf_paths: List[str]) -> Dict:
        new_files = []
        skipped   = []

        for path in pdf_paths:
            fname = Path(path).name
            fhash = _file_hash(path)

            if self.hashes.get(fname) == fhash:
                log.info(f"Skipping '{fname}' (already indexed)")
                skipped.append(fname)
                continue

            pages  = extract_text_from_pdf(path)
            chunks = chunk_pages(pages)

            if not chunks:
                log.warning(f"No usable text in '{fname}', skipping.")
                continue

            texts   = [c["text"] for c in chunks]
            vectors = self.embedder.embed(texts)

            start_idx = len(self.chunks)
            self.chunks.extend(chunks)
            self.faiss_idx.add(vectors)

            self.metadata[fname] = {
                "path":        path,
                "pages":       len(pages),
                "chunks":      len(chunks),
                "ingested_at": datetime.utcnow().isoformat(),
                "start_idx":   start_idx,
            }
            self.hashes[fname] = fhash
            new_files.append(fname)
            log.info(f"✅ Indexed '{fname}' — {len(chunks)} chunks")

        self._save_all()

        return {
            "indexed":       new_files,
            "skipped":       skipped,
            "total_chunks":  len(self.chunks),
            "total_vectors": self.faiss_idx.total,
        }

    def reindex_all(self, pdf_paths: List[str]) -> Dict:
        """Force full reindex (clears existing DB)."""
        log.info("Full reindex requested — clearing existing data…")
        self.chunks   = []
        self.metadata = {}
        self.hashes   = {}
        for p in [CHUNKS_PATH, METADATA_PATH, HASH_PATH]:
            if p.exists():
                p.unlink()
        for f in VECTOR_DB_PATH.glob("*"):
            f.unlink()
        self.faiss_idx = FAISSIndex()   # fresh in-memory + on-disk index
        return self.ingest(pdf_paths)

    def get_indexed_files(self) -> List[Dict]:
        return [{"filename": k, **v} for k, v in self.metadata.items()]

    def _save_all(self):
        _save_json(CHUNKS_PATH, self.chunks)
        _save_json(METADATA_PATH, self.metadata)
        _save_json(HASH_PATH, self.hashes)


# ─────────────────────────────────────────────
# Convenience loader (used by retriever)
# ─────────────────────────────────────────────

def load_index_and_chunks() -> Tuple[FAISSIndex, List[Dict]]:
    """Load persisted FAISS index + chunk metadata."""
    _ensure_dirs()
    idx    = FAISSIndex()
    chunks = _load_json(CHUNKS_PATH, [])
    return idx, chunks
