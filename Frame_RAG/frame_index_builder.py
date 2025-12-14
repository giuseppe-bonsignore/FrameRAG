#!/usr/bin/env python3
# frame_index_builder_min.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import List, Iterable, Tuple, Optional

# Optional .env
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(override=True)
except Exception:
    pass

import numpy as np
from openai import OpenAI

# Optional tqdm (graceful fallback)
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    def tqdm(iterable=None, *args, total=None, disable=False, **kwargs):
        class _NoTqdm:
            def __init__(self, total=None, disable=False): pass
            def update(self, n=1): pass
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, exc_type, exc, tb): pass
        if iterable is None:
            return _NoTqdm(total=total, disable=disable)
        return iterable

# ---- Config from env (with sane defaults) ----
ROOT_DIR = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent))
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data")); DATA_DIR.mkdir(parents=True, exist_ok=True)

FRAME_CORPUS   = Path(os.getenv("FRAME_CORPUS", DATA_DIR / "frame_corpus.txt"))
INDEX_DIR      = Path(os.getenv("INDEX_DIR", DATA_DIR / "frame_index")); INDEX_DIR.mkdir(parents=True, exist_ok=True)
INDEX_VECTORS  = Path(os.getenv("INDEX_VECTORS", INDEX_DIR / "vectors.npy"))
INDEX_IDS_JSON = Path(os.getenv("INDEX_IDS_JSON", INDEX_DIR / "frame_ids.json"))
OPENAI_MODEL   = os.getenv("EMBEDDER_MODEL") or os.getenv("EMBED_MODEL") or "text-embedding-3-small"
BATCH_SIZE     = int(os.getenv("BATCH_SIZE", "128"))

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- Tiny helpers ----
def _batched(seq: List[str], size: int) -> Iterable[List[str]]: # batching generator. It walks through a list in steps of size and yields slices.
    for i in range(0, len(seq), size): 
        yield seq[i:i+size] # makes this a generator: it produces one batch at a time instead of building all batches in memory.

def _normalize_rows(X: np.ndarray) -> np.ndarray: # L2-normalizes each row (for cosine similarity).
    X = X.astype(np.float32, copy=False) # Ensures float32 (saves memory/compute). Avoid copying if dtype already matches.
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12 # Computes each row’s L2 norm (Euclidean distance) and devides by it. +1e... prevents dividing by 0 
    return X / n # each vector is normalized so it becomes of length 1. Dot product becomes = to the cosine of the angle between a query q and the a vector x becomes = to the cosine between the two angles. q.v = |q||v|cos(x)=cos(x)

class OpenAIEmbedder:
    """Exact embedder for OpenAI text-embedding models with .encode(list[str])."""
    def __init__(self, model: str = OPENAI_MODEL):
        self.model = model
    def encode(self, texts: List[str]) -> List[List[float]]: # ["doc1", "doc2"] -> [[embedding1][embedding2]]
        resp = _client.embeddings.create(model=self.model, input=texts) # Creates an embedding vector representing the input text. Pretraining and contrastive tuning allowed for embedding short queries. Even one word now has its "pin on the map", as the models already learned its meaning from the whole web. resp.data is a list of per-input results from the embeddings API
        return [d.embedding for d in resp.data] # Each item d has an .embedding field (the vector for that input). The list comprehension [d.embedding for d in resp.data] builds [[float, float, ...], ...], one vector per input string, in the same order as the inputs.

class CosineANN: # cosine-similarity search index. Given a matrix of vectors find the top-k most similar items to a query vector using cosine similarity.
    """Tiny cosine index with .search(vec, k) -> (D, I). Expects row-normalized matrix."""
    def __init__(self, matrix: np.ndarray):
        if matrix.ndim != 2:
            raise ValueError("Index matrix must be 2D")
        self.mat = matrix.astype(np.float32, copy=False) # forces the index matrix to be float32 and avoids an extra copy if it’s already that dtype.
    def search(self, queries: np.ndarray, k: int):
        if queries.ndim == 1:
            queries = queries[None, :]
        q = queries.astype(np.float32, copy=False) # casts the query vector as float32 and avoids recasting if it's already float32 
        q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12) # taking the norm 
        sims = q @ self.mat.T # math trick that takes the dot product of q by each row of the matrix. sims stands for similarities. It's the list of all the scores. 
        I = np.argsort(-sims, axis=1)[:, :k] # sims has shape (B,N)-> similarity for B queries vs N items; argsort returns the column indices that sorts that row ascending order (sorting ascending on -sims means sorting descending on sims. sims ascending is negated). k -> takes the k most similar indices in the row. This means the top-k most similar items per query. 
        D = sims[np.arange(sims.shape[0])[:, None], I]
        return D, I 
    
    """
    Examples: 

    sims = np.array([[0.6, 0.8, 0.99]])        # B=1, N=3
    I = np.argsort(-sims, axis=1)[:, :2]       # -> [[2, 1]]  (row3 best, then row2)


    sims = np.array([[0.1, 0.9, 0.7],
                 [0.3, 0.2, 0.8]])             # (B=2, N=3)

    I = np.array([[1, 2],                      # top-2 cols per row
              [2, 0]])                         # (B=2, k=2)

    rows = np.arange(2)[:, None]               # [[0],[1]]
    D    = sims[rows, I]                       # -> [[0.9, 0.7],
                                               #    [0.8, 0.3]]
    
    I tells you where is the k most similar in ascending order. 

    """

def read_frames_corpus(path: Path = FRAME_CORPUS) -> Tuple[List[str], List[str]]: # parses frame corpus file and returns two aligned lists: one of frame IDs and one of their text blocks.
    if not path.exists():
        raise FileNotFoundError(f"Frame corpus not found at {path}")
    ids, texts, buf = [], [], []
    current_id: Optional[str] = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("### "):
                if current_id is not None:
                    ids.append(current_id)
                    texts.append("".join(buf).strip())
                    buf = []
                current_id = line.strip()[4:]
            else:
                buf.append(line)
    if current_id is not None:
        ids.append(current_id)
        texts.append("".join(buf).strip())
    return ids, texts


def build_frame_index() -> None:
    """
    Reads FRAME_CORPUS, embeds with OpenAI, and writes:
      - INDEX_VECTORS (row-normalized float32 matrix)
      - INDEX_IDS_JSON (list[str])
    Uses a tqdm progress bar. All paths/model come from env/.env.
    """
    frame_ids, texts = _read_frames_corpus(FRAME_CORPUS)
    print(f"[index] Loaded corpus: {len(texts)} docs")
    print(f"[index] Model: {OPENAI_MODEL}  |  Batch size: {BATCH_SIZE}")

    vectors: List[List[float]] = []
    show_bar = not os.getenv("CI")

    with tqdm(total=len(texts), disable=not show_bar, desc="Embedding") as pbar:
        for batch in _batched(texts, BATCH_SIZE):
            resp = _client.embeddings.create(model=OPENAI_MODEL, input=batch)
            vectors.extend([d.embedding for d in resp.data])
            pbar.update(len(batch))

    M = _normalize_rows(np.asarray(vectors, dtype=np.float32))
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_VECTORS, M)
    with open(INDEX_IDS_JSON, "w", encoding="utf-8") as f:
        json.dump(frame_ids, f, ensure_ascii=False, indent=2)

    print(f"[index] ✅ Saved vectors to {INDEX_VECTORS}")
    print(f"[index] ✅ Saved ids to {INDEX_IDS_JSON}")

def load_frame_index() -> Tuple[CosineANN, OpenAIEmbedder, List[str]]:
    """
    Load vectors.npy and frame_ids.json and return (index, embedder, frame_ids).
    Matches the old retriever.load_frame_index() shape.
    """
    missing = []
    if not INDEX_VECTORS.exists():
        missing.append(str(INDEX_VECTORS))
    if not INDEX_IDS_JSON.exists():
        missing.append(str(INDEX_IDS_JSON))
    if missing:
        raise FileNotFoundError(
            "Index artifacts not found. Build with build_frame_index() first. "
            "Missing: " + ", ".join(missing)
        )

    M = np.load(INDEX_VECTORS)
    if M.ndim != 2:
        raise ValueError(f"Index matrix has wrong shape {M.shape}; expected (N, D).")

    with INDEX_IDS_JSON.open("r", encoding="utf-8") as f:
        ids = json.load(f)

    if len(ids) != M.shape[0]:
        raise ValueError(
            f"Row count mismatch: vectors={M.shape[0]} vs ids={len(ids)}. "
            "Delete both files and rebuild the index."
        )

    return CosineANN(M), OpenAIEmbedder(OPENAI_MODEL), ids
    
# ---- Minimal CLI entry ----
if __name__ == "__main__":
    build_frame_index()
