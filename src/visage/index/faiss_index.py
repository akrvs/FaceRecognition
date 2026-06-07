from __future__ import annotations

from pathlib import Path

import numpy as np

from visage.index.base import SearchResult


class FaissIndex:
    def __init__(self, dim: int) -> None:
        import faiss

        self._faiss = faiss
        self._dim = dim
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return int(self._index.ntotal)

    def add(self, vectors: np.ndarray, labels: list[int]) -> None:
        matrix = self._as_matrix(vectors)
        if matrix.shape[0] != len(labels):
            raise ValueError("Number of vectors and labels must match")
        self._index.add_with_ids(matrix, np.asarray(labels, dtype=np.int64))

    def search(self, vectors: np.ndarray, k: int) -> list[list[SearchResult]]:
        queries = self._as_matrix(vectors)
        if len(self) == 0:
            return [[] for _ in range(queries.shape[0])]
        top = min(k, len(self))
        scores, ids = self._index.search(queries, top)
        results: list[list[SearchResult]] = []
        for row_scores, row_ids in zip(scores, ids, strict=True):
            hits = [
                SearchResult(int(label), float(score))
                for score, label in zip(row_scores, row_ids, strict=True)
                if label != -1
            ]
            results.append(hits)
        return results

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path))

    def load(self, path: Path) -> None:
        self._index = self._faiss.read_index(str(Path(path)))

    def _as_matrix(self, vectors: np.ndarray) -> np.ndarray:
        matrix = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        if matrix.shape[1] != self._dim:
            raise ValueError(f"Expected vectors of dim {self._dim}, got {matrix.shape[1]}")
        return np.ascontiguousarray(matrix)
