from __future__ import annotations

from pathlib import Path

import numpy as np

from visage.index.base import SearchResult


class NumpyIndex:
    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._vectors = np.empty((0, dim), dtype=np.float32)
        self._labels: list[int] = []

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return len(self._labels)

    def add(self, vectors: np.ndarray, labels: list[int]) -> None:
        matrix = self._as_matrix(vectors)
        if matrix.shape[0] != len(labels):
            raise ValueError("Number of vectors and labels must match")
        self._vectors = np.vstack([self._vectors, matrix])
        self._labels.extend(labels)

    def search(self, vectors: np.ndarray, k: int) -> list[list[SearchResult]]:
        queries = self._as_matrix(vectors)
        if len(self._labels) == 0:
            return [[] for _ in range(queries.shape[0])]

        similarities = queries @ self._vectors.T
        top = min(k, similarities.shape[1])
        results: list[list[SearchResult]] = []
        for row in similarities:
            order = np.argpartition(-row, top - 1)[:top]
            order = order[np.argsort(-row[order])]
            results.append([SearchResult(self._labels[i], float(row[i])) for i in order])
        return results

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, vectors=self._vectors, labels=np.asarray(self._labels, dtype=np.int64))

    def load(self, path: Path) -> None:
        data = np.load(Path(path))
        self._vectors = data["vectors"].astype(np.float32)
        self._labels = data["labels"].astype(np.int64).tolist()

    def _as_matrix(self, vectors: np.ndarray) -> np.ndarray:
        matrix = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        if matrix.shape[1] != self._dim:
            raise ValueError(f"Expected vectors of dim {self._dim}, got {matrix.shape[1]}")
        return matrix
