from visage.index.base import SearchResult, VectorIndex
from visage.index.numpy_index import NumpyIndex

__all__ = ["SearchResult", "VectorIndex", "NumpyIndex", "build_index"]


def build_index(backend: str, dim: int) -> VectorIndex:
    if backend == "numpy":
        return NumpyIndex(dim)
    if backend == "faiss":
        from visage.index.faiss_index import FaissIndex

        return FaissIndex(dim)
    raise ValueError(f"Unknown index backend: {backend}")
