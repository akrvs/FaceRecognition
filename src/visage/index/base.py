from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class SearchResult:
    label: int
    score: float


class VectorIndex(Protocol):
    @property
    def dim(self) -> int: ...

    def __len__(self) -> int: ...

    def add(self, vectors: np.ndarray, labels: list[int]) -> None: ...

    def search(self, vectors: np.ndarray, k: int) -> list[list[SearchResult]]: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...
