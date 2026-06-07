from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from visage.models import DetectedFace


@runtime_checkable
class Embedder(Protocol):
    @property
    def embedding_dim(self) -> int: ...

    def embed(self, image: np.ndarray) -> list[DetectedFace]: ...
