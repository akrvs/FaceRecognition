from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def decode_image(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return _to_bgr(np.asarray(image))


def load_image(path: str | Path) -> np.ndarray:
    image = Image.open(Path(path)).convert("RGB")
    return _to_bgr(np.asarray(image))


def _to_bgr(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb[:, :, ::-1])
