from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from visage.models import DetectedFace

EMBEDDING_DIM = 256
NO_FACE = 255


def _identity_embedding(identity: int) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[identity % EMBEDDING_DIM] = 1.0
    return vector


def make_face_array(identity: int, detection_score: float = 0.9) -> np.ndarray:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[0, 0, :] = identity
    image[0, 1, :] = int(detection_score * 100)
    return image


def make_face_png(identity: int, detection_score: float = 0.9) -> bytes:
    array = make_face_array(identity, detection_score)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeEmbedder:
    def __init__(self, faces_per_image: int = 1) -> None:
        self._faces_per_image = faces_per_image

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    def embed(self, image: np.ndarray) -> list[DetectedFace]:
        identity = int(image[0, 0, 0])
        if identity == NO_FACE:
            return []
        detection_score = float(image[0, 1, 0]) / 100.0
        face = DetectedFace(
            bbox=(0, 0, 8, 8),
            embedding=_identity_embedding(identity),
            detection_score=detection_score,
        )
        return [face] * self._faces_per_image


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def recognizer(fake_embedder: FakeEmbedder):
    from visage.index import NumpyIndex
    from visage.service.recognizer import FaceRecognizer

    return FaceRecognizer(fake_embedder, NumpyIndex(EMBEDDING_DIM), match_threshold=0.5)


@pytest.fixture
def client(recognizer):
    from fastapi.testclient import TestClient

    from visage.api.app import create_app
    from visage.api.routes import recognizer_provider

    recognizer_provider.set(recognizer)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    recognizer_provider._recognizer = None
