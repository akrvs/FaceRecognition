from __future__ import annotations

from visage.config import Settings
from visage.service.recognizer import FaceRecognizer


class RecognizerProvider:
    def __init__(self) -> None:
        self._recognizer: FaceRecognizer | None = None

    def set(self, recognizer: FaceRecognizer) -> None:
        self._recognizer = recognizer

    @property
    def is_set(self) -> bool:
        return self._recognizer is not None

    def get(self) -> FaceRecognizer:
        if self._recognizer is None:
            raise RuntimeError("Recognizer has not been initialized")
        return self._recognizer


def build_default_recognizer(settings: Settings) -> FaceRecognizer:
    from visage.embeddings.insightface_embedder import InsightFaceEmbedder
    from visage.index import build_index

    embedder = InsightFaceEmbedder(
        model_name=settings.model_name,
        detection_size=settings.detection_size,
        detection_confidence=settings.detection_confidence,
    )
    index = build_index(settings.index_backend, embedder.embedding_dim)
    return FaceRecognizer(embedder, index, match_threshold=settings.match_threshold)
