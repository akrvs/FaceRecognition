from __future__ import annotations

from pathlib import Path

import numpy as np

from visage.embeddings.base import Embedder
from visage.index.base import VectorIndex
from visage.models import (
    BoundingBox,
    IdentitiesResponse,
    IdentitySummary,
    RecognizedFace,
)
from visage.service.gallery import Gallery


class NoFaceDetectedError(ValueError):
    pass


class FaceRecognizer:
    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        gallery: Gallery | None = None,
        match_threshold: float = 0.35,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._gallery = gallery or Gallery()
        self._match_threshold = match_threshold

    @property
    def indexed_vectors(self) -> int:
        return len(self._index)

    def enroll(self, name: str, image: np.ndarray) -> int:
        faces = self._embedder.embed(image)
        if not faces:
            raise NoFaceDetectedError("No face detected in the provided image")

        embeddings = np.vstack([face.embedding for face in faces])
        labels = [self._gallery.add(name) for _ in faces]
        self._index.add(embeddings, labels)
        return len(faces)

    def recognize(
        self, image: np.ndarray, threshold: float | None = None
    ) -> list[RecognizedFace]:
        threshold = self._match_threshold if threshold is None else threshold
        faces = self._embedder.embed(image)
        if not faces:
            return []

        query = np.vstack([face.embedding for face in faces])
        hits = self._index.search(query, k=1)

        recognized: list[RecognizedFace] = []
        for face, face_hits in zip(faces, hits, strict=True):
            name: str | None = None
            score = 0.0
            if face_hits:
                best = face_hits[0]
                score = best.score
                if score >= threshold:
                    name = self._gallery.name_for(best.label)
            x1, y1, x2, y2 = face.bbox
            recognized.append(
                RecognizedFace(
                    name=name,
                    score=score,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    detection_score=face.detection_score,
                )
            )
        return recognized

    def verify(self, image_a: np.ndarray, image_b: np.ndarray) -> float:
        embedding_a = self._single_embedding(image_a)
        embedding_b = self._single_embedding(image_b)
        return float(np.dot(embedding_a, embedding_b))

    def identities(self) -> IdentitiesResponse:
        counts = self._gallery.counts()
        summaries = [
            IdentitySummary(name=name, embeddings=count) for name, count in sorted(counts.items())
        ]
        return IdentitiesResponse(identities=summaries, total=len(summaries))

    def save(self, index_path: Path, gallery_path: Path) -> None:
        self._index.save(index_path)
        self._gallery.save(gallery_path)

    def load(self, index_path: Path, gallery_path: Path) -> None:
        self._index.load(index_path)
        self._gallery = Gallery.load(gallery_path)

    def _single_embedding(self, image: np.ndarray) -> np.ndarray:
        faces = self._embedder.embed(image)
        if not faces:
            raise NoFaceDetectedError("No face detected in the provided image")
        return max(faces, key=lambda face: face.detection_score).embedding
