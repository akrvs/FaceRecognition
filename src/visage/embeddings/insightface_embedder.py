from __future__ import annotations

import numpy as np

from visage.models import DetectedFace


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


class InsightFaceEmbedder:
    def __init__(
        self,
        model_name: str = "buffalo_l",
        detection_size: int = 640,
        detection_confidence: float = 0.5,
    ) -> None:
        from insightface.app import FaceAnalysis

        self._detection_confidence = detection_confidence
        self._app = FaceAnalysis(name=model_name)
        self._app.prepare(ctx_id=0, det_size=(detection_size, detection_size))
        self._embedding_dim = 512

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, image: np.ndarray) -> list[DetectedFace]:
        faces = self._app.get(image)
        results: list[DetectedFace] = []
        for face in faces:
            if float(face.det_score) < self._detection_confidence:
                continue
            box = face.bbox.astype(int)
            results.append(
                DetectedFace(
                    bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                    embedding=_normalize(np.asarray(face.embedding, dtype=np.float32)),
                    detection_score=float(face.det_score),
                )
            )
        return results
