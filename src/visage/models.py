from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field


@dataclass(slots=True)
class DetectedFace:
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray
    detection_score: float


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class RecognizedFace(BaseModel):
    name: str | None
    score: float
    bbox: BoundingBox
    detection_score: float


class EnrollResponse(BaseModel):
    name: str
    embeddings_added: int
    total_identities: int


class RecognizeResponse(BaseModel):
    faces: list[RecognizedFace]


class VerifyResponse(BaseModel):
    similarity: float
    is_match: bool
    threshold: float


class IdentitySummary(BaseModel):
    name: str
    embeddings: int


class IdentitiesResponse(BaseModel):
    identities: list[IdentitySummary]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_name: str
    index_backend: str
    indexed_vectors: int


class ErrorResponse(BaseModel):
    detail: str = Field(examples=["No face detected in the provided image"])
