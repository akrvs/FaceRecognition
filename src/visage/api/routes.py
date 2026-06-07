from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from visage.api.dependencies import RecognizerProvider
from visage.config import Settings, get_settings
from visage.imaging import decode_image
from visage.models import (
    EnrollResponse,
    HealthResponse,
    IdentitiesResponse,
    RecognizeResponse,
    VerifyResponse,
)
from visage.service.recognizer import FaceRecognizer, NoFaceDetectedError

router = APIRouter()
recognizer_provider = RecognizerProvider()


def get_recognizer() -> FaceRecognizer:
    return recognizer_provider.get()


@router.get("/health", response_model=HealthResponse)
def health(
    recognizer: FaceRecognizer = Depends(get_recognizer),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_name=settings.model_name,
        index_backend=settings.index_backend,
        indexed_vectors=recognizer.indexed_vectors,
    )


@router.post("/enroll", response_model=EnrollResponse)
async def enroll(
    name: str = Form(...),
    file: UploadFile = File(...),
    recognizer: FaceRecognizer = Depends(get_recognizer),
) -> EnrollResponse:
    image = decode_image(await file.read())
    try:
        added = recognizer.enroll(name, image)
    except NoFaceDetectedError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return EnrollResponse(
        name=name,
        embeddings_added=added,
        total_identities=recognizer.identities().total,
    )


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    file: UploadFile = File(...),
    threshold: float | None = Form(default=None),
    recognizer: FaceRecognizer = Depends(get_recognizer),
) -> RecognizeResponse:
    image = decode_image(await file.read())
    faces = recognizer.recognize(image, threshold=threshold)
    return RecognizeResponse(faces=faces)


@router.post("/verify", response_model=VerifyResponse)
async def verify(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    threshold: float | None = Form(default=None),
    recognizer: FaceRecognizer = Depends(get_recognizer),
    settings: Settings = Depends(get_settings),
) -> VerifyResponse:
    image_a = decode_image(await file_a.read())
    image_b = decode_image(await file_b.read())
    effective_threshold = settings.match_threshold if threshold is None else threshold
    try:
        similarity = recognizer.verify(image_a, image_b)
    except NoFaceDetectedError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return VerifyResponse(
        similarity=similarity,
        is_match=similarity >= effective_threshold,
        threshold=effective_threshold,
    )


@router.get("/identities", response_model=IdentitiesResponse)
def identities(
    recognizer: FaceRecognizer = Depends(get_recognizer),
) -> IdentitiesResponse:
    return recognizer.identities()
