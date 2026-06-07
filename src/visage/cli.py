from __future__ import annotations

import json
from pathlib import Path

import typer

from visage.config import get_settings

app = typer.Typer(help="Visage face recognition toolkit", no_args_is_help=True)


@app.command()
def serve(
    host: str | None = typer.Option(None),
    port: int | None = typer.Option(None),
) -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "visage.api.app:create_app",
        factory=True,
        host=host or settings.api_host,
        port=port or settings.api_port,
    )


@app.command()
def enroll(name: str, image: Path) -> None:
    recognizer = _load_recognizer()
    from visage.imaging import load_image

    added = recognizer.enroll(name, load_image(image))
    settings = get_settings()
    recognizer.save(settings.index_path, settings.gallery_path)
    typer.echo(f"Enrolled {name} with {added} embedding(s)")


@app.command()
def recognize(image: Path, threshold: float | None = typer.Option(None)) -> None:
    recognizer = _load_recognizer()
    from visage.imaging import load_image

    faces = recognizer.recognize(load_image(image), threshold=threshold)
    typer.echo(json.dumps([face.model_dump() for face in faces], indent=2))


@app.command()
def evaluate(
    lfw_dir: Path,
    pairs: int = typer.Option(2000),
    seed: int = typer.Option(42),
) -> None:
    from visage.embeddings.insightface_embedder import InsightFaceEmbedder
    from visage.evaluation import build_pairs, evaluate_pairs

    settings = get_settings()
    embedder = InsightFaceEmbedder(
        model_name=settings.model_name,
        detection_size=settings.detection_size,
        detection_confidence=settings.detection_confidence,
    )
    result = evaluate_pairs(embedder, build_pairs(lfw_dir, pairs, seed))
    typer.echo(json.dumps(result.__dict__, indent=2))


def _load_recognizer():
    from visage.api.dependencies import build_default_recognizer

    settings = get_settings()
    recognizer = build_default_recognizer(settings)
    if settings.index_path.exists() and settings.gallery_path.exists():
        recognizer.load(settings.index_path, settings.gallery_path)
    return recognizer


if __name__ == "__main__":
    app()
