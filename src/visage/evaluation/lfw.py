from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from visage.embeddings.base import Embedder
from visage.evaluation.metrics import best_threshold_accuracy, roc_auc
from visage.imaging import load_image


@dataclass(slots=True)
class Pair:
    path_a: Path
    path_b: Path
    same: bool


@dataclass(slots=True)
class EvaluationResult:
    pairs: int
    evaluated: int
    accuracy: float
    threshold: float
    auc: float


def _people_with_multiple_images(lfw_dir: Path) -> dict[str, list[Path]]:
    people: dict[str, list[Path]] = {}
    for person_dir in sorted(p for p in lfw_dir.iterdir() if p.is_dir()):
        images = sorted(person_dir.glob("*.jpg"))
        if images:
            people[person_dir.name] = images
    return people


def build_pairs(lfw_dir: str | Path, num_pairs: int, seed: int = 42) -> list[Pair]:
    rng = random.Random(seed)
    people = _people_with_multiple_images(Path(lfw_dir))
    multi = {name: imgs for name, imgs in people.items() if len(imgs) >= 2}
    names = list(people)

    positive_target = num_pairs // 2
    negative_target = num_pairs - positive_target
    multi_names = list(multi)

    pairs: list[Pair] = []
    for _ in range(positive_target):
        if not multi_names:
            break
        name = rng.choice(multi_names)
        a, b = rng.sample(multi[name], 2)
        pairs.append(Pair(a, b, True))

    for _ in range(negative_target):
        name_a, name_b = rng.sample(names, 2)
        pairs.append(Pair(rng.choice(people[name_a]), rng.choice(people[name_b]), False))

    rng.shuffle(pairs)
    return pairs


def evaluate_pairs(embedder: Embedder, pairs: list[Pair]) -> EvaluationResult:
    scores: list[float] = []
    labels: list[int] = []
    for pair in pairs:
        embedding_a = _embedding(embedder, pair.path_a)
        embedding_b = _embedding(embedder, pair.path_b)
        if embedding_a is None or embedding_b is None:
            continue
        scores.append(float(np.dot(embedding_a, embedding_b)))
        labels.append(1 if pair.same else 0)

    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    accuracy, threshold = best_threshold_accuracy(score_array, label_array)
    return EvaluationResult(
        pairs=len(pairs),
        evaluated=len(scores),
        accuracy=accuracy,
        threshold=threshold,
        auc=roc_auc(score_array, label_array),
    )


def _embedding(embedder: Embedder, path: Path) -> np.ndarray | None:
    faces = embedder.embed(load_image(path))
    if not faces:
        return None
    return max(faces, key=lambda face: face.detection_score).embedding
