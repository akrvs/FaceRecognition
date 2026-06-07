from __future__ import annotations

import numpy as np


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)

    sorted_scores = scores[order]
    sorted_ranks = ranks[order]
    start = 0
    for end in range(1, scores.size + 1):
        if end == scores.size or sorted_scores[end] != sorted_scores[start]:
            sorted_ranks[start:end] = (start + end + 1) / 2.0
            start = end
    ranks[order] = sorted_ranks

    rank_sum_positive = ranks[labels == 1].sum()
    n_pos = float(positives.size)
    n_neg = float(negatives.size)
    return float((rank_sum_positive - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def best_threshold_accuracy(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(scores)
    midpoints = (candidates[:-1] + candidates[1:]) / 2.0
    thresholds = np.concatenate(([candidates[0] - 1e-6], midpoints, [candidates[-1] + 1e-6]))

    best_accuracy = 0.0
    best_threshold = float(thresholds[0])
    for threshold in thresholds:
        predictions = scores >= threshold
        accuracy = float(np.mean(predictions == (labels == 1)))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_accuracy, best_threshold
