from __future__ import annotations

import numpy as np
import pytest

from visage.evaluation.metrics import best_threshold_accuracy, roc_auc


def test_roc_auc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(scores, labels) == pytest.approx(1.0)


def test_roc_auc_chance_level_is_half():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array([0, 1, 1, 0])
    assert roc_auc(scores, labels) == pytest.approx(0.5)


def test_roc_auc_handles_ties():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 1, 0, 1])
    assert roc_auc(scores, labels) == pytest.approx(0.5)


def test_best_threshold_accuracy_separable():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    accuracy, threshold = best_threshold_accuracy(scores, labels)
    assert accuracy == pytest.approx(1.0)
    assert 0.2 < threshold < 0.8
