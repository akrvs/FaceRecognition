from __future__ import annotations

import numpy as np
import pytest

from visage.index import NumpyIndex


def _unit(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_search_returns_nearest_label():
    index = NumpyIndex(3)
    index.add(np.vstack([_unit([1, 0, 0]), _unit([0, 1, 0])]), [10, 20])

    results = index.search(_unit([0.9, 0.1, 0]), k=1)

    assert results[0][0].label == 10
    assert results[0][0].score == pytest.approx(0.9938, abs=1e-3)


def test_search_empty_index_returns_empty_rows():
    index = NumpyIndex(3)
    assert index.search(_unit([1, 0, 0]), k=5) == [[]]


def test_search_k_larger_than_size():
    index = NumpyIndex(2)
    index.add(_unit([1, 0]), [1])
    results = index.search(_unit([1, 0]), k=10)
    assert len(results[0]) == 1


def test_add_validates_label_count():
    index = NumpyIndex(2)
    with pytest.raises(ValueError):
        index.add(np.zeros((2, 2), dtype=np.float32), [1])


def test_add_validates_dimension():
    index = NumpyIndex(2)
    with pytest.raises(ValueError):
        index.add(np.zeros((1, 3), dtype=np.float32), [1])


def test_save_and_load_roundtrip(tmp_path):
    index = NumpyIndex(3)
    index.add(np.vstack([_unit([1, 0, 0]), _unit([0, 1, 0])]), [10, 20])
    path = tmp_path / "index.npz"
    index.save(path)

    restored = NumpyIndex(3)
    restored.load(path)

    assert len(restored) == 2
    assert restored.search(_unit([1, 0, 0]), k=1)[0][0].label == 10
