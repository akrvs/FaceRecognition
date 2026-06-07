from __future__ import annotations

import pytest

from tests.conftest import NO_FACE, make_face_array
from visage.service.recognizer import NoFaceDetectedError


def test_enroll_then_recognize_same_identity(recognizer):
    recognizer.enroll("alice", make_face_array(1))

    faces = recognizer.recognize(make_face_array(1))

    assert len(faces) == 1
    assert faces[0].name == "alice"
    assert faces[0].score == pytest.approx(1.0, abs=1e-5)


def test_recognize_unknown_identity_returns_none(recognizer):
    recognizer.enroll("alice", make_face_array(1))

    faces = recognizer.recognize(make_face_array(2))

    assert faces[0].name is None


def test_recognize_without_faces_returns_empty(recognizer):
    assert recognizer.recognize(make_face_array(NO_FACE)) == []


def test_enroll_without_face_raises(recognizer):
    with pytest.raises(NoFaceDetectedError):
        recognizer.enroll("ghost", make_face_array(NO_FACE))


def test_verify_same_identity_is_high(recognizer):
    similarity = recognizer.verify(make_face_array(7), make_face_array(7))
    assert similarity == pytest.approx(1.0, abs=1e-5)


def test_verify_different_identity_is_low(recognizer):
    similarity = recognizer.verify(make_face_array(7), make_face_array(8))
    assert similarity < 0.9


def test_identities_summary_counts_embeddings(recognizer):
    recognizer.enroll("alice", make_face_array(1))
    recognizer.enroll("alice", make_face_array(1))
    recognizer.enroll("bob", make_face_array(2))

    summary = recognizer.identities()

    assert summary.total == 2
    counts = {item.name: item.embeddings for item in summary.identities}
    assert counts == {"alice": 2, "bob": 1}


def test_threshold_override_rejects_weak_match(recognizer):
    recognizer.enroll("alice", make_face_array(1))
    faces = recognizer.recognize(make_face_array(1), threshold=1.5)
    assert faces[0].name is None


def test_persistence_roundtrip(tmp_path, recognizer, fake_embedder):
    recognizer.enroll("alice", make_face_array(1))
    index_path = tmp_path / "index.npz"
    gallery_path = tmp_path / "gallery.json"
    recognizer.save(index_path, gallery_path)

    from visage.index import NumpyIndex
    from visage.service.recognizer import FaceRecognizer

    restored = FaceRecognizer(fake_embedder, NumpyIndex(fake_embedder.embedding_dim))
    restored.load(index_path, gallery_path)

    assert restored.recognize(make_face_array(1))[0].name == "alice"
