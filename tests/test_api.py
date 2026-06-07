from __future__ import annotations

from tests.conftest import make_face_png


def test_health_reports_status(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["indexed_vectors"] == 0


def test_enroll_and_recognize_flow(client):
    enroll = client.post(
        "/enroll",
        data={"name": "alice"},
        files={"file": ("alice.png", make_face_png(1), "image/png")},
    )
    assert enroll.status_code == 200
    assert enroll.json()["embeddings_added"] == 1

    recognize = client.post(
        "/recognize",
        files={"file": ("query.png", make_face_png(1), "image/png")},
    )
    assert recognize.status_code == 200
    faces = recognize.json()["faces"]
    assert faces[0]["name"] == "alice"


def test_enroll_without_face_returns_422(client):
    response = client.post(
        "/enroll",
        data={"name": "ghost"},
        files={"file": ("ghost.png", make_face_png(255), "image/png")},
    )
    assert response.status_code == 422


def test_verify_endpoint_matches_same_identity(client):
    response = client.post(
        "/verify",
        files={
            "file_a": ("a.png", make_face_png(3), "image/png"),
            "file_b": ("b.png", make_face_png(3), "image/png"),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["is_match"] is True
    assert body["similarity"] > 0.99


def test_identities_endpoint_lists_enrolled(client):
    client.post(
        "/enroll",
        data={"name": "bob"},
        files={"file": ("bob.png", make_face_png(4), "image/png")},
    )
    response = client.get("/identities")
    assert response.status_code == 200
    assert response.json()["total"] == 1
