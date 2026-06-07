from __future__ import annotations

from visage.config import Settings
from visage.service.gallery import Gallery


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("VISAGE_MATCH_THRESHOLD", "0.6")
    monkeypatch.setenv("VISAGE_MODEL_NAME", "buffalo_s")
    settings = Settings()
    assert settings.match_threshold == 0.6
    assert settings.model_name == "buffalo_s"


def test_gallery_assigns_incrementing_labels():
    gallery = Gallery()
    assert gallery.add("alice") == 0
    assert gallery.add("bob") == 1
    assert gallery.name_for(0) == "alice"
    assert gallery.total_identities == 2


def test_gallery_counts_embeddings_per_identity():
    gallery = Gallery()
    gallery.add("alice")
    gallery.add("alice")
    gallery.add("bob")
    assert gallery.counts() == {"alice": 2, "bob": 1}


def test_gallery_roundtrip(tmp_path):
    gallery = Gallery()
    gallery.add("alice")
    gallery.add("bob")
    path = tmp_path / "gallery.json"
    gallery.save(path)

    restored = Gallery.load(path)
    assert restored.name_for(0) == "alice"
    assert restored.name_for(1) == "bob"
    assert restored.add("carol") == 2
