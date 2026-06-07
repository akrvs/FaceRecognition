from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


class Gallery:
    def __init__(self) -> None:
        self._label_to_name: dict[int, str] = {}
        self._next_label = 0

    def add(self, name: str) -> int:
        label = self._next_label
        self._label_to_name[label] = name
        self._next_label += 1
        return label

    def name_for(self, label: int) -> str | None:
        return self._label_to_name.get(label)

    def counts(self) -> dict[str, int]:
        return dict(Counter(self._label_to_name.values()))

    @property
    def total_identities(self) -> int:
        return len(set(self._label_to_name.values()))

    def __len__(self) -> int:
        return len(self._label_to_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_label": self._next_label,
            "labels": {str(label): name for label, name in self._label_to_name.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Gallery:
        gallery = cls()
        labels: dict[str, str] = payload.get("labels", {})
        gallery._label_to_name = {int(key): str(name) for key, name in labels.items()}
        gallery._next_label = int(payload.get("next_label", len(gallery._label_to_name)))
        return gallery

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> Gallery:
        return cls.from_dict(json.loads(Path(path).read_text()))
