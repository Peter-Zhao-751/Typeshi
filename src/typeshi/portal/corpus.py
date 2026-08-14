"""Held-out real sessions, for comparing the model against actual humans."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from typeshi.events import Event

DEFAULT_HELDOUT = Path("data/processed/heldout_aalto")


@dataclass(frozen=True)
class RealSession:
    writer: str
    target: str
    events: list[Event]


def load_test_writers(split_path: Path) -> set[str]:
    """Writer IDs held out at dataset-build time.

    The checkpoint ships its OWN split.json and that is the one that matters:
    a session held out for a different run is training data for this model,
    and comparing against it would flatter the model for memorising.
    """
    try:
        data = json.loads(Path(split_path).read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    return set(data.get("test_writers") or [])


def resolve_split(checkpoint: Path | None, fallback: Path) -> Path | None:
    if checkpoint is not None and (Path(checkpoint) / "split.json").exists():
        return Path(checkpoint) / "split.json"
    return Path(fallback) if Path(fallback).exists() else None


class AaltoHeldout:
    """Lazily indexed pool of held-out Aalto transcription sessions.

    Files are only parsed when a session is actually asked for -- the pool is
    ~14.7k logs and eagerly reading them would add minutes to startup for a
    panel the user may never open.
    """

    def __init__(self, root: Path = DEFAULT_HELDOUT,
                 test_writers: set[str] | None = None) -> None:
        self.root = Path(root)
        self.test_writers = test_writers or set()
        self._files: list[Path] | None = None

    @property
    def available(self) -> bool:
        return self.root.exists()

    def files(self) -> list[Path]:
        if self._files is None:
            if not self.root.exists():
                self._files = []
            else:
                found = sorted(self.root.rglob("*_keystrokes.txt"))
                if self.test_writers:
                    found = [
                        p for p in found
                        if f"aalto:{p.name.split('_')[0]}" in self.test_writers
                    ]
                self._files = found
        return self._files

    def count(self) -> int:
        return len(self.files())

    def sample(self, n: int = 1, seed: int = 0,
               min_chars: int = 8) -> list[RealSession]:
        """Up to `n` real sessions drawn from the held-out pool."""
        from typeshi.adapters import aalto

        files = self.files()
        if not files:
            return []
        rng = random.Random(seed)
        order = list(range(len(files)))
        rng.shuffle(order)

        out: list[RealSession] = []
        for idx in order:
            path = files[idx]
            try:
                sessions = list(aalto.iter_sessions(path))
            except Exception:  # noqa: BLE001 - a corrupt log must not stop us
                continue
            rng.shuffle(sessions)
            for writer, target, events in sessions:
                if len(target) < min_chars or not events:
                    continue
                out.append(RealSession(f"aalto:{writer}", target, events))
                break
            if len(out) >= n:
                break
        return out[:n]
