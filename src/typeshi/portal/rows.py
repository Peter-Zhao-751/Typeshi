"""Turns an Event list into the JSON the browser animates."""

from __future__ import annotations

from typeshi.buffer import ReplayError, TextBuffer
from typeshi.events import Event, EventType
from typeshi.labels import _levenshtein


def event_rows(events: list[Event]) -> list[dict]:
    """One row per event, carrying every field the four event types need.

    The original playground emitted `"bksp" if BACKSPACE else "key"` and an
    unconditional `round(float(e.release_time), 1)`. Both break on
    composition: CURSOR and SELDEL carry release_time=None (TypeError), and
    labelling them "key" with a null char made the client drop them silently,
    so its animated buffer disagreed with the server's replayed text.
    """
    rows = []
    for e in events:
        row = {
            "type": e.type.value,
            "char": e.char,
            "press": round(float(e.press_time), 1),
            "release": None if e.release_time is None
            else round(float(e.release_time), 1),
        }
        if e.type is EventType.CURSOR:
            row["pos"] = e.pos
        elif e.type is EventType.SELDEL:
            row["start"] = e.start
            row["end"] = e.end
        rows.append(row)
    return rows


def span_ms(events: list[Event]) -> float:
    """Wall-clock span of a session, in milliseconds.

    Uses max(release) rather than the last event's press: holds overlap in
    real typing (26% of keystrokes roll over, which the v2 grammar encodes
    deliberately), so release times are NOT sorted and the final key-up can
    belong to an earlier event than the final key-down.
    """
    if not events:
        return 0.0
    starts = [float(e.press_time) for e in events]
    ends = [float(e.release_time) for e in events if e.release_time is not None]
    return max(ends + starts) - min(starts)


def replay_safe(events: list[Event]) -> tuple[str, str | None]:
    """Replays as far as it can, reporting the first invalid event.

    A ReplayError is the interesting thing to SHOW -- unconstrained
    composition emitted out-of-buffer cursor moves in 2 of 5 probe sessions
    -- so the portal renders the partial stream and names the defect instead
    of turning the whole request into a 400.
    """
    buf = TextBuffer()
    for i, e in enumerate(events):
        try:
            buf.apply(e)
        except ReplayError as exc:
            return buf.text, f"event {i} ({e.type.value}): {exc}"
    return buf.text, None


def event_mix(events: list[Event]) -> dict[str, float]:
    if not events:
        return {}
    counts: dict[str, int] = {}
    for e in events:
        counts[e.type.value] = counts.get(e.type.value, 0) + 1
    return {k: round(v / len(events), 4) for k, v in sorted(counts.items())}


def pause_fraction(events: list[Event], threshold_ms: float = 1000.0) -> float:
    """Share of inter-press gaps longer than a second -- the "think pause"
    measure the phase-2 probe reports for real and generated side by side."""
    if len(events) < 2:
        return 0.0
    press = [float(e.press_time) for e in events]
    gaps = [b - a for a, b in zip(press, press[1:])]
    return round(sum(1 for g in gaps if g > threshold_ms) / len(gaps), 4)


def session_stats(events: list[Event], target_text: str) -> dict:
    produced, replay_error = replay_safe(events)
    duration = span_ms(events)
    minutes = duration / 60_000
    return {
        "target": target_text,
        "produced": produced,
        "replay_error": replay_error,
        "similarity": round(
            1 - _levenshtein(produced, target_text) / max(len(target_text), 1), 3
        ),
        "exact": produced == target_text,
        "duration_ms": round(duration, 1),
        "realized_wpm": round((len(produced) / 5) / minutes, 1) if minutes else 0.0,
        "n_events": len(events),
        "backspaces": sum(1 for e in events if e.type is EventType.BACKSPACE),
        "event_mix": event_mix(events),
        "pause_fraction": pause_fraction(events),
    }


def session_payload(events: list[Event], target_text: str) -> dict:
    return {**session_stats(events, target_text), "events": event_rows(events)}
