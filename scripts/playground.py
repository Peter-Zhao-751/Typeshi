"""Local playground: watch the motor model type, live, on a fake keyboard.

    uv run python scripts/playground.py          # newest checkpoint, CPU
    uv run python scripts/playground.py --checkpoint checkpoints/tiny-pilot-100k-e2

Then open http://localhost:8765.

Runs the model on CPU by default ON PURPOSE: a training run usually owns the
MPS device, and a playground that steals it would slow the real work. Pass
--device mps when nothing is training.

Stdlib HTTP only -- no new dependencies for a toy.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
MAX_CHARS = 300  # CPU generation is a few seconds per sentence; keep it snappy


def _supported_chars() -> frozenset[str]:
    from typeshi.tiny_tokenizer import TEXT_CHARS

    return frozenset(TEXT_CHARS)


def newest_checkpoint(root: Path) -> Path | None:
    """The most recently written save under `root`, by mtime.

    Mid-run checkpoints are what makes this fun while training: point it at
    checkpoints/motor-tiny and it picks up tonight's latest save. But
    train_tiny also writes the FINAL model straight to `root` when a run
    completes, so `root` must compete on mtime alongside the checkpoint-N
    dirs -- otherwise a finished run's playground silently serves a stale
    intermediate checkpoint.
    """
    if not root.exists():
        return None
    saves = [p for p in root.glob("checkpoint-*") if (p / "model.safetensors").exists()]
    if (root / "model.safetensors").exists():
        saves.append(root)
    return max(saves, key=lambda p: p.stat().st_mtime) if saves else None


class Playground:
    def __init__(self, checkpoint: Path, device: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        from typeshi.eval.load import load_checkpoint_tokenizer

        self.checkpoint = checkpoint
        self.device = device
        self.lock = threading.Lock()  # one model, one generation at a time
        # load_checkpoint_tokenizer, not AutoTokenizer: Auto resolves this
        # checkpoint to Qwen2Tokenizer, which silently eats every space.
        self.tok = load_checkpoint_tokenizer(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, dtype=torch.float32
        )
        if device != "cpu":
            self.model = self.model.to(device)
        self.model.eval()

    def generate(self, text: str, wpm: float, ecor: float, eunc: float,
                 temperature: float, seed: int) -> dict:
        from typeshi.buffer import replay
        from typeshi.events import EventType
        from typeshi.generate import generate
        from typeshi.labels import SessionLabels, _levenshtein

        # The knobs are the same conditioning the model saw in training:
        # error rates are fractions in [0,1], binned to whole percents.
        labels = SessionLabels(
            wpm=wpm,
            corrected_error_rate=ecor / 100,
            uncorrected_error_rate=eunc / 100,
            revision_rate=0.0,
        )
        with self.lock:
            events = generate(
                self.model, self.tok, text, labels, mode="transcription",
                temperature=temperature, max_new_tokens=4 * len(text) + 64,
                seed=seed,
            )

        rows = []
        for e in events:
            rows.append({
                "type": "bksp" if e.type is EventType.BACKSPACE else "key",
                "char": e.char,
                "press": round(float(e.press_time), 1),
                "release": round(float(e.release_time), 1),
            })
        produced = replay(events)
        duration = float(events[-1].press_time - events[0].press_time) if events else 0.0
        minutes = duration / 60_000
        return {
            "target": text,
            "produced": produced,
            "similarity": round(
                1 - _levenshtein(produced, text) / max(len(text), 1), 3
            ),
            "duration_ms": round(duration, 1),
            "realized_wpm": round((len(produced) / 5) / minutes, 1) if minutes else 0.0,
            "backspaces": sum(1 for r in rows if r["type"] == "bksp"),
            "events": rows,
        }


def make_handler(pg: Playground):
    SUPPORTED_CHARS = _supported_chars()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json")

        def log_message(self, fmt, *args):  # quieter console
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                page = HERE / "playground.html"
                if not page.exists():
                    self._json(500, {"error": "playground.html missing"})
                    return
                self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._json(200, {
                    "checkpoint": str(pg.checkpoint),
                    "device": pg.device,
                    "max_chars": MAX_CHARS,
                })
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/generate":
                self._json(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._json(400, {"error": "bad JSON"})
                return

            text = (req.get("text") or "").strip()
            if not text:
                self._json(400, {"error": "text is empty"})
                return
            if len(text) > MAX_CHARS:
                self._json(400, {"error": f"text longer than {MAX_CHARS} chars"})
                return
            # The vocabulary is 97 ASCII identities: a curly quote or emoji has
            # no token at all and the tokenizer raises a plain pyo3 Exception,
            # not ValueError. Name the offenders instead of failing obscurely.
            bad = sorted(set(text) - SUPPORTED_CHARS)
            if bad:
                self._json(400, {
                    "error": "unsupported characters (the model's vocabulary is "
                             f"ASCII-only): {' '.join(repr(c) for c in bad)}"
                })
                return
            try:
                result = pg.generate(
                    text,
                    wpm=float(req.get("wpm", 60)),
                    ecor=float(req.get("ecor", 2)),
                    eunc=float(req.get("eunc", 1)),
                    temperature=float(req.get("temperature", 1.0)),
                    seed=int(req.get("seed", 0)),
                )
            except Exception as exc:  # noqa: BLE001
                # Broad on purpose: a handler thread that dies takes the whole
                # request down with an empty reply, and the tokenizer/grammar
                # layers raise several unrelated types.
                self._json(400, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._json(200, result)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="defaults to the newest save under checkpoints/motor-tiny")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    ckpt = args.checkpoint or newest_checkpoint(Path("checkpoints/motor-tiny"))
    if ckpt is None or not ckpt.exists():
        raise SystemExit(
            "no checkpoint found; pass --checkpoint (e.g. "
            "checkpoints/tiny-pilot-100k-e2)"
        )
    print(f"loading {ckpt} on {args.device} ...")
    pg = Playground(ckpt, args.device)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(pg))
    print(f"playground ready:  http://localhost:{args.port}   (ctrl-c to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
