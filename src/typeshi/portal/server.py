"""HTTP routes for the portal. Stdlib only -- no new dependencies for a toy."""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from typeshi.config import BASE_MODEL
from typeshi.labels import SessionLabels
from typeshi.portal import corpus as corpus_mod
from typeshi.portal import jobs as jobs_mod
from typeshi.portal import readout, registry, rows
from typeshi.serialize import (
    MARKERS,
    PCT_MAX,
    WPM_BIN_WIDTH,
    WPM_BINS,
    pct_bin,
    supported_chars,
    wpm_bin,
)

SUPPORTED = supported_chars()


class PortalError(Exception):
    """A request the user can fix, reported as 400 with a readable message."""


def validate_target(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if not text:
        raise PortalError("target text is empty")
    if len(text) > max_chars:
        raise PortalError(f"target text is longer than {max_chars} characters")
    for marker in MARKERS:
        if marker in text:
            raise PortalError(
                f"target text contains the prompt marker {marker} -- "
                "build_prompt refuses these because they would be "
                "indistinguishable from real structure"
            )
    bad = sorted(set(text) - SUPPORTED)
    if bad:
        raise PortalError(
            "no keystroke token exists for: "
            + " ".join(repr(c) for c in bad)
            + ". The model can only type the 97 printable-ASCII identities "
            "(a curly quote or em dash pasted from a word processor is the "
            "usual cause)."
        )
    return text


def labels_from(req: dict) -> SessionLabels:
    """Percent-valued UI knobs become the fractions SessionLabels wants."""
    return SessionLabels(
        wpm=float(req.get("wpm", 60)),
        corrected_error_rate=float(req.get("ecor", 2)) / 100,
        uncorrected_error_rate=float(req.get("eunc", 1)) / 100,
        revision_rate=float(req.get("rev", 0)) / 100,
    )


# The library default is 4. That is too loose here for a structural reason:
# stage 2 measures off-path depth as edit distance to the best target prefix,
# which says how WRONG the buffer is, not how far BACK the wrongness sits. One
# early typo reads as depth 1 forever, however many correct characters follow,
# so the excursion guard never fires -- while the only repair the model
# actually emits is backspace, whose cost is the whole distance from the
# cursor back to the error. Measured on a 197-char paragraph: budget 4 gave
# runs of 136-141 consecutive backspaces and a 30-36% backspace rate; budget 1
# gave 205 events, 2% backspaces, longest run 1.
DEFAULT_EXCURSION_BUDGET = 1


def default_budget(text: str, mode: str) -> int:
    """Token budget. Two tokens per keystroke, plus room to be human.

    Transcription reuses the eval's 4*len+64. Composition gets far more
    headroom: under the convergence mask every typo excursion costs two extra
    events to make and two more to undo, and a run that exhausts its budget
    is a failed attempt rather than a shorter one.
    """
    return 4 * len(text) + 64 if mode == "transcription" else 8 * len(text) + 256


class Portal:
    def __init__(self, html: Path, checkpoint_root: Path, max_chars: int) -> None:
        self.html = Path(html)
        self.checkpoint_root = Path(checkpoint_root)
        self.max_chars = max_chars
        self.registry = registry.Registry()
        self.queue = jobs_mod.JobQueue()
        self._band: dict[str, list[float]] | None = None
        self._corpus: corpus_mod.AaltoHeldout | None = None
        self._corpus_key: object = object()
        # Unfiltered pool, used ONLY for the percentile band. The band
        # describes how humans type, not how this checkpoint performs, so it
        # wants sample size rather than held-out purity -- and it needs it:
        # motor-phase2's split matches just 23 of the 14,765 local Aalto logs,
        # which would put p10/p90 on 23 points.
        self._band_pool = corpus_mod.AaltoHeldout()

    # -- lazily built helpers -------------------------------------------
    def corpus(self) -> corpus_mod.AaltoHeldout:
        """Held-out pool for the CURRENTLY loaded checkpoint.

        Keyed on the checkpoint path rather than invalidated by hand: the
        held-out writers are checkpoint-bound, and a stale pool would silently
        compare the model against sessions it was trained on. Keying also
        covers the case where the pool is built while a model is still
        loading, which no explicit invalidation call would catch.
        """
        loaded = self.registry.loaded
        ckpt = loaded.info.path if loaded else None
        if self._corpus is None or self._corpus_key != ckpt:
            split = corpus_mod.resolve_split(ckpt, Path("data/processed/split.json"))
            writers = corpus_mod.load_test_writers(split) if split else set()
            self._corpus = corpus_mod.AaltoHeldout(test_writers=writers)
            self._corpus_key = ckpt
            self._band = None
        return self._corpus

    def band(self, n: int = 120) -> dict[str, list[float]]:
        """Real-human spread for the serial gauges, computed once on demand."""
        if self._band is None:
            sessions = [s.events for s in self._band_pool.sample(n, seed=17)]
            self._band = readout.real_band(sessions)
            self._band_n = len(sessions)
        return self._band

    # -- payloads --------------------------------------------------------
    def info(self) -> dict:
        snap = self.registry.snapshot()
        c = self.corpus()
        return {
            **snap,
            "checkpoints": [i.as_json() for i in
                            registry.discover(self.checkpoint_root)],
            "max_chars": self.max_chars,
            "supported_chars": "".join(sorted(SUPPORTED)),
            "markers": list(MARKERS),
            "limits": {
                "wpm_bins": WPM_BINS,
                "wpm_bin_width": WPM_BIN_WIDTH,
                "wpm_max": WPM_BINS * WPM_BIN_WIDTH - WPM_BIN_WIDTH,
                "pct_max": PCT_MAX,
            },
            "corpus": {"available": c.available, "count": c.count()},
            "queue_depth": self.queue.queue_depth(),
            "base_model": BASE_MODEL,
        }

    def prompt_preview(self, req: dict) -> dict:
        """What the model literally sees, plus what each knob quantized to.

        Every conditioning knob is binned before it reaches the model -- WPM
        into 40 buckets, the rates into whole percents capped at 30 -- so a
        slider can move several steps without changing the prompt at all.
        Showing the rendered tokens is the only honest feedback for "did that
        drag do anything".
        """
        from typeshi.dataset import build_prompt

        mode = req.get("mode", "transcription")
        text = validate_target(req.get("text", ""), self.max_chars)
        labels = labels_from(req)
        return {
            "prompt": build_prompt(text, labels, mode),
            "tokens": labels.to_tokens(mode),
            "bins": {
                "wpm": wpm_bin(labels.wpm),
                "ecor": pct_bin(labels.corrected_error_rate),
                "eunc": pct_bin(labels.uncorrected_error_rate),
                "rev": pct_bin(labels.revision_rate),
            },
        }

    # -- generation ------------------------------------------------------
    def submit_generate(self, req: dict) -> dict:
        loaded = self.registry.loaded
        if loaded is None:
            raise PortalError(
                f"no model loaded (status: {self.registry.status})"
            )
        mode = req.get("mode", "transcription")
        if mode not in ("transcription", "composition"):
            raise PortalError(f"unknown mode {mode!r}")
        text = validate_target(req.get("text", ""), self.max_chars)
        labels = labels_from(req)
        constrained = bool(req.get("constrained", True))
        budget = int(req.get("max_new_tokens") or default_budget(text, mode))
        # Convergence is the default because the alternative silently hands
        # back the wrong text. The grammar mask only guarantees a well-formed
        # keystroke stream -- nothing checks it against the target, so a
        # fumbled word is never corrected and the run ends on "spher".
        converge = bool(req.get("converge", True)) and constrained

        def runner(job: jobs_mod.Job) -> dict:
            if converge:
                return self.run_converged(job, loaded, text, labels, mode, req)
            return self.run_generate(job, loaded, text, labels, mode,
                                     constrained, budget, req)

        job = self.queue.submit("generate", {**req, "text": text}, runner)
        return {"job_id": job.id, "max_steps": budget}

    def _observer(self, job, budget: int | None):
        started = time.time()

        def observe(step: int, buffer_text: str) -> None:
            elapsed = time.time() - started
            rate = step / elapsed if elapsed > 0 else 0.0
            eta = round((budget - step) / rate, 1) if (budget and rate > 0) else None
            job.set_progress(step=step, max_steps=budget, buffer=buffer_text,
                             tok_per_s=round(rate, 2), eta_s=eta)

        return observe

    def _payload(self, events, text, mode, req, **extra) -> dict:
        band = self.band()
        return {
            "session": rows.session_payload(events, text),
            "mode": mode,
            "validity": readout.validity(events, text, mode),
            "serial": readout.serial_readout(events, band),
            "controls": readout.controls(
                events, text, float(req.get("wpm", 60)), int(req.get("seed", 0))
            ),
            "band": band,
            "features": [
                {"key": k, "label": lab, "null": null}
                for k, lab, null in readout.SERIAL_FEATURES
            ],
            **extra,
        }

    def run_converged(self, job, loaded, text, labels, mode, req) -> dict:
        """The guaranteed path: windowed convergence decoding.

        Generates in the ~512-event windows composition was trained on and
        cannot finish on anything but `text` exactly. On failure it still
        returns the partial stream, named -- a stalled run is far more
        informative shown than swallowed.
        """
        from typeshi.generate import ConvergenceError, generate_windowed

        seen = {"steps": 0}
        report = self._observer(job, None)

        def observe(step: int, buffer_text: str) -> None:
            seen["steps"] = step
            report(step, buffer_text)

        failure = None
        converged = True
        try:
            events = generate_windowed(
                loaded.model, loaded.tok, text, labels,
                temperature=float(req.get("temperature", 1.0)),
                seed=int(req.get("seed", 0)),
                mode=mode,
                excursion_budget=int(req.get("excursion_budget",
                                            DEFAULT_EXCURSION_BUDGET)),
                resolve_progress=int(req.get("resolve_progress", 2)),
                observer=observe,
                stop_event=job.stop_event,
            )
        except ConvergenceError as exc:
            events, converged, failure = exc.events, False, str(exc)

        return self._payload(
            events, text, mode, req,
            constrained=True,
            converged=converged,
            failure=failure,
            terminated=converged,
            steps=seen["steps"],
            budget=None,
            prompt=None,
            cancelled=job.cancelled,
            decoder="convergence (windowed)",
        )

    def run_generate(self, job, loaded, text, labels, mode, constrained,
                     budget, req) -> dict:
        from typeshi.generate import generate_session

        observe = self._observer(job, budget)

        result = generate_session(
            loaded.model, loaded.tok, text, labels,
            mode=mode,
            temperature=float(req.get("temperature", 1.0)),
            max_new_tokens=budget,
            seed=int(req.get("seed", 0)),
            constrained=constrained,
            excursion_budget=int(req.get("excursion_budget",
                                            DEFAULT_EXCURSION_BUDGET)),
            resolve_progress=int(req.get("resolve_progress", 2)),
            observer=observe,
            stop_event=job.stop_event,
        )
        events = result.events
        produced = rows.replay_safe(events)[0]
        return self._payload(
            events, text, mode, req,
            constrained=constrained,
            converged=produced == text,
            failure=None,
            terminated=result.terminated,
            steps=result.steps,
            budget=budget,
            prompt=result.prompt,
            cancelled=job.cancelled,
            decoder="grammar mask only" if constrained else "unconstrained",
        )

    def corpus_sample(self, seed: int) -> dict:
        # Held-out for THIS checkpoint first; the wider pool only as a
        # labelled fallback, because a session the model trained on would
        # flatter it and the viewer has to be told which one they got.
        held_out = True
        sessions = self.corpus().sample(1, seed=seed)
        if not sessions:
            held_out = False
            sessions = self._band_pool.sample(1, seed=seed)
        if not sessions:
            raise PortalError(
                "no sessions available -- expected Aalto logs under "
                f"{corpus_mod.DEFAULT_HELDOUT}"
            )
        from typeshi.serialize import codec_roundtrip

        s = sessions[0]
        # Through the codec, same as the eval: raw corpus timings live on a
        # finer grid than anything the model can emit, and comparing the two
        # directly is how a discriminator scores 0.915 on quantization alone.
        events = codec_roundtrip(s.events)
        return {
            "writer": s.writer,
            "target": s.target,
            "session": rows.session_payload(events, s.target),
            "serial": readout.serial_readout(events, self.band()),
            "codec_roundtripped": True,
            "held_out": held_out,
            "pool": self.corpus().count() if held_out else self._band_pool.count(),
            "band_n": getattr(self, "_band_n", 0),
        }


def make_handler(portal: Portal):
    class Handler(BaseHTTPRequestHandler):
        server_version = "typeshi-portal"

        def log_message(self, fmt, *args):  # quieter console
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict | list) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json")

        def _body(self) -> dict:
            try:
                n = int(self.headers.get("Content-Length", 0))
                return json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError):
                raise PortalError("malformed JSON body")

        # -- GET --------------------------------------------------------
        def do_GET(self):
            url = urlparse(self.path)
            path, query = url.path, parse_qs(url.query)
            try:
                if path in ("/", "/index.html"):
                    if not portal.html.exists():
                        self._json(500, {"error": f"{portal.html} missing"})
                        return
                    self._send(200, portal.html.read_bytes(),
                               "text/html; charset=utf-8")
                elif path == "/api/info":
                    self._json(200, portal.info())
                elif path == "/api/jobs":
                    self._json(200, portal.queue.listing())
                elif path.startswith("/api/jobs/") and path.endswith("/stream"):
                    self._stream(path.split("/")[3])
                elif path.startswith("/api/jobs/"):
                    job = portal.queue.get(path.split("/")[3])
                    if job is None:
                        self._json(404, {"error": "no such job"})
                    else:
                        self._json(200, job.snapshot())
                elif path == "/api/corpus/sample":
                    seed = int((query.get("seed") or ["0"])[0])
                    self._json(200, portal.corpus_sample(seed))
                else:
                    self._json(404, {"error": "not found"})
            except PortalError as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def _stream(self, job_id: str) -> None:
            """Server-sent events for one job's progress.

            No Content-Length and HTTP/1.0 framing: the browser reads until
            the socket closes, which is what we want since the length is
            unknown until the job ends.
            """
            job = portal.queue.get(job_id)
            if job is None:
                self._json(404, {"error": "no such job"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            version = -1
            try:
                while True:
                    version, snap = job.wait_for_change(version, timeout=10.0)
                    self.wfile.write(
                        f"data: {json.dumps(snap)}\n\n".encode()
                    )
                    self.wfile.flush()
                    if snap["state"] in ("done", "error", "cancelled"):
                        return
            except (BrokenPipeError, ConnectionResetError):
                return  # the tab went away; the job keeps running

        # -- POST -------------------------------------------------------
        def do_POST(self):
            path = urlparse(self.path).path
            try:
                if path == "/api/generate":
                    self._json(200, portal.submit_generate(self._body()))
                elif path == "/api/prompt":
                    self._json(200, portal.prompt_preview(self._body()))
                elif path == "/api/checkpoint":
                    body = self._body()
                    target = body.get("path")
                    if not target:
                        raise PortalError("no checkpoint path given")
                    if portal.registry.status == "loading":
                        raise PortalError("a checkpoint is already loading")
                    portal.registry.load_async(Path(target))
                    self._json(200, portal.registry.snapshot())
                elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                    job = portal.queue.get(path.split("/")[3])
                    if job is None:
                        self._json(404, {"error": "no such job"})
                        return
                    job.cancel()
                    self._json(200, job.snapshot())
                else:
                    self._json(404, {"error": "not found"})
            except PortalError as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                # Broad on purpose: the tokenizer, grammar and convergence
                # layers raise several unrelated types, and a handler thread
                # that dies takes the request down with an empty reply.
                self._json(400, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def serve(portal: Portal, port: int) -> ThreadingHTTPServer:
    """Binds 127.0.0.1 only.

    Not configurable, deliberately: the phase-2 checkpoint is KLiCKe-derived,
    KLiCKe ships no license terms, and anything this server emits inherits
    that hold. A --host flag would be a footgun with no upside on a laptop.
    """
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(portal))
