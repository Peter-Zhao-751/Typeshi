"""A one-at-a-time job queue with progress, cancellation and change waiting.

Generation cannot run inside an HTTP handler here. Without the CUDA linear-
attention kernels this architecture decodes at roughly 15 tok/s, and the
grammar spends two tokens per keystroke, so a single sentence is tens of
seconds and a paragraph is minutes. A blocking request would give no
progress, no cancel, and a browser timeout on the long ones.

Exactly one job runs at a time, which is also a correctness requirement
rather than only a memory one: generate() calls torch.manual_seed() as
global process state, so two concurrent generations would silently corrupt
each other's reproducibility.
"""

from __future__ import annotations

import itertools
import threading
import time
from typing import Callable

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
ERROR = "error"
CANCELLED = "cancelled"

_TERMINAL = (DONE, ERROR, CANCELLED)


class Job:
    def __init__(self, job_id: str, kind: str, request: dict) -> None:
        self.id = job_id
        self.kind = kind
        self.request = request
        self.state = QUEUED
        self.progress: dict = {}
        self.result: dict | None = None
        self.error: str | None = None
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.stop_event = threading.Event()
        self._cond = threading.Condition()
        self._version = 0

    # -- mutation -------------------------------------------------------
    def _bump(self) -> None:
        with self._cond:
            self._version += 1
            self._cond.notify_all()

    def set_state(self, state: str) -> None:
        self.state = state
        if state == RUNNING and self.started is None:
            self.started = time.time()
        if state in _TERMINAL:
            self.finished = time.time()
        self._bump()

    def set_progress(self, **fields) -> None:
        self.progress.update(fields)
        self._bump()

    def cancel(self) -> None:
        self.stop_event.set()
        if self.state == QUEUED:
            self.set_state(CANCELLED)
        else:
            self._bump()

    @property
    def cancelled(self) -> bool:
        return self.stop_event.is_set()

    # -- observation ----------------------------------------------------
    def snapshot(self) -> dict:
        elapsed = (self.finished or time.time()) - (self.started or self.created)
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "progress": dict(self.progress),
            "result": self.result,
            "error": self.error,
            "elapsed_s": round(elapsed, 2),
        }

    def wait_for_change(self, since: int, timeout: float) -> tuple[int, dict]:
        """Blocks until the job changes past version `since`, or times out.

        Returns (version, snapshot) either way, so a timed-out caller still
        emits a heartbeat rather than dropping the connection.
        """
        with self._cond:
            if self._version <= since:
                self._cond.wait(timeout)
            return self._version, self.snapshot()

    @property
    def version(self) -> int:
        with self._cond:
            return self._version


class JobQueue:
    def __init__(self) -> None:
        self._ids = itertools.count(1)
        self._jobs: dict[str, Job] = {}
        self._pending: list[tuple[Job, Callable[[Job], dict]]] = []
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def submit(self, kind: str, request: dict,
               runner: Callable[[Job], dict]) -> Job:
        job = Job(f"j{next(self._ids)}", kind, request)
        with self._lock:
            self._jobs[job.id] = job
            self._pending.append((job, runner))
            self._prune_locked()
            self._wake.notify()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def listing(self) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [j.snapshot() for j in sorted(jobs, key=lambda j: j.created,
                                             reverse=True)]

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._pending)

    def _prune_locked(self, keep: int = 40) -> None:
        finished = sorted(
            (j for j in self._jobs.values() if j.state in _TERMINAL),
            key=lambda j: j.finished or 0,
        )
        for job in finished[:max(0, len(finished) - keep)]:
            self._jobs.pop(job.id, None)

    def _run(self) -> None:
        while True:
            with self._lock:
                while not self._pending:
                    self._wake.wait()
                job, runner = self._pending.pop(0)
            if job.cancelled:
                job.set_state(CANCELLED)
                continue
            job.set_state(RUNNING)
            try:
                job.result = runner(job)
            except BaseException as exc:  # noqa: BLE001
                # BaseException so a SystemExit raised deep in a loader path
                # becomes a job error instead of silently killing the only
                # worker thread and hanging every later request.
                job.error = f"{type(exc).__name__}: {exc}"
                job.set_state(CANCELLED if job.cancelled else ERROR)
                continue
            # A cancelled run still carries its partial session -- the stream
            # decoded up to the stop is worth showing -- but the STATE must
            # say cancelled, or the UI would report a truncated stream as a
            # finished one.
            job.set_state(CANCELLED if job.cancelled else DONE)
