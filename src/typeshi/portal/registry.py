"""Finding checkpoints on disk and loading one without killing the server."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

from typeshi.train_motor import select_backend


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    kind: str  # "adapter" | "full"
    base_model: str | None
    mtime: float
    bytes: int
    corpora: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return str(self.path)

    @property
    def modes(self) -> tuple[str, ...]:
        """Modes this checkpoint was actually trained for.

        Derived from which corpora its split covers: Aalto is the
        transcription task, KLiCKe is the composition one. A checkpoint with
        no KLiCKe in its split has never seen `<MODE:C>`, so composition on it
        is out of distribution -- the convergence mask would still force it to
        type the target, but the timing and revision behaviour would come from
        the transcription distribution and read as a model failure rather than
        a mode mismatch. Unknown provenance reports both rather than guessing.
        """
        if not self.corpora:
            return ("transcription", "composition")
        modes = []
        if "aalto" in self.corpora:
            modes.append("transcription")
        if "klicke" in self.corpora:
            modes.append("composition")
        return tuple(modes) or ("transcription", "composition")

    def as_json(self) -> dict:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "base_model": self.base_model,
            "mtime": self.mtime,
            "gb": round(self.bytes / 1e9, 2),
            "corpora": list(self.corpora),
            "modes": list(self.modes),
        }


def _weights_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in d.glob("*.safetensors"))


_CORPORA_CACHE: dict[tuple[str, float, int], tuple[str, ...]] = {}


def split_corpora(split_path: Path) -> tuple[str, ...]:
    """Corpus prefixes named in a checkpoint's split, e.g. ("aalto", "klicke").

    Cached on (path, mtime, size): discover() runs on every /api/info, which
    the browser polls every two seconds during a load, and motor-full's
    split.json is 2.3 MB of writer ids.
    """
    try:
        stat = split_path.stat()
    except OSError:
        return ()
    key = (str(split_path), stat.st_mtime, stat.st_size)
    if key not in _CORPORA_CACHE:
        try:
            data = json.loads(split_path.read_text())
        except (OSError, json.JSONDecodeError):
            return ()
        found = set()
        for group in ("test_writers", "train_writers"):
            for writer in data.get(group) or ():
                if ":" in writer:
                    found.add(writer.split(":", 1)[0])
        _CORPORA_CACHE[key] = tuple(sorted(found))
    return _CORPORA_CACHE[key]


def describe(path: Path) -> CheckpointInfo | None:
    """CheckpointInfo for a directory, or None if it holds no loadable save."""
    path = Path(path)
    corpora = split_corpora(path / "split.json")
    adapter = path / "adapter_config.json"
    if adapter.exists():
        base = None
        try:
            base = json.loads(adapter.read_text()).get("base_model_name_or_path")
        except (OSError, json.JSONDecodeError):
            pass
        return CheckpointInfo(path, "adapter", base, path.stat().st_mtime,
                              _weights_bytes(path), corpora)
    if (path / "model.safetensors").exists():
        return CheckpointInfo(path, "full", None, path.stat().st_mtime,
                              _weights_bytes(path), corpora)
    return None


def discover(root: Path) -> list[CheckpointInfo]:
    """Every loadable save under `root`, newest first.

    Accepts BOTH adapter directories and full-weights ones -- the original
    playground globbed only `model.safetensors`, which made every motor-*
    checkpoint invisible to it. Mid-run `checkpoint-N` saves are included so
    the portal stays useful while a training run is still going, and `root`
    itself competes on mtime because a finished run writes the final model
    straight there.
    """
    root = Path(root)
    if not root.exists():
        return []
    found: list[CheckpointInfo] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        info = describe(d)
        if info is not None:
            found.append(info)
        for sub in sorted(d.glob("checkpoint-*")):
            if sub.is_dir():
                sub_info = describe(sub)
                if sub_info is not None:
                    found.append(sub_info)
    return sorted(found, key=lambda i: i.mtime, reverse=True)


def inference_backend(has_cuda: bool, has_bf16: bool, has_mps: bool) -> dict:
    """select_backend, but for inference rather than training.

    The training policy pins fp32 on MPS. That is right for a Trainer and
    wrong here: fp32 costs ~17 GB for the resized Qwen3.5-4B where bf16 costs
    ~8.5 GB, and since both the base shards and the adapter's embedding
    tensors are stored bf16 on disk, loading bf16 is LOSSLESS -- fp32 merely
    up-converts them. The documented Apple-Silicon segfault is specifically
    bfloat16 COMBINED WITH device_map="auto"; this path keeps device_map=None
    and moves the model afterwards, so that combination never forms.

    float16 stays excluded on purpose: the qwen3_5 gated-delta kernels carry
    an explicit "A might be -inf" warning under fp16, and fp16 hangs on MPS
    at real model scale.
    """
    backend = dict(select_backend(has_cuda, has_bf16, has_mps))
    if not has_cuda and has_mps:
        backend["dtype"] = "bfloat16"
    return backend


def detect_inference_backend() -> dict:
    import torch

    return inference_backend(
        has_cuda=torch.cuda.is_available(),
        has_bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        has_mps=torch.backends.mps.is_available(),
    )


def target_device(backend: dict) -> str:
    import torch

    if backend["device_map"] is not None:
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Loaded:
    info: CheckpointInfo
    tok: object
    model: object
    device: str
    dtype: str


class Registry:
    """Holds at most one resident model and loads replacements in the
    background.

    Loading is asynchronous because the port must be listening first: a cold
    start pulls 9.3 GB of base weights and materialises ~8.5 GB of them, and
    the original playground did all that BEFORE binding the socket, so the
    browser saw a dead localhost for minutes with no way to tell whether
    anything was happening.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.loaded: Loaded | None = None
        self.status = "idle"  # idle | loading | ready | error
        self.detail = ""
        self.pending: Path | None = None

    def snapshot(self) -> dict:
        with self.lock:
            out = {"status": self.status, "detail": self.detail}
            if self.loaded is not None:
                out["checkpoint"] = self.loaded.info.as_json()
                out["device"] = self.loaded.device
                out["dtype"] = self.loaded.dtype
            if self.pending is not None:
                out["pending"] = str(self.pending)
            return out

    def _set(self, status: str, detail: str = "") -> None:
        with self.lock:
            self.status = status
            self.detail = detail

    def load(self, path: Path) -> None:
        """Loads synchronously. Call from a worker thread, never a handler."""
        from typeshi.eval.load import load_checkpoint_model, load_checkpoint_tokenizer

        path = Path(path)
        info = describe(path)
        if info is None:
            self._set("error", f"{path} holds no adapter_config.json or "
                               "model.safetensors")
            return
        with self.lock:
            self.pending = path
        self._set("loading", f"loading {path}")
        try:
            self.release()
            backend = detect_inference_backend()
            tok = load_checkpoint_tokenizer(path)
            self._set("loading", f"loading weights for {path} "
                                 f"({backend['dtype']}) -- first run for a new "
                                 "base model downloads it")
            model = load_checkpoint_model(path, backend)
            device = target_device(backend)
            if backend["device_map"] is None and device != "cpu":
                self._set("loading", f"moving {path} to {device}")
                model = model.to(device)
            model.eval()
        except BaseException as exc:  # noqa: BLE001
            # BaseException, not Exception: load_checkpoint_tokenizer raises
            # SystemExit when its byte-exactness probe fails, which would
            # otherwise unwind straight past this thread and take the server
            # with it, leaving the browser with no error to show.
            with self.lock:
                self.pending = None
            self._set("error", f"{type(exc).__name__}: {exc}")
            return
        with self.lock:
            self.loaded = Loaded(info, tok, model, device, backend["dtype"])
            self.pending = None
            self.status = "ready"
            self.detail = ""

    def load_async(self, path: Path) -> None:
        threading.Thread(target=self.load, args=(path,), daemon=True).start()

    def release(self) -> None:
        """Drops the resident model and returns its memory to the allocator."""
        import gc

        with self.lock:
            if self.loaded is None:
                return
            self.loaded = None
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cache eviction is best-effort
            pass
