"""Portal internals that must hold without a model loaded.

Everything here is deliberately model-free: checkpoint discovery, the dtype
policy, request validation, the event->JSON shaping, the job queue and the
per-sample readout are each testable on their own, which is most of why the
portal is a package rather than another 900 lines of script.
"""

import json
import time

import pytest

from typeshi.events import Event


def _wait(job, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if job.state in ("done", "error", "cancelled"):
            return job
        time.sleep(0.01)
    raise AssertionError(f"job stayed {job.state}")


# -- backend policy -----------------------------------------------------

def test_inference_backend_uses_bf16_on_apple_silicon():
    """The training policy pins fp32 on MPS; inference must not inherit it.

    fp32 costs ~17 GB for the resized 4B where bf16 costs ~8.5, and since the
    weights are bf16 on disk it is not a precision trade at all.
    """
    from typeshi.portal.registry import inference_backend

    backend = inference_backend(has_cuda=False, has_bf16=False, has_mps=True)
    assert backend["dtype"] == "bfloat16"
    # device_map must stay None: bf16 COMBINED with "auto" is the documented
    # segfault, and "auto" alone SIGABRTs on hybrid-attention archs.
    assert backend["device_map"] is None


def test_inference_backend_leaves_cuda_and_cpu_alone():
    from typeshi.portal.registry import inference_backend
    from typeshi.train_motor import select_backend

    for args in ((True, True, False), (True, False, False), (False, False, False)):
        assert inference_backend(*args) == select_backend(*args)


# -- checkpoint discovery -----------------------------------------------

def test_describe_finds_adapters_and_full_weights(tmp_path):
    from typeshi.portal.registry import describe

    adapter = tmp_path / "adapter-ckpt"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-4B"})
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"x" * 10)

    full = tmp_path / "full-ckpt"
    full.mkdir()
    (full / "model.safetensors").write_bytes(b"x" * 20)

    empty = tmp_path / "nothing"
    empty.mkdir()

    a, f = describe(adapter), describe(full)
    assert a.kind == "adapter" and a.base_model == "Qwen/Qwen3.5-4B"
    assert f.kind == "full" and f.base_model is None
    assert describe(empty) is None


def test_modes_come_from_the_corpora_in_the_split(tmp_path):
    """A transcription-only checkpoint has never seen <MODE:C>, so the portal
    must be able to say so before someone reads OOD output as a model bug."""
    from typeshi.portal.registry import describe

    def make(name, writers):
        d = tmp_path / name
        d.mkdir()
        (d / "adapter_config.json").write_text("{}")
        (d / "split.json").write_text(
            json.dumps({"train_writers": writers, "test_writers": writers})
        )
        return describe(d)

    aalto_only = make("phase1", ["aalto:1", "aalto:2"])
    assert aalto_only.corpora == ("aalto",)
    assert aalto_only.modes == ("transcription",)

    mixed = make("phase2", ["aalto:1", "klicke:9"])
    assert mixed.corpora == ("aalto", "klicke")
    assert mixed.modes == ("transcription", "composition")

    # No split at all means unknown provenance, not "supports nothing".
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "adapter_config.json").write_text("{}")
    assert describe(bare).modes == ("transcription", "composition")


def test_split_corpora_is_cached_per_file_stat(tmp_path):
    """discover() runs on every /api/info, which the browser polls every two
    seconds during a load, and motor-full's split.json is 2.3 MB."""
    from typeshi.portal import registry

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"test_writers": ["aalto:1"]}))
    registry._CORPORA_CACHE.clear()
    assert registry.split_corpora(split) == ("aalto",)
    assert len(registry._CORPORA_CACHE) == 1
    registry.split_corpora(split)
    assert len(registry._CORPORA_CACHE) == 1
    assert registry.split_corpora(tmp_path / "absent.json") == ()


def test_discover_includes_adapter_dirs_and_midrun_saves(tmp_path):
    """The old playground globbed only model.safetensors, which made every
    motor-* adapter checkpoint invisible to it."""
    from typeshi.portal.registry import discover

    root = tmp_path / "checkpoints"
    (root / "motor-x").mkdir(parents=True)
    (root / "motor-x" / "adapter_config.json").write_text("{}")
    (root / "tiny" / "checkpoint-500").mkdir(parents=True)
    (root / "tiny" / "checkpoint-500" / "model.safetensors").write_bytes(b"x")

    paths = {i.path.name for i in discover(root)}
    assert paths == {"motor-x", "checkpoint-500"}
    assert discover(tmp_path / "missing") == []


# -- request validation --------------------------------------------------

def test_validate_target_rejects_unsupported_characters():
    from typeshi.portal.server import PortalError, validate_target

    with pytest.raises(PortalError) as exc:
        validate_target("smart “quotes”", 300)
    assert "“" in str(exc.value)


def test_validate_target_rejects_prompt_markers():
    from typeshi.portal.server import PortalError, validate_target

    with pytest.raises(PortalError, match="<TARGET>"):
        validate_target("hello <TARGET> world", 300)


def test_validate_target_rejects_empty_and_overlong():
    from typeshi.portal.server import PortalError, validate_target

    with pytest.raises(PortalError, match="empty"):
        validate_target("   ", 300)
    with pytest.raises(PortalError, match="longer"):
        validate_target("a" * 301, 300)
    assert validate_target("  ok then  ", 300) == "ok then"


def test_labels_from_converts_percents_to_fractions():
    from typeshi.portal.server import labels_from

    labels = labels_from({"wpm": 80, "ecor": 12, "eunc": 3, "rev": 5})
    assert labels.wpm == 80
    assert labels.corrected_error_rate == pytest.approx(0.12)
    assert labels.uncorrected_error_rate == pytest.approx(0.03)
    assert labels.revision_rate == pytest.approx(0.05)


def test_composition_gets_a_larger_token_budget():
    from typeshi.portal.server import default_budget

    text = "a" * 50
    assert default_budget(text, "transcription") == 4 * 50 + 64
    assert default_budget(text, "composition") > default_budget(text, "transcription")


# -- event rows ----------------------------------------------------------

def test_event_rows_carry_cursor_and_seldel_without_crashing():
    """The old serializer did round(float(release)) unconditionally and
    labelled everything key/bksp -- both break on composition output."""
    from typeshi.portal.rows import event_rows

    events = [
        Event.key("a", 0, 50),
        Event.cursor(0, 100),
        Event.seldel(0, 1, 200),
    ]
    rows = event_rows(events)
    assert [r["type"] for r in rows] == ["key", "cursor", "seldel"]
    assert rows[1]["release"] is None and rows[1]["pos"] == 0
    assert rows[2]["start"] == 0 and rows[2]["end"] == 1


def test_span_ms_uses_the_last_release_not_the_last_press():
    """Holds overlap -- 26% of real keystrokes roll over -- so release times
    are not sorted and the final key-up can belong to an earlier press."""
    from typeshi.portal.rows import span_ms

    events = [Event.key("a", 0, 900), Event.key("b", 100, 150)]
    assert span_ms(events) == 900.0
    assert span_ms([]) == 0.0


def test_replay_safe_returns_the_partial_text_and_names_the_bad_event():
    from typeshi.portal.rows import replay_safe

    events = [Event.key("a", 0, 10), Event.cursor(9, 20), Event.key("b", 30, 40)]
    text, error = replay_safe(events)
    assert text == "a"
    assert "event 1 (cursor)" in error and "outside buffer" in error


def test_replay_safe_is_silent_on_a_clean_stream():
    from typeshi.portal.rows import replay_safe

    text, error = replay_safe([Event.key("h", 0, 10), Event.key("i", 20, 30)])
    assert (text, error) == ("hi", None)


def test_session_stats_reports_mix_and_pauses():
    from typeshi.portal.rows import session_stats

    events = [Event.key("h", 0, 10), Event.key("i", 2000, 2010)]
    stats = session_stats(events, "hi")
    assert stats["exact"] is True
    assert stats["similarity"] == 1.0
    assert stats["event_mix"] == {"key": 1.0}
    assert stats["pause_fraction"] == 1.0  # the single gap is over a second


# -- job queue -----------------------------------------------------------

def test_job_runs_and_reports_its_result():
    from typeshi.portal.jobs import JobQueue

    q = JobQueue()
    job = q.submit("test", {}, lambda j: {"answer": 42})
    _wait(job)
    assert job.state == "done" and job.result == {"answer": 42}


def test_job_error_is_captured_not_raised():
    from typeshi.portal.jobs import JobQueue

    q = JobQueue()

    def boom(_job):
        raise ValueError("nope")

    job = q.submit("test", {}, boom)
    _wait(job)
    assert job.state == "error" and "ValueError: nope" in job.error


def test_systemexit_in_a_job_does_not_kill_the_worker():
    """Loader paths raise SystemExit; a bare `except Exception` would let it
    unwind and leave the queue with no worker and every later job hanging."""
    from typeshi.portal.jobs import JobQueue

    q = JobQueue()

    def exits(_job):
        raise SystemExit("tokenizer probe failed")

    first = q.submit("test", {}, exits)
    _wait(first)
    assert first.state == "error"

    second = q.submit("test", {}, lambda j: {"ok": True})
    _wait(second)
    assert second.state == "done"


def test_cancel_before_start_never_runs_the_job():
    from typeshi.portal.jobs import Job, JobQueue

    q = JobQueue()
    blocker = q.submit("test", {}, lambda j: time.sleep(0.3) or {"ok": True})
    ran = []
    job = q.submit("test", {}, lambda j: ran.append(1) or {"ok": True})
    job.cancel()
    _wait(blocker)
    _wait(job)
    assert job.state == "cancelled"
    assert ran == []
    assert isinstance(job, Job)


def test_cancel_midrun_keeps_the_partial_result_but_says_cancelled():
    from typeshi.portal.jobs import JobQueue

    q = JobQueue()

    def slow(job):
        for _ in range(200):
            if job.cancelled:
                break
            time.sleep(0.005)
        return {"partial": True}

    job = q.submit("test", {}, slow)
    while job.state != "running":
        time.sleep(0.005)
    job.cancel()
    _wait(job)
    assert job.state == "cancelled"
    assert job.result == {"partial": True}


def test_wait_for_change_returns_on_progress():
    from typeshi.portal.jobs import Job

    job = Job("j1", "test", {})
    job.set_progress(step=3)
    version, snap = job.wait_for_change(-1, timeout=0.1)
    assert version > 0 and snap["progress"]["step"] == 3


# -- readout -------------------------------------------------------------

def test_validity_matches_the_tier1_gate_threshold():
    from typeshi.config import REPLAY_SIM_MIN
    from typeshi.portal.readout import validity

    good = [Event.key(c, i * 10, i * 10 + 5) for i, c in enumerate("hello")]
    result = validity(good, "hello", "transcription")
    assert result["ok"] is True and result["threshold"] == REPLAY_SIM_MIN

    bad = validity(good, "completely different text", "transcription")
    assert bad["ok"] is False and "similarity" in bad["reason"]


def test_validity_flags_cursor_events_only_in_transcription():
    from typeshi.portal.readout import validity

    events = [Event.key("h", 0, 5), Event.cursor(0, 10), Event.key("i", 20, 25)]
    assert validity(events, "ih", "transcription")["off_type_events"] == ["cursor"]
    assert validity(events, "ih", "transcription")["ok"] is False
    # Composition legitimately emits cursor ops, so they are reported without
    # failing the sample.
    assert validity(events, "ih", "composition")["ok"] is True


def test_serial_readout_has_one_row_per_feature_with_its_null():
    from typeshi.portal.readout import SERIAL_FEATURES, serial_readout

    events = [Event.key("a", i * 90, i * 90 + 40) for i in range(30)]
    readout_rows = serial_readout(events)
    assert len(readout_rows) == len(SERIAL_FEATURES) == 9
    assert [r["null"] for r in readout_rows] == [f[2] for f in SERIAL_FEATURES]
    assert all("value" in r for r in readout_rows)


def test_serial_readout_marks_band_membership_when_given_one():
    from typeshi.portal.readout import serial_readout

    events = [Event.key("a", i * 90, i * 90 + 40) for i in range(30)]
    band = {"drift": [0.0, 0.5, 1.0]}
    row = next(r for r in serial_readout(events, band) if r["key"] == "drift")
    assert row["p10"] == 0.0 and row["p90"] == 1.0
    assert row["in_band"] == (0.0 <= row["value"] <= 1.0)


def test_readout_survives_json_dumps():
    """numpy scalars compare to numpy bools, which json.dumps rejects with an
    opaque "Object of type bool is not JSON serializable" -- and the readout
    only ever reaches the browser through json.dumps."""
    from typeshi.portal.readout import real_band, serial_readout

    sessions = [
        [Event.key("a", i * (80 + s * 7), i * (80 + s * 7) + 40)
         for i in range(30)]
        for s in range(4)
    ]
    band = real_band(sessions)
    json.dumps(serial_readout(sessions[0], band))  # must not raise
    json.dumps(band)


def test_real_band_is_empty_without_sessions():
    from typeshi.portal.readout import real_band

    assert real_band([]) == {}


# -- corpus --------------------------------------------------------------

def test_load_test_writers_reads_the_split(tmp_path):
    from typeshi.portal.corpus import load_test_writers

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"test_writers": ["aalto:1", "klicke:2"]}))
    assert load_test_writers(split) == {"aalto:1", "klicke:2"}
    assert load_test_writers(tmp_path / "missing.json") == set()


def test_resolve_split_prefers_the_checkpoints_own(tmp_path):
    """A session held out for a different run is training data for this one."""
    from typeshi.portal.corpus import resolve_split

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "split.json").write_text("{}")
    fallback = tmp_path / "global.json"
    fallback.write_text("{}")

    assert resolve_split(ckpt, fallback) == ckpt / "split.json"
    assert resolve_split(tmp_path / "bare", fallback) == fallback
    assert resolve_split(None, tmp_path / "absent.json") is None
