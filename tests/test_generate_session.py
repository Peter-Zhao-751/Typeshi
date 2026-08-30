"""generate_session's termination flag, observer and cancellation.

Offline by construction: a fake tokenizer over the registered grammar
vocabulary and a fake model that replays canned token ids through the real
processor chain. The chain is exercised for real -- that is the point, since
the observer's whole job is to agree with what the constraint processors
committed.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from typeshi.labels import SessionLabels
from typeshi.serialize import special_tokens

LABELS = SessionLabels(wpm=60, corrected_error_rate=0.0,
                       uncorrected_error_rate=0.0, revision_rate=0.0)


class TokOut(dict):
    def to(self, _device):
        return self


class FakeTok:
    """An id per registered grammar token, plus EOS/pad."""

    def __init__(self):
        self.vocab = {t: i for i, t in enumerate(
            t for t in special_tokens() if t.endswith(">")
        )}
        self.eos_token_id = len(self.vocab)
        self.pad_token_id = len(self.vocab) + 1
        self.unk_token_id = len(self.vocab) + 2
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.prompt_len = 4

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, self.unk_token_id)

    def __call__(self, _prompt, return_tensors=None):
        import torch

        return TokOut(
            input_ids=torch.zeros((1, self.prompt_len), dtype=torch.long),
            attention_mask=torch.ones((1, self.prompt_len), dtype=torch.long),
        )

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.id_to_token.get(i, "") for i in ids)

    def ids(self, *tokens):
        return [self.vocab[t] for t in tokens]

    @property
    def vocab_size(self):
        return self.unk_token_id + 1


class FakeModel:
    """Replays `canned` ids, running the processor chain at every step."""

    def __init__(self, tok, canned):
        self.tok = tok
        self.canned = canned
        self.device = "cpu"
        self.generation_config = None
        self.seen_steps = 0

    def generate(self, input_ids=None, attention_mask=None, do_sample=False,
                 max_new_tokens=16, pad_token_id=None, eos_token_id=None,
                 logits_processor=None, stopping_criteria=None, **_kw):
        import torch

        self.eos_token_id = eos_token_id
        stops = set(eos_token_id or [])
        ids = input_ids
        for token_id in self.canned[:max_new_tokens]:
            scores = torch.zeros((1, self.tok.vocab_size))
            for proc in logits_processor or []:
                scores = proc(ids, scores)
            ids = torch.cat([ids, torch.tensor([[token_id]])], dim=1)
            self.seen_steps += 1
            if token_id in stops:
                break
            if stopping_criteria is not None:
                if bool(stopping_criteria[0](ids, scores).all()):
                    break
        return ids


def _stream(tok, *tokens):
    return tok.ids(*tokens)


def test_terminated_true_when_stream_ends_on_eos():
    from typeshi.generate import generate_session

    tok = FakeTok()
    canned = _stream(tok, "<h:50>", "<DT:50>", "<i:50>") + [tok.eos_token_id]
    model = FakeModel(tok, canned)

    result = generate_session(model, tok, "hi", LABELS, constrained=False,
                              max_new_tokens=16)

    assert result.terminated is True
    assert [e.char for e in result.events] == ["h", "i"]
    assert result.steps == 4


def test_terminated_false_when_budget_runs_out():
    """A budget cutoff deserializes to a shorter but perfectly legal stream.

    Without the flag this is indistinguishable from a converged run, which is
    exactly the distinction the convergence guarantee rests on.
    """
    from typeshi.generate import generate_session

    tok = FakeTok()
    canned = _stream(tok, "<h:50>", "<DT:50>", "<i:50>", "<DT:50>")
    model = FakeModel(tok, canned)

    result = generate_session(model, tok, "hip", LABELS, constrained=False,
                              max_new_tokens=4)

    assert result.terminated is False
    # The dangling trailing <DT:> is trimmed rather than rejected.
    assert [e.char for e in result.events] == ["h", "i"]


def test_observer_sees_the_buffer_grow_and_shrink():
    from typeshi.generate import generate_session

    tok = FakeTok()
    canned = _stream(tok, "<h:50>", "<DT:50>", "<i:50>", "<DT:50>",
                     "<BKSP:50>") + [tok.eos_token_id]
    model = FakeModel(tok, canned)
    seen = []

    generate_session(model, tok, "hi", LABELS, constrained=False,
                     max_new_tokens=16,
                     observer=lambda step, text: seen.append((step, text)))

    texts = [text for _step, text in seen]
    # The observer reports what was committed BEFORE the position being
    # sampled, so each token shows up on the FOLLOWING call: the backspace is
    # visible in the last entry, where the buffer is back down to "h".
    assert texts == ["", "h", "h", "hi", "hi", "h"]
    assert [step for step, _ in seen] == [0, 1, 2, 3, 4, 5]


def test_observer_failure_never_breaks_decoding():
    from typeshi.generate import generate_session

    tok = FakeTok()
    canned = _stream(tok, "<h:50>", "<DT:50>", "<i:50>") + [tok.eos_token_id]
    model = FakeModel(tok, canned)

    def broken(_step, _text):
        raise RuntimeError("the browser went away")

    result = generate_session(model, tok, "hi", LABELS, constrained=False,
                              max_new_tokens=16, observer=broken)
    assert [e.char for e in result.events] == ["h", "i"]


def test_stop_event_halts_generation():
    import threading

    from typeshi.generate import generate_session

    tok = FakeTok()
    canned = _stream(tok, "<h:50>", "<DT:50>", "<i:50>", "<DT:50>", "<p:50>")
    model = FakeModel(tok, canned)
    stop = threading.Event()
    stop.set()

    result = generate_session(model, tok, "hip", LABELS, constrained=False,
                             max_new_tokens=16, stop_event=stop)

    assert model.seen_steps == 1  # stopped after the first committed token
    assert result.terminated is False


def test_generate_wrapper_still_returns_a_bare_event_list():
    """run_eval and probe_phase2 call generate(); its contract must not move."""
    from typeshi.generate import generate

    tok = FakeTok()
    canned = _stream(tok, "<h:50>", "<DT:50>", "<i:50>") + [tok.eos_token_id]
    model = FakeModel(tok, canned)

    events = generate(model, tok, "hi", LABELS, constrained=False,
                     max_new_tokens=16)

    assert isinstance(events, list)
    assert [e.char for e in events] == ["h", "i"]


def test_generate_is_told_to_stop_at_the_same_ids_truncation_uses():
    """The stop condition and the truncation must read one set.

    The fine-tune samples tok.eos_token_id (what its grammar mask unmasks)
    while the base config declares a DIFFERENT terminator, so a generate()
    left to its own config never stops: measured on motor-phase2, a 43-char
    transcription earned EOS after ~88 tokens then burned its remaining 148
    on garbage that truncation discarded -- identical output, 2.7x the time.
    """
    from typeshi.generate import generate_session, terminator_ids

    tok = FakeTok()
    canned = _stream(tok, "<h:50>", "<DT:50>") + [tok.eos_token_id] + \
        _stream(tok, "<z:50>", "<DT:50>", "<z:50>")
    model = FakeModel(tok, canned)

    result = generate_session(model, tok, "hi", LABELS, constrained=False,
                              max_new_tokens=16)

    assert set(model.eos_token_id) == terminator_ids(tok, model)
    assert tok.eos_token_id in model.eos_token_id
    assert tok.pad_token_id in model.eos_token_id
    # Stopped at the EOS instead of running on into the trailing garbage.
    assert model.seen_steps == 3
    assert result.terminated is True
    assert [e.char for e in result.events] == ["h"]


def test_windowed_generation_stops_at_eos_like_the_single_shot_path():
    """generate_windowed budgets 2*window_events+64 tokens per window.

    Without the terminator set reaching model.generate, every window would run
    all 1088 of them -- on Apple Silicon that is over a minute of decoding per
    window thrown away.
    """
    from typeshi.generate import generate_windowed, terminator_ids

    tok = FakeTok()
    canned = _stream(tok, "<h:50>", "<DT:50>", "<i:50>") + [tok.eos_token_id] + \
        _stream(tok, "<z:50>", "<DT:50>", "<z:50>")
    model = FakeModel(tok, canned)

    events = generate_windowed(model, tok, "hi", LABELS, window_events=8)

    assert set(model.eos_token_id) == terminator_ids(tok, model)
    assert model.seen_steps == 4  # stopped at EOS, not at the window budget
    assert [e.char for e in events] == ["h", "i"]


def test_windowed_failure_carries_the_partial_stream_and_names_it():
    from typeshi.buffer import replay
    from typeshi.generate import ConvergenceError, generate_windowed

    tok = FakeTok()
    # Types "hi" every window and stops -- never reaches the "p".
    canned = _stream(tok, "<h:50>", "<DT:50>", "<i:50>") + [tok.eos_token_id]
    model = FakeModel(tok, canned)

    with pytest.raises(ConvergenceError) as exc:
        generate_windowed(model, tok, "hip", LABELS, window_events=8)

    err = exc.value
    assert isinstance(err, ValueError)  # existing callers still catch it
    assert replay(err.events).startswith("hi")
    assert err.progress and err.progress[-1] == 2  # two on-path chars
    assert err.stalled is True
    assert "STALLED" in str(err)


def test_windowed_run_honours_cancellation():
    import threading

    from typeshi.generate import ConvergenceError, generate_windowed

    tok = FakeTok()
    canned = _stream(tok, "<h:50>", "<DT:50>", "<i:50>") + [tok.eos_token_id]
    model = FakeModel(tok, canned)
    stop = threading.Event()
    stop.set()

    with pytest.raises(ConvergenceError) as exc:
        generate_windowed(model, tok, "hip", LABELS, stop_event=stop)
    assert exc.value.cancelled is True
    assert model.seen_steps == 0


def test_windowed_generation_conditions_each_window_on_its_own_labels():
    """Training labels are per-window (window_labels); handing generation one
    session-level SessionLabels for every window prompts the model with
    conditioning it was taught means something else -- it is never told
    "this window is a revision pass". A label SEQUENCE must be accepted and
    consumed one entry per window."""
    from typeshi.generate import ConvergenceError, generate_windowed
    from typeshi.serialize import rev_bin

    class RecordingTok(FakeTok):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def __call__(self, prompt, return_tensors=None):
            self.prompts.append(prompt)
            return super().__call__(prompt, return_tensors)

    tok = RecordingTok()
    canned = _stream(tok, "<h:50>", "<DT:50>", "<i:50>") + [tok.eos_token_id]
    model = FakeModel(tok, canned)
    typing_window = SessionLabels(60, 0.0, 0.0, 0.0)
    revision_window = SessionLabels(12, 0.0, 0.0, 0.012)

    with pytest.raises(ConvergenceError):
        generate_windowed(model, tok, "hip",
                          [typing_window, revision_window], window_events=8)

    assert len(tok.prompts) >= 2
    assert "<REV:0>" in tok.prompts[0]
    assert "<WPM:12>" in tok.prompts[0]
    for later in tok.prompts[1:]:  # window 1 onward: the revision entry
        assert f"<REV:{rev_bin(0.012)}>" in later
        assert "<WPM:2>" in later


def test_windowed_generation_reuses_the_last_label_when_windows_outrun_them():
    """The schedule comes from a real session whose window count need not
    match generation's; running past the end must hold the last entry rather
    than crash or wrap around to drafting labels."""
    from typeshi.generate import ConvergenceError, generate_windowed

    class RecordingTok(FakeTok):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def __call__(self, prompt, return_tensors=None):
            self.prompts.append(prompt)
            return super().__call__(prompt, return_tensors)

    tok = RecordingTok()
    canned = _stream(tok, "<h:50>", "<DT:50>", "<i:50>") + [tok.eos_token_id]
    model = FakeModel(tok, canned)

    with pytest.raises(ConvergenceError):
        generate_windowed(model, tok, "hip",
                          [SessionLabels(60, 0.0, 0.0, 0.012)], window_events=8)

    import re

    assert len(tok.prompts) >= 2
    revs = {re.search(r"<REV:\d+>", p).group(0) for p in tok.prompts}
    assert len(revs) == 1, \
        "the lone schedule entry must repeat for every later window"


class OpFakeTok(FakeTok):
    """FakeTok plus the per-char pieces cursor ops decode through."""

    def __init__(self):
        super().__init__()
        extra = sorted(set("<CUR:SELDEL->0123456789"))
        base = self.unk_token_id + 1
        for k, ch in enumerate(extra):
            self.vocab[ch] = base + k
        self._size = base + len(extra)
        self.id_to_token = {i: t for t, i in self.vocab.items()}

    def encode(self, s, add_special_tokens=False):
        return [self.vocab[c] for c in s]

    @property
    def vocab_size(self):
        return self._size


def test_windowed_affordability_budget_is_run_scoped(monkeypatch):
    """The affordability check must see the RUN's remaining tokens, not one
    window's slice. Window-scoped budgets are the measured starvation: a
    348-char target spends ~700 of its 1088-token window just typing, so a
    long revision is never affordable anywhere in a long target -- even when
    the run as a whole has windows of slack. State is replayed across
    boundaries, so an excursion cut by a window edge resumes rather than
    strands, and every spent token is counted against the same pool."""
    import typeshi.converge as converge_mod
    from typeshi.generate import ConvergenceError, generate_windowed

    real = converge_mod.ConvergenceProcessor
    budgets = []

    class Recording(real):
        def __init__(self, *args, **kwargs):
            budgets.append(kwargs.get("token_budget"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(converge_mod, "ConvergenceProcessor", Recording)
    tok = FakeTok()
    canned = _stream(tok, "<h:50>", "<DT:50>", "<i:50>") + [tok.eos_token_id]
    model = FakeModel(tok, canned)

    with pytest.raises(ConvergenceError):
        generate_windowed(model, tok, "hip", LABELS, window_events=8)

    allowance = 2 * 8 + 64
    run_budget = (2 + (4 * 3) // 8) * allowance  # max_windows * allowance
    assert budgets[0] == run_budget
    assert budgets[1] == run_budget - 4  # window 0 spent 4 tokens (EOS too)
    assert budgets[2] == run_budget - 8


def test_a_revising_window_is_not_branded_stalled():
    """Stall was measured as target-prefix growth alone, so a window spent
    revising -- cursor ops and deletions, no new prefix chars -- read as
    'advanced nothing'. Revision ops are progress of the other kind."""
    from typeshi.generate import ConvergenceError, generate_windowed

    tok = OpFakeTok()
    canned = (_stream(tok, "<h:50>", "<DT:50>", "<i:50>", "<DT:50>")
              + [tok.vocab[c] for c in "<CUR:1>"]
              + _stream(tok, "<DT:50>") + [tok.eos_token_id])
    model = FakeModel(tok, canned)

    with pytest.raises(ConvergenceError) as exc:
        generate_windowed(model, tok, "hip", LABELS, window_events=8)

    assert exc.value.stalled is False
    assert "STALLED" not in str(exc.value)


def test_windowed_observer_reports_cumulative_steps():
    """Otherwise the progress bar would reset to zero at every boundary."""
    from typeshi.generate import ConvergenceError, generate_windowed

    tok = FakeTok()
    canned = _stream(tok, "<h:50>", "<DT:50>", "<i:50>") + [tok.eos_token_id]
    model = FakeModel(tok, canned)
    steps = []

    try:
        generate_windowed(model, tok, "hip", LABELS, window_events=8,
                          observer=lambda step, text: steps.append(step))
    except ConvergenceError:
        pass

    assert steps == sorted(steps), "step counter went backwards across windows"
    assert max(steps) > 3, "second window restarted the counter"


def test_convergence_knobs_reach_the_processor():
    from typeshi.generate import generate_session

    tok = FakeTok()
    canned = _stream(tok, "<h:50>", "<DT:50>", "<i:50>") + [tok.eos_token_id]
    model = FakeModel(tok, canned)

    result = generate_session(model, tok, "hi", LABELS, mode="composition",
                              constrained=True, max_new_tokens=16,
                              excursion_budget=0, resolve_progress=0)
    # excursion_budget=0 forbids off-path characters outright, so the only
    # thing the mask can produce is the target itself.
    assert [e.char for e in result.events] == ["h", "i"]


def _spy_processor(monkeypatch):
    """Captures the kwargs the decoder hands the convergence processor."""
    from typeshi import converge

    seen = {}
    real = converge.ConvergenceProcessor

    class Spy(real):
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(converge, "ConvergenceProcessor", Spy)
    return seen


def test_single_shot_passes_its_token_budget_to_the_processor(monkeypatch):
    """Affordability is inert unless the decoder tells the mask what the
    budget actually is -- the processor cannot see max_new_tokens itself."""
    from typeshi.generate import generate_session

    seen = _spy_processor(monkeypatch)
    tok = FakeTok()
    canned = _stream(tok, "<h:50>", "<DT:50>", "<i:50>") + [tok.eos_token_id]
    model = FakeModel(tok, canned)

    generate_session(model, tok, "hi", LABELS, mode="composition",
                     constrained=True, max_new_tokens=64)

    assert seen.get("token_budget") == 64


def test_windowed_passes_the_run_budget_to_the_processor(monkeypatch):
    """The budget the mask reasons about is the RUN's, not one window's
    slice: window-scoped budgets starved long-target revision (open-work.md),
    and state replay across boundaries is what makes run scope sound."""
    from typeshi.generate import generate_windowed

    seen = _spy_processor(monkeypatch)
    tok = FakeTok()
    canned = _stream(tok, "<h:50>", "<DT:50>", "<i:50>") + [tok.eos_token_id]
    model = FakeModel(tok, canned)

    generate_windowed(model, tok, "hi", LABELS, window_events=8)

    max_windows = 2 + (4 * 2) // 8
    assert seen.get("token_budget") == max_windows * (2 * 8 + 64)


def test_windowed_threads_the_staleness_window_through(monkeypatch):
    """Staleness -- not the excursion budget -- is what closes a long
    excursion, so it is the knob a caller has to be able to reach."""
    from typeshi.generate import generate_windowed

    seen = _spy_processor(monkeypatch)
    tok = FakeTok()
    canned = _stream(tok, "<h:50>", "<DT:50>", "<i:50>") + [tok.eos_token_id]
    model = FakeModel(tok, canned)

    generate_windowed(model, tok, "hi", LABELS, staleness_window=400)

    assert seen.get("staleness_window") == 400


def test_windowed_can_start_from_an_existing_draft():
    """Draft -> final revision, driven by the mask rather than hoped for.

    Seeding the buffer with a draft makes the convergence decoder produce the
    edit sequence that turns it into the target -- which is what "write a
    draft then revise it" actually needs, since the mask can permit an
    excursion but cannot make it a plausible earlier draft.
    """
    from typeshi.buffer import replay
    from typeshi.generate import generate_windowed

    tok = FakeTok()
    # Continuation windows open in GAP slot, so the stream starts with a gap.
    canned = _stream(tok, "<DT:50>", "<BKSP:50>", "<DT:50>", "<i:50>") + \
        [tok.eos_token_id]
    model = FakeModel(tok, canned)

    events = generate_windowed(model, tok, "hi", LABELS, draft="ho")

    # The replayed session must be read as edits ON the draft, not from empty.
    buf_text = "ho"
    from typeshi.buffer import TextBuffer
    buf = TextBuffer(buf_text)
    for e in events:
        buf.apply(e)
    assert buf.text == "hi"
    assert [e.type.value for e in events] == ["bksp", "key"]
