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
