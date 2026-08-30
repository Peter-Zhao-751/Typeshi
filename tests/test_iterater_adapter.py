"""IteraTeR -> synthetic composition sessions.

The dataset gives real human edit actions with char offsets into the draft;
the adapter's job is to execute them as canonical events with plausible
timing, ending byte-exact on the revised text. Every session it yields must
satisfy replay(events) == target -- that is the same gate the real-corpus
adapters live under, and with synthesized streams an offset bug is the
expected failure mode, so the gate is the test.
"""

import json

import pytest

from typeshi.buffer import replay
from typeshi.events import EventType


class FakeSampler:
    """Deterministic timing so tests assert structure, not distributions."""

    def key_timing(self, pause_class):
        return {"within_word": 100.0, "word_boundary": 250.0,
                "clause_boundary": 600.0}[pause_class], 50.0

    def think_pause(self):
        return 1500.0

    def op_gap(self):
        return 300.0


def _doc(doc_id, before, after, actions, depth=1):
    return {
        "doc_id": doc_id, "revision_depth": depth, "domain": "arxiv",
        "before_revision": before, "after_revision": after,
        "sents_char_pos": [], "edit_actions": actions,
    }


def _action(kind, before, after, start, end, intent="clarity"):
    return {"type": kind, "before": before, "after": after,
            "start_char_pos": start, "end_char_pos": end,
            "major_intent": intent, "raw_intents": [intent]}


def _write_corpus(tmp_path, docs, sents=()):
    (tmp_path / "human_doc").mkdir(parents=True)
    (tmp_path / "human_sent").mkdir(parents=True)
    with open(tmp_path / "human_doc" / "train.json", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    with open(tmp_path / "human_sent" / "train.json", "w") as f:
        for s in sents:
            f.write(json.dumps(s) + "\n")
    return tmp_path


def test_single_replace_replays_to_the_revised_text(tmp_path):
    from typeshi.adapters import iterater

    before = "the model is good at typing."
    after = "the model is capable at typing."
    docs = [_doc("d1", before, after,
                 [_action("R", "good", "capable", 13, 17)])]
    root = _write_corpus(tmp_path, docs)

    sessions = list(iterater.iter_sessions(root, FakeSampler()))
    assert len(sessions) == 1
    writer, target, events = sessions[0]
    assert writer == "d1"
    assert target == after
    assert replay(events) == after
    kinds = {e.type for e in events}
    assert EventType.CURSOR in kinds and EventType.SELDEL in kinds


def test_multiple_actions_apply_in_reading_order_with_offset_shift(tmp_path):
    """Offsets index the ORIGINAL draft; applying edits top-down shifts every
    later offset by the length delta of the earlier ones. Getting this wrong
    is silent text corruption, which is why replay-equality is the gate."""
    from typeshi.adapters import iterater

    before = "aa bb cc dd"
    after = "aaXX bb ccYY dd"  # two length-changing inserts
    docs = [_doc("d1", before, after, [
        _action("A", "", "YY", 8, 8),   # deliberately listed out of order
        _action("A", "", "XX", 2, 2),
    ])]
    root = _write_corpus(tmp_path, docs)

    sessions = list(iterater.iter_sessions(root, FakeSampler()))
    assert len(sessions) == 1
    _, target, events = sessions[0]
    assert replay(events) == after == target


def test_a_revision_chain_becomes_one_session_targeting_the_final_draft(tmp_path):
    from typeshi.adapters import iterater

    d0 = "one two three"
    d1 = "one 2 three"
    d2 = "one 2 four"
    docs = [
        _doc("d1", d0, d1, [_action("R", "two", "2", 4, 7)], depth=1),
        _doc("d1", d1, d2, [_action("R", "three", "four", 6, 11)], depth=2),
    ]
    root = _write_corpus(tmp_path, docs)

    sessions = list(iterater.iter_sessions(root, FakeSampler()))
    assert len(sessions) == 1
    _, target, events = sessions[0]
    assert target == d2
    assert replay(events) == d2
    assert sum(1 for e in events if e.type is EventType.SELDEL) == 2


def test_word_processor_homoglyphs_are_normalized_with_offsets_remapped(tmp_path):
    """IteraTeR is arxiv/news/wiki prose: curly quotes are everywhere. They
    normalize to ASCII identities, and the edit offsets -- which index the
    ORIGINAL text -- must be remapped through the normalization."""
    from typeshi.adapters import iterater

    before = "“quoted” words here"
    after = "“quoted” terms here"
    docs = [_doc("d1", before, after,
                 [_action("R", "words", "terms", 9, 14)])]
    root = _write_corpus(tmp_path, docs)

    sessions = list(iterater.iter_sessions(root, FakeSampler()))
    assert len(sessions) == 1
    _, target, events = sessions[0]
    assert target == '"quoted" terms here'
    assert replay(events) == target


def test_residual_whitespace_is_reconciled_with_a_cleanup_edit(tmp_path):
    """IteraTeR's edit actions do not always span the whitespace around a
    deleted word, so executing them faithfully leaves 'aa  cc' where
    after_revision says 'aa cc' -- 303 of 559 docs, median difference one
    character. The gate must stay byte-exact (replay == target is what the
    whole pipeline converges on), so the residue is repaired with a small
    synthesized cleanup edit, the way a human tidies a doubled space."""
    from typeshi.adapters import iterater

    before = "aa bb cc"
    docs = [_doc("d1", before, "aa cc", [_action("D", "bb", "", 3, 5)])]
    root = _write_corpus(tmp_path, docs)

    sessions = list(iterater.iter_sessions(root, FakeSampler()))
    assert len(sessions) == 1
    _, target, events = sessions[0]
    assert target == "aa cc"
    assert replay(events) == "aa cc"


def test_a_large_residue_still_drops_the_doc(tmp_path):
    """Reconciliation exists for whitespace slack, not for rows whose
    actions and after-text genuinely disagree -- inventing a 50-character
    edit no human made would be fabricated behaviour."""
    from typeshi.adapters import iterater

    docs = [_doc("d1", "aa bb cc", "something else entirely different here",
                 [_action("D", "bb", "", 3, 5)])]
    root = _write_corpus(tmp_path, docs)

    assert list(iterater.iter_sessions(root, FakeSampler())) == []


def test_a_doc_whose_offsets_do_not_match_is_skipped_not_corrupted(tmp_path):
    from typeshi.adapters import iterater

    docs = [
        _doc("bad", "aaaa", "bbbb", [_action("R", "zzzz", "bbbb", 0, 4)]),
        _doc("good", "aaaa", "bbbb", [_action("R", "aaaa", "bbbb", 0, 4)]),
    ]
    root = _write_corpus(tmp_path, docs)

    sessions = list(iterater.iter_sessions(root, FakeSampler()))
    assert [w for w, _, _ in sessions] == ["good"]


def test_times_are_monotone_and_revisions_carry_a_think_pause(tmp_path):
    from typeshi.adapters import iterater

    before = "ab cd"
    after = "ab xy"
    docs = [_doc("d1", before, after, [_action("R", "cd", "xy", 3, 5)])]
    root = _write_corpus(tmp_path, docs)

    _, _, events = next(iter(iterater.iter_sessions(root, FakeSampler())))
    presses = [e.press_time for e in events]
    assert presses == sorted(presses)
    assert all(int(p) == p for p in presses)
    first_op = next(i for i, e in enumerate(events)
                    if e.type is EventType.CURSOR)
    gap = events[first_op].press_time - events[first_op - 1].press_time
    assert gap >= 1500, "the pause before a revision is think time"


def test_sentence_rows_become_dense_mini_sessions(tmp_path):
    from typeshi.adapters import iterater

    sents = [
        {"before_sent": "it was good.", "after_sent": "it was great.",
         "labels": "clarity", "doc_id": "d9", "revision_depth": 1},
        {"before_sent": "unchanged.", "after_sent": "unchanged.",
         "labels": "fluency", "doc_id": "d9", "revision_depth": 1},
    ]
    root = _write_corpus(tmp_path, [], sents)

    sessions = list(iterater.iter_sentence_sessions(root, FakeSampler()))
    assert len(sessions) == 1, "a no-op row synthesizes nothing"
    writer, target, events = sessions[0]
    assert writer == "d9"
    assert target == "it was great."
    assert replay(events) == target
    assert any(e.type is EventType.SELDEL for e in events)


def test_timing_sampler_draws_from_class_conditioned_pools():
    from typeshi.adapters.timing import TimingSampler

    sampler = TimingSampler(
        key_pairs={"within_word": [(90.0, 45.0)],
                   "word_boundary": [(240.0, 50.0)],
                   "clause_boundary": [(900.0, 55.0)]},
        think_pauses=[2000.0],
        op_gaps=[350.0],
        seed=1,
    )
    assert sampler.key_timing("word_boundary") == (240.0, 50.0)
    assert sampler.key_timing("clause_boundary") == (900.0, 55.0)
    assert sampler.think_pause() == 2000.0
    assert sampler.op_gap() == 350.0


def test_timing_sampler_harvests_pools_from_real_sessions():
    """from_sessions reads (gap, hold) pairs off KEY streams with the same
    boundary classification the eval uses, and think pauses off the gaps
    preceding cursor/seldel traffic."""
    from typeshi.adapters.timing import TimingSampler
    from typeshi.events import Event

    t, events = 0, []
    for ch in "ab cd. ef":  # 'c' is a word boundary, 'e' a clause boundary
        events.append(Event.key(ch, t, t + 60))
        t += 200
    events.append(Event.cursor(2, t + 3000))
    events.append(Event.seldel(0, 2, t + 3400))

    sampler = TimingSampler.from_sessions([events], seed=0)
    assert sampler.key_pairs["within_word"]
    assert sampler.key_pairs["word_boundary"]
    assert sampler.key_pairs["clause_boundary"]
    assert sampler.think_pauses and sampler.think_pauses[0] >= 3000
    assert sampler.op_gaps == [400.0]
