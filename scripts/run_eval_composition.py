"""Scores a checkpoint on composition: the Tier-2 harness.

Mirrors run_eval.py's discipline -- held-out writers only, paired
comparisons, codec-roundtripped both sides -- over KLiCKe essays generated
under the convergence guarantee (typeshi.converge). Adds the Tier-2
measurements: composition signatures (spec 8.4) and knob fidelity (8.3).

Gate thresholds for the discriminator/control/teeth carry over from Tier-1;
signature and knob numbers are REPORTED without pass/fail until enough runs
exist to set honest thresholds -- a gate invented before its distribution is
known is a gate tuned to pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from typeshi.adapters import klicke
from typeshi.buffer import replay
from typeshi.eval.discriminator import (
    real_vs_real_control,
    shuffle_timing,
    train_discriminator,
)
from typeshi.eval.distributional import compare_sessions
from typeshi.eval.knobs import knob_fidelity, realized_labels
from typeshi.dataset import window_label_schedule
from typeshi.eval.signatures import compare_signatures
from typeshi.generate import generate_windowed
from typeshi.labels import compute_labels
from typeshi.serialize import codec_roundtrip, unsupported_chars

PASS_MODEL_BAND = (0.40, 0.55)
PASS_TEETH_MIN = 0.75
CONTROL_BAND = (0.40, 0.60)
PASS_CONVERGENCE_MIN = 0.95
TARGET_CAP = 400  # chars; probe-validated budget scale, keeps attempts bounded


def jsonable(value):
    import math

    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/motor-phase2"))
    ap.add_argument("--klicke", type=Path, default=Path("data/raw/klicke"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("eval_report_composition.json"))
    args = ap.parse_args()

    from typeshi.eval.load import load_checkpoint_model, load_checkpoint_tokenizer
    from typeshi.train_motor import _detect_backend

    tok = load_checkpoint_tokenizer(args.checkpoint)
    model = load_checkpoint_model(args.checkpoint, _detect_backend())
    model.eval()

    split = json.loads((args.checkpoint / "split.json").read_text())
    test_writers = set(split["test_writers"])
    print(f"scoring against {len(test_writers)} held-out writers", flush=True)

    real, fake, requested = [], [], []
    scored_writers: list[str] = []
    failures: list[dict] = []
    attempts = converged = 0
    max_attempts = 2 * args.n
    for path in sorted(args.klicke.rglob("*.csv")):
        if len(real) >= args.n or attempts >= max_attempts:
            break
        if klicke.gold_text_path(path) is None:
            continue
        for writer, final_text, events in klicke.iter_sessions(path):
            if f"klicke:{writer}" not in test_writers:
                continue
            target = final_text[:TARGET_CAP]
            if unsupported_chars(events):
                continue
            labels = compute_labels(events, final_text)
            attempts += 1
            try:
                # Windowed: matches the 512-event training windows -- the
                # single-shot path measured 76% convergence on long essays.
                # Conditioned on the REAL session's per-window schedule, not
                # its session average: training labels describe their own
                # window (window_labels), so a constant average is a prompt
                # the model was taught means something else. `labels` stays
                # the requested/reported aggregate for knob fidelity.
                gen = generate_windowed(
                    model, tok, target,
                    window_label_schedule(events, labels),
                    temperature=args.temperature, seed=attempts,
                )
            except ValueError as exc:
                # Failures must say WHY: the first Tier-2 run reported a
                # convergence rate with no way to tell budget exhaustion
                # from a stall, and length turned out not to be the cause
                # (every capped target is the same size).
                failures.append({
                    "writer": writer,
                    "wpm": labels.wpm,
                    "corrected_error_rate": labels.corrected_error_rate,
                    "revision_rate": labels.revision_rate,
                    "reason": str(exc)[:300],
                })
                print(f"  FAIL wpm={labels.wpm:.0f} "
                      f"rev={labels.revision_rate:.3f}: {exc}", flush=True)
                break  # counted failure
            if replay(gen) != target:
                break  # defensive; generate_windowed raises instead
            converged += 1
            scored_writers.append(writer)
            real.append(events)
            fake.append(gen)
            requested.append(labels)
            if attempts % 10 == 0:
                print(f"  attempt {attempts}: {converged} converged", flush=True)
            break  # one session per file, mirroring the probes

    convergence_rate = converged / attempts if attempts else 0.0
    report: dict = {
        "sessions_scored": len(real),
        "generation_attempts": attempts,
        "convergence_rate": convergence_rate,
        "temperature": args.temperature,
        "constrained_convergent_decoding": True,
        "codec_roundtripped_real": True,
        "held_out_writers_only": True,
        "failures": failures,
    }

    if len(real) >= 10:
        rt_real = [codec_roundtrip(s) for s in real]
        rt_fake = [codec_roundtrip(s) for s in fake]
        # Writer-grouped, for the reason documented in run_eval.py: KLiCKe
        # writers contribute several sessions, so pair-grouped folds let the
        # classifier score typist recognition as a proxy for realism.
        _, acc_model = train_discriminator(
            rt_real, rt_fake, paired=True, writers=scored_writers
        )
        _, acc_model_pair = train_discriminator(rt_real, rt_fake, paired=True)
        shuffled = [shuffle_timing(s, seed=i) for i, s in enumerate(rt_real)]
        _, teeth = train_discriminator(
            rt_real, shuffled, paired=True, writers=scored_writers
        )
        control = real_vs_real_control(rt_real)
        realized = [realized_labels(g, t) for g, t in
                    zip(rt_fake, (replay(f) for f in fake))]
        gates = {
            "pass_convergence": convergence_rate >= PASS_CONVERGENCE_MIN,
            "pass_model": PASS_MODEL_BAND[0] <= acc_model <= PASS_MODEL_BAND[1],
            "pass_serial_dependence_teeth": teeth >= PASS_TEETH_MIN,
            "pass_control_near_chance": CONTROL_BAND[0] <= control <= CONTROL_BAND[1],
        }
        report.update({
            "writer_grouped_cv": True,
            "distinct_writers_scored": len(set(scored_writers)),
            "discriminator_accuracy_vs_model": acc_model,
            "discriminator_accuracy_vs_model_pair_grouped": acc_model_pair,
            "discriminator_accuracy_vs_shuffled_real": teeth,
            "discriminator_accuracy_real_vs_real_control": control,
            "distributional": compare_sessions(rt_real, rt_fake),
            "signatures": compare_signatures(rt_real, rt_fake),
            "knob_fidelity": knob_fidelity(requested, realized),
            **gates,
            "tier2_provisional_met": all(gates.values()),
        })
    else:
        report["note"] = "fewer than 10 converged sessions; only counts reported"

    payload = json.dumps(jsonable(report), indent=2)
    args.out.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
