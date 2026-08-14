# Diagnostics — the throwaway scripts that produced the findings

Ad-hoc probes and job chains from the 2026-08-11→14 GPU session, preserved
because several of them are the *evidence* behind claims in
`docs/gpu-run-chronicle.md`, `docs/results-qwen35-4b-gpu.md` and
`docs/open-work.md`. They were written to run once on that box: paths are
absolute in places, checkpoints are assumed present, and none of them are
covered by the test suite. Read them as lab notes, not as tools.

The ones that matter:

| script | what it established |
|---|---|
| `protocol_test.py` | **The Tier-1 correction.** Recovers writer IDs for the dumped pairs and scores pair-grouped vs writer-grouped CV, plus the direct writer-identification check (0.595 against 0.071 chance) |
| `stability.py` | The same comparison across 8 seeds: 0.6322 ± 0.0120 vs 0.5181 ± 0.0085 — what made the finding safe to act on |
| `dump_and_analyze.py` | Dumped 200 real/generated pairs and ranked discriminator feature importances; produced `data/generation_dump_e3.jsonl` |
| `eos_probe.py` | Showed the trained model puts p≥0.9998 on EOS exactly where the mask forbade it (the EOS-parity bug) |
| `converge_probe.py` | Convergence rate under the decoder: 1/5 → 5/5 after the cooldown fix, then 50/50 |
| `diag_probe.py` | Composition behaviour probe — event mixes, pause fractions, the "types its own essay" finding |
| `pause_probe.py` | Real vs generated inter-key-interval histograms |
| `temp_sweep.py` | Temperature sweep (the monotone non-lever result) |
| `user_text_probe.py` | Reproduced the grammar-mask-vs-convergence difference on a user-supplied sentence |
| `teeth_check.py` | Verified the ordering test actually fails under an `imap_unordered` regression |

The `*.sh` files are the job chains that sequenced the overnight runs
(training → eval → RAFT → re-eval). `build_ccv.sh` is the causal-conv1d
source build against the CUDA 13 toolkit, which is the only one likely to
be useful again on a fresh box.
