# SDD ledger — plan: docs/superpowers/plans/2026-08-10-tiny-motor-poc.md
Task 1: minor (deferred): task-1-report.md says "Deviations: None" while documenting the (necessary) regex deviation elsewhere — report-file wording only
Task 1: complete (commits c9a520f..45c0627, review clean)
Task 2: review found Important (gap branch bypasses _cached() device tensors; file evolved after plan was written) -> fix round 1 dispatched
Task 2: minor (deferred): module-level docstring (constrain.py:8-16) still says EOS legal only in EVENT state
Task 2: minor (deferred): new mask tests are CPU-only; device-cache path untested off-CPU
Task 2: fix round 1/5 (1 addressed, 0 open — cached _dt_eos in _cached(); commits 739ff91..55eb48e)
Task 2: complete (commits 3371a52..55eb48e, review clean after 1 fix round)
Task 3: minor (deferred): save_steps=... passed even under save_strategy="epoch" (dead value, inherited from brief)
Task 3: complete (commits 55eb48e..2e839fc, review clean)
Task 4: minor (deferred): build_prompt runs twice per attempt (pre-check + generate) — pure function, plan-mandated pattern
Task 4: complete (commits 2e839fc..46b6eee, review clean)
Task 5: minor (deferred): conftest would stack two skip marks if a test ever carried both network+slow (inert today)
Task 5: complete (commits 46b6eee..a51f1f8, review clean). e2e evidence: loss 4.67->1.97/100 steps, 35 events vs 236 budget, EOS terminated, ~10s runtime
Task 7: script complete + reviewed clean (commits a51f1f8..c04d9dc); measurement run pending dataset rebuild
Task 6: complete (full rebuild: 1,975,019 train / 1,950,110 transcription / 219,574 test; 14,765 held-out logs symlinked)
Task 7: MEASUREMENT: harness ceiling = 0.915 full / 0.910 timing-only vs band [0.40,0.55] -> STOP CONDITION FIRED (predeclared): pass_model unreachable for ANY generator incl. 7B until featurization fixed. Surfaced to user; Tasks 8-9 (validity/throughput-gated) continue; Task 10 (overnight) already user-gated.
Task 8: complete (smoke 8.13M: 117.7 ex/s train, dataset ops ~350k ex/s, 0 unencodable prompts in full 1.95M corpus). Ladder: full epoch overnight viable if 19M config >= ~46 ex/s; definitive number from pilot runs.
Task 7b: dispatched (user approved eval symmetrization)
Task 7b: fix round 1/5 (2 addressed, 0 open — honest int|float annotations; commits 0056d3a..b9993fe)
Task 7b: minor (deferred): serialize.py pending_dt local annotated int|None, holds float (pre-existing)
Task 7b: complete (commits c04d9dc..b9993fe, review clean after 1 fix round). from_bin now returns float bin centers (16/128 bins collided under int rounding); eval symmetrized.
Task 9: 25k pilot: 67.25 ex/s, loss 7.28->4.11, eval 0/150 valid (all wrong-text, 0 malformed). Ladder locked: full epoch ~= 8.1h, overnight OK.
Task 9: 100k pilot: 70.46 ex/s, loss 3.53, tok-acc 14.8%; eval CRASHED with 1-4 valid pairs (sklearn n_splits) -> harness gap; validity ~1-3%, below bands -> predeclared lever applied (2 epochs on 100k, tiny-pilot-100k-e2, training now)
Task 7c: dispatched (eval honest-report guard below 5 valid pairs)
Task 7c: minor (deferred): guard comment reworded slightly beyond brief (clearer, harmless)
Task 7c: complete (commits b9993fe..72cc3f9, review clean)
Task 9: lever (100k x 2ep): loss ->2.79, eval 0/150 valid. STOP CONDITION FIRED per bands. Diagnostic: median replay similarity 0.52 (11/15 in 0.33-0.64, 4 near 0); failure is systematic space-omission + start/end derail, NOT babble. Copying circuit learned; " "-><SPC:h> is the one asymmetric binding and lags. Surfaced to user with full-run recommendation.
Task 10: user approved full overnight run at pilot stop-gate. Launched: 1 epoch x 1.95M, save-steps 3000, checkpoints/motor-tiny, ~8h ETA. Final eval --n 200 on completion.
Task 9 CORRECTION: pilot "failure" was an EVAL ARTIFACT. AutoTokenizer resolves the tiny checkpoint to Qwen2Tokenizer (config.json model_type=qwen2 preempts the unresolvable TokenizersBackend class name) which EATS ALL SPACES at encode; model typed spaceless prompts faithfully. CPU probe with correct tokenizer: 100k-e2 = 8/8 valid, sim median 0.966, space ratio 1.09. The "asymmetric SPC binding" narrative was wrong. PreTrainedTokenizerFast.from_pretrained loads correctly (verified on scratch copy). Task 7d dispatched.
User dashboard: watch_training.py extended via Codex (it/s+s/it rates, logs/*.log auto-pick, checkpoint-probes panel), verified rendering, committed. Note: tiny_full.log loss lines arrive in ~8KB stdout-buffer bursts; use PYTHONUNBUFFERED=1 on future runs.
Task 7d: fix round 1/5 (2 addressed, 0 open — clean_up pin in prepare_tokenizer, probe covers adjacent tokens + PEFT branch via network test; commits 8f92a5b..61c6c76)
Task 7d: minor (deferred): preserves-spaces test duplicates probe literal instead of importing _PROBE
Task 7d: complete (commits 72cc3f9..61c6c76, review clean after 1 fix round)
Morning agenda: final eval --n 200 (fixed loader) after overnight completes; corrected pilot evals for scaling curve; results doc; final whole-branch review.
Overnight probe ckpt-3000 (~10% epoch): 8/8 valid, sim median 1.00 mean 0.948 min 0.80, space ratio 1.05. Healthy.
Health check 01:20 (step ~4400/30471, 14%): process alive, log fresh, 0 failure signatures, loss 9.29->2.69, tok-acc 16.8% (already past 100k-2ep pilot's 2.79/15.8% at <10% of epoch). Offline suite 162 passed/8 skipped. Expanded probe ckpt-3000 n=16: 16/16 valid, 0 malformed, sim median 0.966, space ratio 1.016. Qualitative: exact-copy 59-char sentence, session durations matching real humans (13.8s vs 13.8s).
Knob fidelity measured on ckpt-3000 (playground API): requested WPM 30/60/110 -> realized 30.6/65.3/114.6; ecor 2%->1 bksp, 25%->13 bksp. Strong conditioning; note for results doc (design spec eval target #3).
Playground (user request): scripts/playground.py (stdlib server, CPU by default so it never steals MPS from training) + scripts/playground.html (Codex-built UI: live replay, QWERTY keycap sim with real hold durations/rollover, WPM/error/temp/seed knobs, playback speed). Bug found+fixed during verification: OOV chars raised a pyo3 exception past the ValueError handler and killed the request thread -> now pre-checked against TEXT_CHARS with a named-offender message, plus broad except so a handler thread can never die silently. Committed.
Overnight probe ckpt-6000 (20% epoch, step 6107): 16/16 valid, 0 malformed, sim median 0.981 (up from 0.966), mean 0.953, min 0.852 (up from 0.80). Loss 2.66, tok-acc 17.2%. Healthy, improving. ETA 6h30m.
Overnight probe ckpt-9000 (30% epoch, step 9073): 16/16 valid, sim median 0.977 mean 0.971 min 0.907 (tail keeps tightening: 0.800 -> 0.852 -> 0.907), space 1.056. Loss 2.625, tok-acc 17.7%. Healthy. ETA 5h46m.
Overnight probe ckpt-12000 (40% epoch, step 12075): 16/16 valid, sim median 0.985 mean 0.973 min 0.914, space 1.060. Loss 2.610, tok-acc 18.1%. Healthy, 283Gi disk free. ETA 4h58m.
Overnight probe ckpt-15000 (49% epoch, step 15093): 16/16 valid, sim median 0.985 (flat) mean 0.976 min 0.914 (flat), space 1.099. Loss 2.583, tok-acc 18.3%. Probe metrics plateauing; loss still falling with cosine LR decay ahead. Healthy. ETA 4h30m (total ~8h20m, slightly over estimate due to CPU probe/playground contention).
Overnight probe ckpt-18000 (59% epoch, step 18084) WIDENED to n=32: 32/32 valid, 0 malformed, sim median 0.977 mean 0.974 min 0.914, space 1.065. Loss 2.581, tok-acc 18.2%. 32/32 with min 0.914 (gate cutoff 0.80) = strong predictor for the n=200 validity gate. Healthy. ETA 3h27m.
Overnight probe ckpt-21000 (69% epoch, step 21110) n=32: 32/32 valid, sim median 0.985 mean 0.975 min 0.882 (min dipped from 0.914 -- sample noise on different sessions, still well above the 0.80 gate). Loss 2.564, tok-acc 18.8%. Healthy. ETA 2h35m.
Overnight probe ckpt-24000 (79% epoch, step 24132) n=32: 32/32 valid, sim median 0.985 mean 0.976 min 0.898. Loss 2.556, tok-acc 18.9%. LR now 6.8e-05 (cosine tail). Healthy. ETA 1h47m.
Overnight probe ckpt-27000 (89% epoch, step 27105) n=32: 32/32 valid, sim median 0.980 mean 0.978 (best yet) min 0.917 (best yet at n=32). Loss 2.550, tok-acc 19.0%. LR 2.0e-05. Healthy. ETA 56m.
Task 10 FINAL EVAL (checkpoints/motor-tiny, n=200 held-out): 200/200 valid (100%), 0 malformed, 0 wrong-text.
  HARD BAR 3/3 PASS: validity 1.00 (>=0.90), teeth vs heuristic 0.995 (>=0.90), control 0.445 (in [0.40,0.60]).
  STRETCH 0/2: model-vs-real 0.640 (needs <=0.55; timing-only 0.622 so not a length artifact), serial-dependence teeth 0.500 (needs >=0.75) = discriminator itself has NO serial sensitivity at n=200.
  tier1_met False (by stretch gates only).
DIAGNOSTIC (serial-dependence gate): measured on 120 held-out sessions -- real-vs-timing-shuffled accuracy is 0.542 on RAW ms and 0.483 round-tripped, both far below the 0.75 gate. Cause: lag-1 log-IKI autocorrelation of real Aalto sessions is +0.009 (~zero), so the ONE order-sensitive feature in the 33-dim vector carries no signal; the other 32 are marginal and identical by construction under shuffling. The gate is UNPASSABLE with this featurization for ANY model -- not caused by symmetrization (raw is equally bad) and not a property of our generations. Affects Phase 1 GPU eval identically. Recommend redesigned order-sensitive features (multi-lag autocorr, run-length stats, digraph-conditioned deviations) as a Phase-1 task.
DISTRIBUTIONAL (final): iki KL 0.015 (excellent), hold KL 0.059, burst KL 0.090, pause KL 0.489 (the outlier -> long thinking pauses are where the model's timing diverges; consistent with model-vs-real 0.64).
SCALING CURVE (corrected loader, n=50 pilots / n=200 full): 25k x1ep 0% valid | 100k x1ep 77% valid, model-vs-real 0.470 (PASSES realism!) | 100k x2ep 98%, 0.610 | 1.95M x1ep 100%, 0.640. Finding: text fidelity and timing realism TRADE OFF -- more training sharpens timing toward the mean and makes it more distinguishable. Temperature sweep (T=1.1/1.2/1.35, n=100) launched per spec's named first lever.
TEMPERATURE SWEEP: T=1.0 optimal (0.640); 1.1->0.645, 1.2->0.710, 1.35->0.740 with pause KL exploding 0.489->2.41 and validity eroding to 88%. Spec's named first lever is exhausted; gap is capacity/distribution not sampling sharpness.
Task 11: complete -- docs/results-tiny-poc.md written and committed with all eval JSONs.
FINAL REVIEW: APPROVED WITH FOLLOW-UPS (no correctness bug in shipped code; reviewer independently reproduced the tokenizer trap, confirmed training data bit-identical under from_bin float change, confirmed eval symmetrization sound, re-measured +0.009/0.542/0.483 exactly). 1 Important doc error (order-sensitive feature count 1 -> actually 8) + 5 minors. Fix wave applied (7172446 docs+shuffle_diagnostic, db447fe code). Deferred minors triaged: only constrain.py module docstring required fixing; rest fine to defer.
NOTE: tests/test_interp_reconstruct.py failure is untracked work from a concurrent plan (keyboard reconstruction), not this branch.
Fix-wave re-review: all 7 findings ADDRESSED, new breakage none, suite 179 passed/8 skipped/1 failed (out-of-scope untracked interp test from concurrent plan, confirmed unaffected by this diff).
Residual non-blocking notes: (1) doc stated burst-pause count as measured when it was inferred -> CONTROLLER-APPLIED doc-only correction (commit below), reworded to match shuffle_diagnostic.json's literal field; (2) final-fixes-report wording about the fixture's manual side-check -- parked, report-file wording only, no code impact.
ALL TASKS COMPLETE.
FINISH: user chose "keep branch as-is". Nothing merged or pushed. Branch feat/data-pipeline-motor-model, 64 commits vs main (23 from this plan). Committed tree green (176 passed, 8 skipped); the 1 failing test is untracked concurrent-plan WIP (tests/test_interp_reconstruct.py).
