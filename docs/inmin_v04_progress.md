# INMIN v0.4 progress — 2026-11-03

> Subsequent authorized work: full v3 design domain + multi-step RL is now in `INMIN_FULL_SYSTEM_GUIDE.md` / `inmin_v05_progress.md`. The v0.4 handoff below is a completed historical cohort. The new one-off overnight study is active; recurring automation remains paused.

## FINAL HANDOFF — supersedes the chronological notes below

Implementation and the authorized representative experiments are complete. No experiment is left running; recurring automation remains paused. Do not restart historical formal workflows. The source/evidence ZIP inventory and CRC/SHA256 results are recorded in `output/uploads/INMIN_v04_20261103_package.json`.

- Current delivery source identity: `047989c517830c753019ba3cee5b5014b46c3190b183b3a97aa4bc0fede48e2a`. Final146 regressions pass with stable source. Updated-parent journal smoke completes/audits4 actual Toy PPA and returns0. Four-method stop/resume validation passes: same sequence, best mapping and update count; no durable re-evaluation.
- Main128-query supplement is completed/source-stable/audit-passed, return0:128 full PPA=common2+search126,121 feasible search,5 completed capacity-infeasible,0 search timeout. Search1077.351s (7.017 full/min), best Cost9.1350% below the common best Tangram-style seed, latency−20.9343%, energy−30.2187%, same area/MACs. Source139b remains immutable; cleanup incident and exact resume provenance are retained.
- A separate single measured archive design from that same run has latency−29.2692%, energy−32.5836%, EDP−52.3159%, Cost−8.3272%. Both complete recommendations are exported in `inputs/replay_v04/recommendations`. EDP archive selection is post-hoc; training objective remains Balanced.
- Same-space pilot controls remain visible: RL64 cost gain8.9984%; Random64 gain9.1302%; SA58 gain8.7869% with one tail timeout. No stable method/policy superiority claim. Current cross-model and multi-RNG statistical claims remain unproven; the local overnight launcher lets the user run them independently.
- Exact benchmark2.3972× on the same seed; fixed-spatial seed+inner1.5752% cost gain versus RL64 total8.9984%. Additional8 complete PPA, separately charged. Generic Q ranking learns (held-out rho0.7650 for64-query fit;0.7938 for128-query fit on a separately deduplicated set), but bottleneck-feature advantage is not established by fixed-data ablation.
- Windows cleanup failure was real and preserved. Live source now prefers EOF and verifies OS parent/child ownership before fallback termination.146 tests include a real Windows venv fallback test. The128 run resumed unchanged source with zero seed recomputation; known child-invocation wall totals1131.074s, with scope and human-repair pause exclusion explicit.
- Deliverables contain text/numeric reports, local `.cmd`/`.ps1` launchers, source-hash CPP/DLL/build recipe, portable inputs and recommendations. No images or private reference chat transcripts are included. New GUI buttons are not claimed; use the v0.4 launcher, while GUI04 retains its historical v0.3 controller.

The rest of this file preserves intermediate work notes. Any earlier 'current active' or 'remaining work' entries below are superseded by this final handoff.

## Historical work notes: authorization and scope

The user's latest request authorizes implementing a substantially more useful local RL search and running representative, honest comparisons. Preserve the paused recurring automation. Do not restart it. Reports contain text and numeric tables only. No external LLM API or paid quota resets. Existing dirty work and historical experiments are preserved.

This version fixes hardware at the existing v0.3 single-array profile: 16 PE per Compute chiplet, shared L2, no physical Core, IO count = 2 × Compute rows. Initial representative case: ResNet18, batch 1, two Compute chiplets in 2×1, four IO chiplets, Fusion Off. All PPA is uncalibrated analytical estimation. Baseline is the project's bounded Tangram-style DP adapter, not the original Tangram executable.

## Completed implementation

- Exact native resource-calendar scheduler in `native_calendar.cpp` with a Python fallback. Six thousand randomized DAG events preserve event order and exact start/finish times. Full ResNet18 seed replay preserves latency, energy, power, area, MACs and all byte/hop counters. New exact C++ slack and pure geometry caching are undergoing regression; frozen running experiments retain their original code.
- Structured global mapping and local critical-path-window proposals. Same spatial quota, partition, placement, DRAM flow and paid weight-retention domain is available to Random, SA and replay-Q. Bounded inner dataflow search ranks proposals; only the full oracle scores PPA and checks actual buffer lifetimes.
- Replay-Q is explicitly gamma=0 contextual episodic RL. Every feasible completed PPA provides one physical reward; replay can produce many SGD updates but these are never called PPA evaluations. Physical reward is a monotone logarithmic transform of the declared objective. Invalid/timeout states never supply fabricated utility.
- Standalone `simplified_rapidchiplet/replay_mapping_dse.py` freezes source and native artifacts, preserves per-trial mappings, features, probabilities, gradients, full metrics, budgets, checkpoints and stop/resume support. Saved candidate sets allow later policy attribution without extra PPA.

## Current experiment — DO NOT DUPLICATE

Output: `output/inmin_v04_20261103/r18_retention_pilot`

Frozen identity: `73ec2f92c2f33d02b47104103d491952f84839f86e9a768a66e5437191823390`.

Started with 64 full PPA ceiling (including two shared full seed replays), 900 seconds per arm, methods replay_rl / random / sa, seed 7, shortlist 64, 16 local proposals, 90-second per-oracle timeout. Methods execute sequentially. Replay RL is complete: 64 charged full PPA, 62 search full PPA (59 feasible), 1,888 SGD updates, zero timeout; best cost improvement 8.9984%, latency improvement 24.6103%, energy improvement 31.5539%, EDP improvement 48.3987%. Search wall 879.558 s; 4.229 full PPA/min. Last ten decisions normalized entropy 0.7420. Best state was found at search step 47, not the initial uniform warmup. Random/SA still need completion before claims about policy advantage. Random has already exceeded 9.13% cost improvement, so RL superiority must not be claimed. Observe `status.json`, `runner.log` and per-method checkpoints; process visibility alone is unreliable in this tool environment.

Use `<output>/STOP_REQUESTED` for a cooperative stop; current bounded oracle finishes and durable results remain. Resume using `--resume --out <output>`; never edit frozen code or relabel a new engine's results as the old experiment.

Earlier development evidence is preserved: r18_pilot v4_1 intentionally stopped at 22 full PPA with 5.6478% cost improvement; toy_dev and toy_retention are completed engineering checks. Toy_retention SA beats RL; it does not establish RL superiority.

## Latest completed validation and remaining work

The three-arm pilot is complete and source-stable: RL64 full / 8.9984% gain, Random64 / 9.1302%, SA58 / 8.7869% (900-second search ceiling, one tail timeout). Same-query58 comparison has the same best costs. Unique physical PPA across the shared common and these arms =182. No stable policy superiority claim.

Controlled same-source full-seed benchmark: Python coalesced median23.9362s versus native9.9850s, 2.3972×, exact physical metrics unchanged. Four fixed-spatial-mapping inner controls: dataflow-only0 gain; auto retention1.5752% gain; both1.5752%. Separate8 full PPA, never trained on by RL. Compared with seed+inner, RL64 still improves cost7.5420%, latency21.0511%, energy27.2354%.

Held-out92 feasible control states excluded from RL training: initial Q Spearman−0.3465, final0.7650; fixed-data features-off0.7726. Final Q learns useful ranking, but bottleneck feature advantage remains unproven. Reconstructed features-on final predictions exactly match the saved final model. All audit checks pass.

146 regression checks and the four-method Toy stop/resume validation pass. Cleanup was then repaired after an actual Windows taskkill five-second timeout following completed seed PPA: graceful EOF first, verified-own-child fallback; real Windows venv fallback test passes. The latest live parent additionally records invocation journals and clears stale return codes on resume; final source-stable regression is to be rerun after the active DNN check.

CURRENT ACTIVE: `r18_release_128`, session23257 after a resume. It reuses the two durable common PPA from its initial failed-cleanup invocation; frozen139b7fa2dffa727dafbaaea11c27fe6e7efa36197a47e1b2d70ad9dec6fb9310 unchanged. Initial failure and resume provenance are saved. Search budget128 full (including common2),1800s, seed7, RL only. Do not start another large run or mutate this snapshot. This is a throughput/quality supplement, not a128-query comparison against64-query controls.

1. Finish and audit the active128-query supplement; if the old frozen cleanup fails again after a durable checkpoint, resume without recomputing completed PPA. Do not patch the frozen source.
2. Final source-stable regression after live parent metadata changes; verify public wrapper syntax/dry-run and preserve all failure evidence.
3. Update final text report and create code/evidence ZIPs with CRC/SHA256 validation. Include `.cmd`, `.ps1`, portable inputs, matching CPP/DLL/build recipe. Exclude private reference conversations and images.
4. Do not claim stable multi-seed policy or bottleneck-feature superiority until actually tested. Recurring automation remains paused.

Native slack and exact geometry/retention cache tests now pass (two calendar tests plus seven replay-mapping tests). New native source SHA prefix is `d26087780f6d7b04`; the old running snapshot continues to use `67dc95dbda29ccbe`. New CLI freezes original seed input files too. Local overnight wrapper and portable unaltered ResNet18/ResNet50/MobileNetV2 seed inputs are ready; dry-run passed for all three. Default wrapper executes three RNGs from the same fixed seed per model with 256 full PPA / 3600 search seconds per arm. Source cohort changes are rejected. Actual integrated stop/resume and final regression still remain pending the current large experiment's global lock.

New analyzer independently checks source hashes, physical/query/reward/probability/gradient-update accounting and writes numeric curves at equal query/time budgets. An initial test on 13 unseen Random-arm states improved prediction Spearman from 0.313 to 0.467; this is provisional, not a search win or multi-seed conclusion. Final held-out and fixed-data feature ablation must be recomputed after all controls finish. No control-arm PPA trains the saved search policy.

Reference conversation has been downloaded read-only to `tmp/inmin_v04/shared_conversation.html` and extracted to `.txt`. The user's pasted P5/Gemini analysis was read. Local Gemini source and paper are available; its published architecture+mapping gains must not be attributed solely to SA or compared as same-hardware mapping gains.
