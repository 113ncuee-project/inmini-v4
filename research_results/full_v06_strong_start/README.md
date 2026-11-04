# Completed stronger initialization; RL comparison still running

Source `5f2fdce00493d95868b4b0837c03c283d590a0c5a274a34de6a6bc3ec37766de`, ResNet18, Batch1, normalized EDP objective. This is an INMIN Tangram-style adapter with widened contiguous segment DP/beam16 and declared inner refinements, not the original Tangram executable. No RL/search/control labels were imported into this archive.

Two architectures each consumed 24 new complete PPA queries (48 in total), including 15 infeasible outcomes. Combined with the original paid common2 + layout50, every comparison arm inherits 100 charged queries, 2230.0106331997085 seconds, and 83 feasible observations. The best EDP is 0.05149608430588703: T=3.1112512264ms, E=11.9126931784mJ, A=106.2217197mm2, average P=3.8289075W, compute c8_2x4. Analytical PPA is not calibrated chip measurement.

This is 14.354% below the old two-RNG RL best, demonstrating that some old gains came from an insufficiently strong initialization. It must not be reported as RL learning benefit. The new RL, same-domain SA/Random, feature/projection ablations, frozen-Q and uniform-horizon comparisons start from this same paid archive. `controller_study_protocol.json` records the frozen prospective protocol; its inclusion does not mean all arms have completed.

`proposals.json` contains complete proposed mappings; `events.jsonl` and `worker/accounting.json` retain feasible, infeasible, timing and real evaluation counts. DP prefix checkpoints retain progress and completed search coverage. `charged_shared_summary.json` records cumulative charging. The shared setup wall includes actual DP generation and evaluation time; it is not just the sum of worker CPU times.

The root code can generate its own widened initialization; these records are evidence, not a requirement to have the original machine's output paths. See [local guide](../../docs/INMIN_FULL_SYSTEM_GUIDE.md) and [numeric report](../../docs/INMIN_GRAPH_EDIT_RESULTS.md).
