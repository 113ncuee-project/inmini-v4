# Full v0.8 stratified-search evidence

Frozen source9ee5237e2f5ab48bb98d6cb388f88427170faafec4e8c7218edf3e2cb3651562,212files. ResNet18/224/Batch1, full effectiveV3 domain, normalizedEDP, no hard PPA caps. Every arm inherits100paid fullPPA/2230.010633s with83feasible shared labels. Each primary ceiling96new/1800s; actual counts and timeouts are separate. The physical oracle is unchanged.

See completed_cases.json for the exact study status at export and all completed audited method rows. Missing cases remain pending; an extension is the same RNG with inherited costs. RNG7 includes Random/uniform wins; RNG37 includes an RL win. This is not a stable-win or original-paper result. EQUAL_BUDGET.csv and QUALITY_THRESHOLDS.csv retain matched floors and all5/10/20%thresholds.

Original JSON/JSONL bytes are preserved directly or as gzip. payload_locations.json maps every logical source path to its actual file; identical screening payloads are stored once under shared_payloads. original_sha256.json verifies decompressed bytes. The donor screening may contain older148D cached feature vectors: execution regenerates current260D vectors from physical decisions; do not train on stale cached vectors.

inference/ contains only safe weights/indices/source hashes, readable with torch.load(weights_only=True), not optimizer/replay/RNG or resumable training. replay_learning.py reproduces initial/shared/final after-cost diagnostics on identical untouched control designs. It removes the exact known before-cost term and reports same-architecture/group concordance; this does not prove future-Q accuracy or causal search benefit.

QA:188regression coverage uses the original tested7d98result plus explicit unchanged-source proof; the only changed CLIharness was actually rerun under9ee for313.599857s. ActualGUI16fields/8ToyPPA passed. Prior failed validation analyses are retained. A separate four-scenario overnight-launcher validation performed0newPPA/SGD.

New residual-controller development is not part of this source or its completed results. No teammate ZIP/Word/source or images are bundled. PPA remains an uncalibrated analytical resource-calendar model; Tangram/Gemini comparisons are same-domain adaptations, not original-paper binary reproduction. Read [team entry](../../docs/INMIN_TEAM_README.md), [numeric interpretation](../../docs/INMIN_STRATIFIED_SEARCH.md) and [learning diagnostic](../../docs/INMIN_POLICY_LEARNING_RESULTS.md).
