# v0.6 completed development and adaptive stop

Source5f2, ResNet18 Batch1, identical charged strong start100PPA/2230.011s, EDP0.05149608430588703. All four arms completed and passed audits. F=graph/before-schedule/locality features; P=resource projection. All retain the complete effective V3 domain. Analytical PPA is uncalibrated.

|F/P|New PPA|Charged PPA|Best EDP|After-start gain %|Search s|Timeouts|Best step / phase|
|---|---:|---:|---:|---:|---:|---:|---|
|F0P0|21|121|0.051452048|0.08551|901.594|3|1 / uniform_warmup|
|F1P0|14|114|0.051452048|0.08551|903.326|3|1 / uniform_warmup|
|F0P1|13|113|0.051078444|0.81101|903.483|3|5 / uniform_warmup|
|F1P1|33|133|0.050291041|2.34007|905.222|1|18 / learning_search|

F0P0/F1P0/F0P1 best designs were found during uniform warmup; they do not demonstrate learned action choice. F1P1 best was a rank1 Q-selected A10 at step18 (p=.674695), after real updates, but no completed same-domain or frozen/uniform confirmation exists here. The observed2.3401% improvement cannot establish policy causality or statistical superiority.

F1P1 best: T3.067680ms,E11.799167mJ,A106.221720mm2,averageP3.846283W, c8_2x4, one Temporal group, Fusion0. Relative to strong start, it also changes five layer mappings; the A10 move merged the group boundary. Whole-design recommendation.json is retained; no metric-wise fabricated minima.

The prospective original study was adaptively stopped AFTER all four development audits: the best gain remained below the newly declared5% development handoff threshold and a real warm-start context-shift defect was diagnosed. ADAPTIVE_STOP.json preserves the reason. Pending SA/Random/LP5/frozen/uniform and RNG17 confirmation are NOT completed results. Source/protocol never changed; only the owned coordinator was stopped, and any active CLI was asked to save after its bounded query. No evaluator worker was killed.

policy_start_shift.json uses83 previously paid observations only,0PPA/0SGD. Rebased reward prediction error grew from~.5 to~6.6 for F1. Calibration/constant offsets alone do not prove bad action ranking. A separate cost-factored Double-Q representation is under engineering and fresh-PPA validation; no benefit of that new version is claimed here.

Raw compressed events, actual worker query/time ledgers, policy-update records, source/input identities and complete best mappings preserve negative and timeout results. Shared initialization evidence is in ../full_v06_strong_start. No original Tangram/Gemini binary was reproduced.
