# Cost-value full-domain RL: numeric study report

Study status: interrupted_after_development. This report includes only completed cases; missing/pending methods are not assumed successful. All queries, invalid schedules, timeouts and SGD are separate. Continuations retain old costs and are not new independent RNGs.

|Case / RNG|Method|New / charged PPA|EDP|After-shared gain %|Search s|T ms|E mJ|Area mm2|Average W|Timeouts|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|development/cost_restart/7|rl|91 / 191|0.043227503|16.0567|1804.02|2.771740|11.224797|106.222|4.049729|1|
|development/frozen/7|rl|68 / 168|0.051037865|0.8898|1803.02|3.090803|11.884803|106.222|3.845215|1|
|development/no_restart_reference/7|rl|60 / 160|0.048699425|5.4308|1804.09|2.999040|11.687252|106.222|3.896998|1|
|development/one_step_cost/7|rl|52 / 152|0.046729661|9.2559|1803.26|2.440240|13.782591|147.649|5.648047|1|
|development/selected_controls/7|sa|81 / 181|0.050428660|2.0728|1802.39|2.737647|13.257782|133.840|4.842766|1|
|development/selected_controls/7|random|66 / 166|0.038918734|24.4239|1803.61|2.243626|12.484728|159.333|5.564532|1|
|development/transition_restart/7|rl|35 / 135|0.050476491|1.9799|1803.42|2.965349|12.251355|94.538|4.131505|2|
|development/uniform/7|rl|47 / 147|0.050342586|2.2400|1803.54|2.735530|13.245391|133.840|4.841983|2|

The shared widened-DP/inner benefits are not RL gains. A change in model capacity, scaling shortcut, trajectory, or projection must not be presented as an isolated policy-only effect. Frozen/shared-Q and uniform-horizon controls test parts of this attribution; immediate-reward head ranking does not prove future-Q causality.

EQUAL_BUDGET.csv and QUALITY_THRESHOLDS.csv use real charged floors and measured achievement times. Legacy/proportional generator differences are explicitly separate configurations; compare the cost/transition proportional pair to assess representation under the same generator. No extra PPA hard caps were applied in these declared cases. Latency, energy, area and average power describe one complete recommended design.

This evaluator uses a fixed uncalibrated analytical hardware profile; neither a chip measurement nor an original Tangram/Gemini binary result is claimed. Every case retains its source/input/environment identity, exact configuration, worker ledger, events, learning records and checkpoint. Failed or incomplete cases remain in status/execution logs and are not removed by this reporter.
