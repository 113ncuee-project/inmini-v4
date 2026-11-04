# Wider Tangram-style seed comparison

All-segment DP/beam=16, screened and no-whole-output-prune variants, raw and declared q8 inner refinement. Complete INMIN oracle decides feasibility and EDP. This is an adapted baseline, not the original executable.

|Architecture|New full PPA|Charged full PPA|Best normalized EDP|Generation s|Search s|
|---|---:|---:|---:|---:|---:|
|single_array_c2_2x1|24|26|0.23483868|46.801|466.808|
|single_array_c8_2x4|24|76|0.05149608|81.170|449.233|
