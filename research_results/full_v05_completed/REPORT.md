# Completed original full-space experiment (source336)

Read-only extraction from two completed, audited RNG cases. This is development evidence preceding the stronger-start experiment; gains may partly correct bounded DP grouping. EDP is normalized to a common fixed reference. No new PPA or SGD.

|RNG|Method|New / charged full PPA|EDP|Gain after shared %|Latency ms|Energy mJ|Area mm2|Average W|Search s|Groups|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|7|rl|141 / 193|0.06109825|23.221|3.620577|12.145677|106.221720|3.354624|2403.502|1|
|7|sa|107 / 159|0.07943955|0.173|4.182666|13.669557|106.221720|3.268145|2401.913|15|
|7|random|197 / 249|0.06634240|16.631|3.767000|12.675539|106.221720|3.364890|2401.328|8|
|17|rl|204 / 256|0.06012576|24.444|3.603077|12.010410|106.221720|3.333376|2086.290|1|
|17|sa|126 / 178|0.07690129|3.363|4.097998|13.506186|106.221720|3.295801|2402.978|14|
|17|random|204 / 256|0.06324071|20.529|3.634766|12.522502|106.221720|3.445201|2037.859|10|

Same-query and same-time comparisons:

|RNG|Axis|Budget|RL EDP|Random EDP|SA EDP|RL vs Random reduction %|
|---:|---|---:|---:|---:|---:|---:|
|7|full_ppa|159.000|0.06136426|0.07468540|0.07943955|17.836|
|7|charged_wall_s|3579.559|0.06109825|0.06634240|0.07943955|7.905|
|17|full_ppa|178.000|0.06047397|0.06945064|0.07690129|12.925|
|17|charged_wall_s|2609.696|0.06012576|0.06324071|0.07690129|4.926|

First verified target crossings after the original shared start:

|RNG|Method|Additional target %|Reached|Charged PPA|Charged s|Search s|
|---:|---|---:|---|---:|---:|---:|
|7|rl|5|True|78|1613.893|434.146|
|7|rl|10|True|88|1742.784|563.037|
|7|rl|20|True|147|2795.124|1615.377|
|7|sa|5|False|not reached|not reached|not reached|
|7|sa|10|False|not reached|not reached|not reached|
|7|sa|20|False|not reached|not reached|not reached|
|7|random|5|True|123|2135.563|955.816|
|7|random|10|True|202|3094.691|1914.944|
|7|random|20|False|not reached|not reached|not reached|
|17|rl|5|True|70|710.370|136.973|
|17|rl|10|True|94|905.428|332.031|
|17|rl|20|True|111|1168.243|594.845|
|17|sa|5|False|not reached|not reached|not reached|
|17|sa|10|False|not reached|not reached|not reached|
|17|sa|20|False|not reached|not reached|not reached|
|17|random|5|True|83|851.287|277.890|
|17|random|10|True|107|1083.339|509.942|
|17|random|20|True|250|2490.542|1917.145|

The shared best is c8_2x4, area106.221720mm2, compared with original c2_2x1 area64.794232mm2 (+63.937%). Thus joint DSE gains include additional paid hardware. Fixed-hardware controls and wider DP are reported separately.

Only two RNGs, no statistical superiority claim.
Initial bounded DP max-group bound may weaken the baseline: widened DP comparison remains pending.
Time-to-quality is observed first completed event; common stage is charged, not amortized away.
All PPA is uncalibrated analytical estimation; Gemini/Tangram original binaries were not executed.
