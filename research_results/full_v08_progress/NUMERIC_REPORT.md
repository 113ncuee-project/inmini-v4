# Cost-value full-domain RL: numeric study report

Study status: completed. This report includes only completed cases; missing/pending methods are not assumed successful. All queries, invalid schedules, timeouts and SGD are separate. Continuations retain old costs and are not new independent RNGs.

|Case / RNG|Method|New / charged PPA|EDP|After-shared gain %|Search s|T ms|E mJ|Area mm2|Average W|Timeouts|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|development/controls/7|sa|46 / 146|0.050603473|1.7334|1802.90|2.397929|15.188496|231.031|6.334005|1|
|development/controls/7|random|96 / 196|0.034905992|32.2162|1646.38|1.994401|12.596746|161.458|6.316055|1|
|development/frozen/7|rl|46 / 146|0.044697498|13.2021|1803.10|2.663623|12.077617|120.031|4.534282|1|
|development/stratified/7|rl|56 / 156|0.043013014|16.4732|1803.86|2.458148|12.593973|133.840|5.123359|1|
|development/uniform/7|rl|96 / 196|0.035371769|31.3117|1737.55|1.889071|13.476565|212.443|7.133963|0|
|confirmation/frozen/37|rl|64 / 164|0.044404875|13.7704|1802.80|2.328620|13.724697|180.046|5.893918|1|
|confirmation/frozen/47|rl|65 / 165|0.044290426|13.9926|1802.98|2.646027|12.047208|120.031|4.552942|0|
|confirmation/stratified/37|rl|78 / 178|0.036402697|29.3098|1803.01|2.160158|12.128830|159.333|5.614789|1|
|confirmation/stratified/37|sa|96 / 196|0.047666239|7.4372|1031.19|2.347924|14.611597|231.031|6.223198|0|
|confirmation/stratified/37|random|96 / 196|0.047065591|8.6036|1722.29|2.329522|14.541446|231.031|6.242246|1|
|confirmation/stratified/47|rl|58 / 158|0.042156854|18.1358|1803.12|2.430629|12.483040|133.840|5.135724|1|
|confirmation/stratified/47|sa|64 / 164|0.042815727|16.8563|1803.48|2.444999|12.603624|133.840|5.154858|1|
|confirmation/stratified/47|random|65 / 165|0.048132212|6.5323|1803.86|2.663524|13.006188|133.840|4.883075|1|
|confirmation/uniform/37|rl|91 / 191|0.045022468|12.5711|1801.18|2.247461|14.418097|231.031|6.415282|0|
|confirmation/uniform/47|rl|77 / 177|0.042493447|17.4822|1802.91|2.434337|12.563544|133.840|5.160972|1|
|extended_development/controls_7|sa|113 / 213|0.047133260|8.4721|3601.62|2.592956|13.082872|159.333|5.045543|1|
|extended_development/controls_7|random|192 / 292|0.032964994|35.9854|2744.13|1.744963|13.596827|314.414|7.792042|1|
|extended_development/stratified_7|rl|159 / 259|0.039315256|23.6539|3601.99|2.255260|12.546867|159.333|5.563379|2|

The shared widened-DP/inner benefits are not RL gains. A change in model capacity, scaling shortcut, trajectory, or projection must not be presented as an isolated policy-only effect. Frozen/shared-Q and uniform-horizon controls test parts of this attribution; immediate-reward head ranking does not prove future-Q causality.

EQUAL_BUDGET.csv and QUALITY_THRESHOLDS.csv use real charged floors and measured achievement times. Legacy/proportional generator differences are explicitly separate configurations; compare the cost/transition proportional pair to assess representation under the same generator. No extra PPA hard caps were applied in these declared cases. Latency, energy, area and average power describe one complete recommended design.

This evaluator uses a fixed uncalibrated analytical hardware profile; neither a chip measurement nor an original Tangram/Gemini binary result is claimed. Every case retains its source/input/environment identity, exact configuration, worker ledger, events, learning records and checkpoint. Failed or incomplete cases remain in status/execution logs and are not removed by this reporter.
