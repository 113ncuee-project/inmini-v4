# Fixed Random-best Fusion pair

Same architecture, groups, per-layer PE assignment, partitions, dataflow, retention and FD. Only FusionStack differs. One new complete Off PPA, no policy updates or shared/search-label import. Analytical model only.

|Variant|Feasible|EDP|Latency ms|Energy mJ|DRAM read B|NoC hop-B|NoP hop-B|Executed MAC|Recompute MAC|Units|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|On (measured Random best)|True|0.163746090|9.152150|12.877117|18882212|99329408|38809328|1814073344|0|2500|
|Off (new counterfactual)|True|0.163006939|9.156854|12.812404|18082724|99520640|37705504|1814073344|0|2408|

Fusion-On EDP reduction relative to this exact Off mapping: -0.453448%.
