> 此文件是歷史 semantic-group RL 設計，透過 `--legacy` 執行。
> 新預設採每 block N_active + ordered physical assignment；請看 [目前模型與介面](COMMUNICATION_AWARE_DSE.md)。

# Semantic Block / Preference-aware DSE

> Phase 1.5（2026-10-03）：已修正 RL cache、producer boundary tensor，以及
> E2E/Rapid traffic 與 bandwidth 不一致。唯一 traffic source 是 canonical
> Flow list；官方 E2E 使用官方 per-flow latency / PHY bandwidth，fallback
> 則明確使用 proxy context。FPS 僅為 `1 / E2E_seconds`，沒有 FPS input、
> bonus、penalty 或 feasibility constraint。OC/IC/spatial 效率仍保留到 Phase 2；
> exact TorchVision operator graph / intra-block ownership 留待 Phase 3–5。
> 完整結果與限制見 `../docs/phase_1_5_correctness_repair.md`。


## 1. Overall architecture

The implementation now separates model understanding from design-space
search:

```text
model metadata / IR
    -> semantic block extraction and validation
    -> BlockGraph: fused operators + tensor/memory metadata + dependencies
    -> RL action: group contiguous blocks and allocate chiplets
    -> workload / RapidChiplet or local proxy evaluation
    -> preference-conditioned reward
    -> tabular Q-learning update
```

This follows the requested `State = (Model, User Constraint, Preference,
partial design)` concept. The old stage-level evaluator remains available for
comparison, but the new preference DSE uses the extracted block graph.

## 2. Block definition

A block is a model-native semantic unit with a useful tensor input/output. It
is not an arbitrary slice of MACs. The metadata for each block contains:

- `type`: semantic family such as `Stem`, `Bottleneck`, or `Head`;
- `operators`: fused operator sequence, such as `Conv+BN+ReLU`;
- `macs_g`: compute workload;
- `input_mb`, `output_mb`, `weight_mb`: tensor and weight memory estimates;
- `branch_closed`: residual/concat branches are joined before the boundary;
- `depends_on`: graph edges. Edge records also expose output traffic bytes.

For the configured ResNet-50, the block graph is exactly 18 units:

```text
Stem
  -> Conv2_x: 3 Bottlenecks
  -> Conv3_x: 4 Bottlenecks
  -> Conv4_x: 6 Bottlenecks
  -> Conv5_x: 3 Bottlenecks
  -> Head
```

Each Bottleneck keeps `1x1 Conv -> 3x3 Conv -> 1x1 Conv`, the shortcut, Add,
and ReLU inside one block. Conv/BN/Activation are represented as fused
operators and are not exposed as tiny independent DSE units. The same
principle is the extension point for MobileNet InvertedResidual, ShuffleNet
ShuffleUnit, EfficientNet MBConv, YOLO C2f/CSP, and TransformerBlock.

`simple_rapidchiplet/model.py` parses the JSON IR. `block_extractor.py` checks
the semantic boundary, operator presence, dependency validity, and DAG
acyclicity. An old `stages` entry can use `--block-split` as a compatibility
fallback, but an explicit semantic `blocks` entry is authoritative and is
never equal-split.

## 3. DSE grouping and parallelism

Block extraction is intentionally separate from grouping. After the 18 fixed
ResNet-50 blocks are created, the RL agent chooses:

```text
assign(group_length, chiplets)
```

The group must cover the next contiguous blocks, but both values are free
within the resource budget. In particular, all of these are legal when the
remaining budget permits them:

```text
assign(1, 4)   # one block replicated over four chiplets
assign(3, 2)   # a three-block group replicated over two chiplets
assign(6, 8)   # a six-block group replicated over eight chiplets
```

The former limitations—multi-block group fixed to one chiplet and only a
single block allowed to parallelize—are removed in both the direct-action
agent and the legacy brute-force/DP workload generators. The current
evaluator now also records an intra-block mapping strategy and a
dependency-aware schedule. The local scheduler is an architecture-level
estimator; RapidChiplet/local proxy still evaluates the interposer PPA.

## 4. Intra-block mapping and feasibility

For a parallel group, the action now contains:

```text
assign(group_length, chiplets, mapping_strategy)
```

Supported strategies are:

- `single`: one chiplet; no replication traffic;
- `output_channel`: every chiplet receives the input and computes a slice of
  output channels, so input broadcast traffic is added;
- `input_channel`: each chiplet computes partial output, so a reduction is
  added at the end;
- `spatial`: spatial partition with a conservative halo-exchange estimate.

The feasibility checker verifies that a group covers legal blocks, that a
parallel strategy exists for the selected block type, and that the estimated
working set fits the chiplet SRAM budget (`chiplet.sram_mb`, 16 MB in the
default config). Invalid mappings receive architecture violations before PPA
preference is considered.

The output-channel and input-channel traffic is intentionally different:

```text
output_channel: input broadcast ≈ input_tensor × (N - 1)
input_channel:  partial reduction ≈ output_tensor × (N - 1)
```

Therefore two mappings with the same `MACs / N` can have different network
latency, power, and reward.

## 5. Dependency-aware DAG schedule

The scheduler builds a topological order from `depends_on`, tracks when each
chiplet becomes available, and starts a block only after all predecessors and
communication are ready. If two blocks share only the same predecessor, they
can receive the same start time on different chiplets. A join block waits for
both.

Each candidate now includes:

- `mapping_plan`: strategy, partition fractions, memory per chiplet, and
  broadcast/reduction traffic;
- `schedule`: block start/finish cycles, assigned chiplets, critical path, and
  concurrent-ready information;
- `architecture_feasible` and `architecture_violations`;
- `schedule_latency_ns` as a dependency/resource makespan diagnostic;
- `e2e_latency_ns`, `e2e_compute_latency_ns`,
  `e2e_communication_serialization_ns`,
  `e2e_communication_path_latency_ns`, and
  `e2e_communication_latency_ns` as the canonical Batch=1 latency breakdown;
- `rapid_avg_latency_ns` and `rapid_avg_latency_cycles` as Rapid's
  traffic-weighted ICI diagnostics, not complete inference latency.

This is stronger than an ordered pipeline approximation, but it remains an
architecture-level estimator. SRAM banking, exact PE utilization, DRAM timing,
NoC microarchitecture, RTL timing closure, thermal, and package signal
integrity still require downstream tools.

## 6. Batch=1 E2E latency model

The E2E implementation follows the method in the reference
`260815_cnn-e2e-constrained-ppa` project. It is deliberately separate from
the RapidChiplet evaluator:

```text
E2E = sum(group compute times) + sum(boundary communication service times)
boundary service = serialization time + longest routed path latency
```

For each group, the compute time is the maximum assigned work on one chiplet
divided by the effective PE throughput. The mapping strategy efficiency is
included in that throughput. For each adjacent group boundary, the output
tensor produces canonical tensor-id/source/destination/byte flows. The same
flows feed E2E and RapidChiplet. Serialization uses each backend link bandwidth;
path delay uses backend per-flow latency, including actual official PHY geometry.
There is no second traffic matrix or local reconstruction of official bandwidth.

The current mapping extension also evaluates intra-group broadcast,
reduction, and spatial-halo traffic as additional communication events. This
keeps the E2E objective sensitive to the RL action that selected
`output_channel`, `input_channel`, or `spatial`.

The three latency values must not be conflated:

```text
avg_latency_ns / e2e_latency_ns = canonical Batch=1 E2E objective
rapid_avg_latency_ns            = Rapid traffic-weighted ICI diagnostic
schedule_latency_ns             = DAG dependency/resource makespan diagnostic
```

`achieved_fps` is now `1 / e2e_latency_seconds`, a Batch=1 diagnostic only. It does not affect ranking or feasibility.

## 7. User preference and reward

`PreferenceProfile` stores hard constraints and the user's soft direction:

```text
balanced = (1/3, 1/3, 1/3)
latency = (0.80, 0.10, 0.10)
area    = (0.10, 0.80, 0.10)
power   = (0.10, 0.10, 0.80)
```

這一版先固定四種 preference 對應的權重，讓 reward 行為容易理解與比較；
暫時不加入 preference strength 或 custom weight。這些權重也不能覆寫 hard
constraint。

For metric `x`, a supplied user limit `L` gives `ratio = x / L`. Without a
limit, a stable fallback scale keeps the metric as a soft preference. FPS is an inverse-E2E diagnostic and is never a constraint or reward term.

```text
cost = w_latency * latency_ratio
     + w_area    * area_ratio
     + w_power   * power_ratio

if all hard constraints pass:
    reward = 1 - cost
else:
    reward = -1 - min(sum(constraint_violation), 1)
```

This makes the preference change the feedback direction among feasible
designs, while a design that violates a hard constraint cannot win solely by
being good on its preferred metric. The profile name, model name, and exact
constraint tuple are part of the RL state, so separate user profiles do not
share an ambiguous Q-state.

## 8. Running the implementation

```powershell
cd C:\Users\user\Desktop\rapidchiplet\simplified_rapidchiplet
py -3 .\preference_dse.py `
  --models resnet50 `
  --preference balanced `
  --budget 64 `
  --max-area-mm2 800 `
  --max-power-w 16 `
  --out .\results\preference_balanced
```

Latency-first profile:

```powershell
py -3 .\preference_dse.py --models resnet50 --preference latency --budget 64 --max-latency-ns 5000
```

`--block-split N` is only for legacy stage-only configs. For ResNet-50 it is
ignored because the explicit 18-block semantic graph is authoritative.

The run writes:

- `preference_search_results.json`: complete graph, selected state, reward
  breakdown, and evaluated candidate;
- `preference_search_summary.csv`: compact per-model/per-preference summary;
- `manifest.json`: method/version and output metadata.

## 9. Verification

The test suite covers semantic ResNet-50 block extraction, MAC preservation,
operator/branch metadata, arbitrary multi-block parallel groups, action-mask
legality, mapping traffic differences, SRAM/Head feasibility, DAG branch
concurrency, preference-direction changes, topology, evaluator, and local
proxy fallback behavior.
