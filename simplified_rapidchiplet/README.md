# Gemini 2x2-core joint RL DSE (2026-10-31, current default)

The production hybrid CLI/GUI now uses four complete cores per compute chiplet, explicit IO/DRAM, a source-traced Tangram-style DAG DP/beam seed, and transactional candidate actor-critic search over architecture and mapping. Power/Area/Latency preferences and strict constraints determine a separate best-feasible archive. [Runbook and limitations](../docs/rl_dse_runbook.md) · [hardware contract](../docs/gemini_hardware_contract.md) · [Tangram adapter provenance](TANGRAM_SEED_ADAPTER.md).

```powershell
..\.venv\Scripts\python.exe -X utf8 hybrid_dse.py --model toy --batch 1 --budget 24 --out results/gemini_rl_demo
```

Use `--engine legacy_v2` for the historical fixed-mesh implementation below. The analytical PPA is not silicon-calibrated, and the adapter does not execute the original Tangram binary.

# INMIN V2 fixed mesh update (2026-10-14, historical)

The current hybrid CLI/GUI deploys identical 4x4-PE chiplets on a direct interposer mesh, with a fixed on-die mesh and no I/O clusters. The system selects 1/2/4/8/16 compute chiplets, and exports placement, mapping, routes, link traffic and timelines. See [specification and run commands](../docs/inmin_v2_fixed_mesh.md). Earlier architecture descriptions below are historical.


# Simplified RapidChiplet

> 最新主路徑是 `hybrid_dse.py / hybrid_gui.py`，2026-10-11已加入Tile/Fusion/Reuse內層DP，見 [目前架構與驗證](../docs/hybrid_df_dse.md)。下方Phase與Q-learning說明屬歷史入口。

> Phase 2–10 已完成：預設為實際 FX+ShapeProp ResNet50 / integer K ownership /
> operator 內與 block 間 canonical traffic / official RapidChiplet / bounded assignment Q-learning。
> [目前執行方法、公式、限制與輸出](COMMUNICATION_AWARE_DSE.md)。
> 以下保留歷史 analytical pipeline 使用說明；所有舊命令需加 `--legacy`。


Traditional Chinese guide: [README_zh-TW.md](README_zh-TW.md)

This folder contains a compact DNN chiplet exploration prototype. It sweeps
model partitions and active chiplet counts on a mesh interconnect, then reports
latency, area, power, FPS, and a normalized PPA score.

The main evaluator uses only the Python standard library. If the official
RapidChiplet project path in `configs/defaults.json` exists, the evaluator calls
that engine for package/link metrics. If it does not exist, the evaluator falls
back to the local proxy in `simple_rapidchiplet/rapid_proxy.py`, so teammates can
still clone and run the simplified project directly.

Latency semantics: `avg_latency_ns` is the canonical Batch=1 analytical
end-to-end latency, computed as group compute time plus communication service
time. Rapid's traffic-weighted ICI latency is retained separately as
`rapid_avg_latency_ns` and is not used as the complete inference latency.

## Legacy Quick Start

```powershell
cd .\simplified_rapidchiplet
.\scripts\run_demo.ps1
```

Or run Python directly:

```powershell
python .\run.py --legacy --models shufflenet_v2_x1_0 --ppa-goal balanced --workload-search pareto-dp --dp-top-k 12 --out .\results\demo
```

Outputs are written to the selected output folder:

```text
best.csv
best.json
summary.csv
summary.json
chiplet_results.csv
report.html
```

Open `report.html` in a browser for the visual summary.

## Legacy Defaults

- models: all 6 entries in `configs/models.json` when `--models` is omitted
- available chiplets: up to `16`
- chiplet capacity: `8 x 8` PEs at `200 MHz`
- topology: `mesh` only
- workload search: `pareto-dp` with `--dp-top-k 12` (use brute-force as an oracle for validation)
- PPA goal: `balanced`

The initial model catalog intentionally covers several CNN families instead of
only ResNet-50:

| Model | Block representation | Purpose |
|---|---|---|
| `resnet18` | legacy stage fallback | small residual baseline |
| `resnet50` | semantic `Bottleneck` blocks | deeper residual network |
| `shufflenet_v2_x1_0` | legacy stage fallback | lightweight channel-shuffle model |
| `shufflenet_v2_x2_0` | legacy stage fallback | wider channel-shuffle model |
| `mobilenet_v2` | semantic `InvertedResidual` blocks | depthwise-separable mobile model |
| `efficientnet_b0` | semantic `MBConv` blocks | compound-scaled mobile model |

These entries are analytical workload metadata in
`configs/models.json`. They are deliberately dependency-free; the DSE engine
does not need PyTorch or pretrained weights to compare partitioning, mapping,
chiplet count, communication, and E2E latency. Pass one or more names to
`--models`, or omit the option to run the complete six-model catalog.

## Useful Commands

```powershell
python .\run.py --legacy --ppa-goal latency
python .\run.py --legacy --ppa-goal area
python .\run.py --legacy --ppa-goal power
python .\run.py --legacy --workload-search pareto-dp --dp-top-k 0
python -m unittest discover -s tests
```

## Edit Inputs

Change chiplet, network, power, and RapidChiplet assumptions in:

```text
configs/defaults.json
```

Change model workload assumptions in:

```text
configs/models.json
```
