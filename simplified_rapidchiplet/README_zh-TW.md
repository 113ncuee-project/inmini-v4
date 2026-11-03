# Gemini 2×2-core 與複合動作 RL（2026-10-31，目前預設）

目前 `hybrid_dse.py`／`hybrid_gui.py` 使用四個完整 core 的 Gemini-like chiplet，各自有 PE array／GLB／vector／通訊資源，並保留獨立 IO 與 DRAM。DAG 先經可稽核的 Tangram 式 DP/beam adapter 產生起點，再由候選式 actor-critic 聯合探索硬體顆數、模板、mapping 與跨層動作。[新操作手冊、PPA 定義與限制](../docs/rl_dse_runbook.md) · [來源對應](TANGRAM_SEED_ADAPTER.md)。

```powershell
..\.venv\Scripts\python.exe -X utf8 hybrid_dse.py --model toy --batch 1 --budget 24 --out results/gemini_rl_demo
```

`--engine legacy_v2` 可重播以下舊 V2；不會改寫歷史實驗。

# INMIN V2 固定 mesh 更新（2026-10-14，歷史版本）

目前 hybrid CLI／GUI 採相同的 4×4 PE chiplet，片內與 interposer 均為 mesh，移除 I/O cluster。系統自動推薦 1／2／4／8／16 顆 compute chiplet，保留完整 layout、mapping、traffic 與 timeline。[規格、文獻假設與執行方式](../docs/inmin_v2_fixed_mesh.md)。下方較早的架構說明屬歷史紀錄。


# Simplified RapidChiplet 中文說明

> Phase 2–10 已完成：預設為實際 FX+ShapeProp ResNet50 / integer K ownership /
> operator 內與 block 間 canonical traffic / official RapidChiplet / bounded assignment Q-learning。
> [目前執行方法、公式、限制與輸出](COMMUNICATION_AWARE_DSE.md)。
> 以下保留歷史 analytical pipeline 使用說明；所有舊命令需加 `--legacy`。


這個資料夾是一個簡化版 DNN chiplet 評估工具，用來快速搜尋不同模型、chiplet 數量、mesh workload partition 與 PPA 目標之間的取捨。因為目前採用 passive interposer 的問題定義不區分 topology，正式流程固定使用 mesh。

注意：這是本 repository 的新版本，不是把 `260815_cnn-e2e-constrained-ppa`
小修後重新上傳。這裡只參考並移植 0815 的 Batch=1 analytical E2E latency
計算方式；block 定義、mapping strategy、DAG scheduler、preference-aware RL、
輸出格式與檔案結構都是本專案自己的實作。

## 新版：Fine-grained Block + Preference-aware RL

如果要使用附圖所示的 block-level AI DSE 流程，請執行：

```powershell
py -3 .\preference_dse.py --legacy `
  --preference balanced `
  --budget 64 `
  --out .\results\preference_balanced
```

不指定 `--models` 時，這個指令會依序跑目前內建的 6 個模型；如果只想跑其中幾個，可以明確指定：

```powershell
python .\preference_dse.py --legacy `
  --models resnet18 resnet50 mobilenet_v2 efficientnet_b0 `
  --preference latency `
  --budget 8 `
  --out .\results\preference_latency_four_models
```

目前的模型 catalog 是：

| 模型 | block 形式 | 用來覆蓋的需求 |
|---|---|---|
| `resnet18` | legacy stage fallback | 較小的 residual baseline |
| `resnet50` | 18 個 `Bottleneck` semantic blocks | 深層 residual network |
| `shufflenet_v2_x1_0` | legacy stage fallback | 輕量 channel-shuffle network |
| `shufflenet_v2_x2_0` | legacy stage fallback | 較寬的 channel-shuffle network |
| `mobilenet_v2` | `InvertedResidual` semantic blocks | depthwise-separable mobile network |
| `efficientnet_b0` | `MBConv` semantic blocks | compound-scaled mobile network |

這些模型目前是 `configs/models.json` 裡的 analytical workload metadata，
不是把 PyTorch 權重打包進 repository；所以執行工具不需要安裝 PyTorch，也不需要下載模型權重。每個 semantic block 都明確記錄 block type、融合後的 operators、MACs、input/output/weight memory、branch closure 和 dependency。這讓同一套 DSE 可以比較 residual、channel-shuffle、InvertedResidual 與 MBConv 等不同 CNN 結構。

ResNet-50 會依照模型原生階層建立 18 個 semantic blocks：
`Stem + Conv2_x 的 3 個 Bottleneck + Conv3_x 的 4 個 Bottleneck +
Conv4_x 的 6 個 Bottleneck + Conv5_x 的 3 個 Bottleneck + Head`。
因此不會再把 stage 依 MAC 數任意等分。對尚未提供 semantic `blocks` 的舊
`stages` 設定，才可用 `--block-split N` 作為相容性的 inferred fallback。

新版和舊版的差異：

- model 會輸出 `semantic_block_dag`，每個 block 包含 type、融合後 operators、MACs、
  input/output/weight memory、tensor I/O、branch closure 與 dependency traffic。
- Bottleneck 內的 `1x1 Conv -> 3x3 Conv -> 1x1 Conv + shortcut + Add + ReLU`
  是一個完整 block；Conv+BN+Activation 不拆開，residual branch 也不跨 block。
- block extraction 與 DSE grouping 分離：前者決定模型語意邊界，後者才決定連續 blocks
  如何 merge/group 以及配置幾顆 chiplet。
- 任意連續 block group 都可以配置任意數量的 chiplets；不再限制「多 block group 只能 1 顆」或「只有 single block 可以平行化」。
- RL action 是 `assign(group_length, chiplets, mapping_strategy)`，同時決定 block group、chiplet 數量與 `single`、`output_channel`、`input_channel`、`spatial` mapping。
- feasibility checker 會檢查 mapping 是否可用、Head 是否被錯誤平行化，以及每顆 chiplet 的工作集是否超過 `chiplet.sram_mb`。
- scheduler 會依照 dependency DAG 計算 block 的 start/finish cycle；互不依賴且資源足夠的 ready blocks 可以同時執行，join block 必須等待所有 predecessor。
- candidate 會輸出 mapping plan、broadcast/reduction traffic、critical path、schedule、E2E latency breakdown 與 architecture feasibility。
- user preference 會進入 state 與 reward。目前只保留 `balanced`、`latency`、`area`、`power` 四種固定模式，分別使用清楚可比較的固定權重；暫時不加入 strength 或 custom weights。
- `--max-latency-ns`、`--max-area-mm2`、`--max-power-w` 是 hard constraints；沒有提供的限制會作為 soft preference。

Reward 採 constraint-first：

```text
feasible:   reward = 1 - (wL * latency_ratio + wA * area_ratio + wP * power_ratio)
infeasible: reward = -1 - min(total_constraint_violation, 1)
```

因此 preference 會改變可行設計之間的回饋方向，但不會讓違反 user constraint 或硬體 feasibility 的設計超過可行設計。

`output_channel` 會增加 input broadcast，`input_channel` 會增加 partial-output
reduction，兩者即使都有 `MACs / N`，network latency 和 power 也不會相同。
這讓 RL 開始學習真正的 model-to-hardware mapping，而不是只學 chiplet 數量。

## E2E latency 修正：Rapid ICI 與 inference latency 分開

這一版參考 `260815_cnn-e2e-constrained-ppa` 的做法，修正原本把 Rapid
average network latency 和 scheduler makespan 混在一起的問題。

原本的作法是：

```text
latency = max(Rapid average ICI latency, block scheduler makespan)
```

這不是完整的 Batch=1 inference latency。Rapid 的 `latency.avg` 是所有
inter-chiplet traffic flow 的 traffic-weighted path latency，只表示 ICI
network 的診斷值。

目前正式使用的 E2E analytical proxy 是：

```text
E2E latency
  = sum(group compute time)
  + sum(communication event service time)

communication event service time
  = bottleneck-link serialization time
  + longest routed path latency
```

### 1. Group compute time

每個連續 block group 先由 RL 決定使用幾顆 chiplet。若 group 內有
`parts` 顆 chiplet：

```text
group_ops = group 內所有 block 的 MACs × op_per_mac
ops_per_chiplet = group_ops / parts
effective_ops_per_second
  = PE rows × PE cols × ops_per_PE_per_cycle
  × frequency × utilization × mapping_efficiency
group_compute_time = ops_per_chiplet / effective_ops_per_second
```

`mapping_efficiency` 由 mapping strategy 決定；例如 output-channel、
input-channel、spatial 會因同步與資料交換而比 single 有不同效率。這部分
保留了目前 mapping model 的影響，不再只用 `MACs / N` 當成完整效能。

### 2. Group boundary communication

對每一對相鄰 group，取前一個 group 最後一個 block 的 output tensor，並把
它平均分成所有 source-chiplet 到 destination-chiplet 的 directed flows：

```text
pair_bits = boundary_tensor_bits / (source_count × destination_count)
```

Phase 1.5 的 E2E 與 Rapid 共用 canonical Flow list。每個 event 直接使用
backend 回傳的 route、per-flow path latency 與每條 directed link bandwidth：

```text
serialization_time = max_link(link_load_bits / effective_link_bandwidth) / frequency
path_latency = max(backend_per_flow_path_cycles) / frequency
boundary_service_time = serialization_time + path_latency
```

官方 path 使用官方 chiplet / packaging / PHY intermediates，沒有再使用
本地中心距離或 proxy bandwidth 重建。Local fallback 明確標示不同 backend。
Rapid 的 global traffic-weighted average 仍為診斷值，不代替 event critical path。

### 3. Mapping communication

目前細 block 版本還允許一個 group 用多顆 chiplet，因此會有 group 內的
mapping traffic：

- `output_channel`：input broadcast；
- `input_channel`：partial-output reduction；
- `spatial`：halo exchange 的保守估計。

這些 traffic 會依 group 合併成額外 communication event，使用同一個
`bottleneck serialization + longest path` 公式。這是針對本專案 mapping
action 的延伸：參考專案主要描述 group-to-group boundary，而本專案不能
忽略 group 內 parallel mapping 的資料交換，否則 RL 選不同 mapping 時
E2E reward 會看不出差異。

### 4. E2E、scheduler、Rapid 三個 latency 的角色

結果中的欄位意義如下：

| 欄位 | 意義 | 是否作為 preference reward 的 latency |
|---|---|---|
| `avg_latency_ns` / `e2e_latency_ns` | Batch=1 analytical E2E | 是 |
| `e2e_compute_latency_ns` | 所有 group compute time 的總和 | 是 E2E 的組成項 |
| `e2e_communication_latency_ns` | serialization + path latency 的總和 | 是 E2E 的組成項 |
| `rapid_avg_latency_ns` | Rapid traffic-weighted ICI average path latency | 否，network diagnostic |
| `avg_latency_cycles` | Rapid ICI average cycles，與 `rapid_avg_latency_ns` 對應 | 否 |
| `schedule_latency_ns` | DAG scheduler 的 dependency/resource makespan | 否，architecture feasibility/debug |

因此：

```text
avg_latency_ns == e2e_latency_ns
e2e_latency_ns = e2e_compute_latency_ns + e2e_communication_latency_ns
rapid_avg_latency_ns != e2e_latency_ns（一般情況）
schedule_latency_ns != e2e_latency_ns（兩者模型不同）
```

`achieved_fps = 1 / E2E_latency_seconds`，只作 Batch=1 diagnostic，不參與 reward、constraint 或 candidate filtering。

## 從 user input 到 reward 的程式流程

目前一個 candidate 的完整資料流如下：

```text
configs/models.json
        ↓
model.py / block_extractor.py
  讀取 semantic blocks、operators、tensor memory、dependencies
        ↓
preference_dse.py
  RL state/action：下一段 block、group chiplet 數、mapping strategy
        ↓
workload.py
  將 contiguous block groups 轉成 ops、group_specs、traffic
        ↓
mapping.py
  檢查 SRAM、Head、strategy，建立 broadcast/reduction/halo traffic
        ↓
e2e_latency.py
  計算 group compute + boundary/mapping communication service
        ↓                 ↓
dag_scheduler.py       rapidchiplet_engine.py
  dependency/resource   Rapid area/power/ICI/throughput
  makespan diagnostic   network diagnostic
        ↓                 ↓
evaluator.py
  合併 E2E、Rapid、schedule 與 feasibility
        ↓
preference_dse.py / score_result()
  依 balanced/latency/area/power 固定方向計算 reward
```

各部分的責任是分開的：

1. `model.py` 不做任意 MAC 等分；它把 JSON 中的原生 semantic block 讀成
   `ModelSpec`，並驗證 dependency 名稱。`block_extractor.py` 再檢查 block
   邊界、operator metadata 與 DAG 無循環。
2. `preference_dse.py` 的 action 可以選 `group_length`、`chiplets` 與
   `mapping_strategy`。因此 RL 探索的不是只有 chiplet 數量，也會比較
   output-channel、input-channel、spatial 與 single。
3. `workload.py` 把 action 轉成實際的 group partition；multi-block group
   和 single-block group 都可以使用多顆 chiplet。
4. `mapping.py` 讓不同 parallelism 產生不同 efficiency 與 traffic，並在
   PPA reward 前先判斷 SRAM、Head 不可平行化等 architecture constraint。
5. `e2e_latency.py` 讀取 canonical events 與 backend NetworkContext，使用每條 link 的 effective bandwidth 及官方 per-flow path latency，再加總 serial compute 與 communication。
6. `dag_scheduler.py` 仍保留，用來做 block dependency、resource ready time、
   branch concurrency 與 critical path 的架構檢查；它的 makespan 不會再被
   偷拿來和 Rapid latency 做 `max()` 當成 E2E。
7. `rapidchiplet_engine.py` 仍負責 Rapid 官方 engine 或 local proxy 的
   area、power、throughput、ICI average latency。`evaluator.py` 將它命名
   成 `rapid_avg_latency_ns`，並把 E2E 放入 `avg_latency_ns`。
8. `simple_rapidchiplet/preference_dse.py` 的 `score_result()` 讀取
   `candidate.avg_latency_ns`，所以現在 user 的 latency preference / limit
   自動作用在 E2E，而不是 Rapid ICI average。四種 preference 的固定
   reward 方向保持不變，沒有加入 high/medium/low 或 custom weight。

輸出檔案：

- `preference_search_results.json`：完整 block graph、最佳 state、reward breakdown 與 candidate。
- `preference_search_summary.csv`：每個 model/preference 的摘要。
- `manifest.json`：這次實驗的版本與方法資訊。

`preference_search_results.json` 的 candidate 內也會保留：

- `e2e_group_compute_times_ns`：每個 group 的 compute time；
- `e2e_boundary_timings`：每個 group boundary 或 mapping event 的 traffic、
  bottleneck link、max path、serialization、path latency；
- `e2e_path_latency_source`：目前是
  `official_rapidchiplet_per_flow` 或 `local_proxy_per_flow`；
- `rapid_avg_latency_ns`：Rapid ICI 診斷值，方便和 E2E 對照。

完整設計與 reward 說明請看 `PREFERENCE_BLOCK_DSE.md`。

## 專案內容

```text
simplified_rapidchiplet/
├─ run.py                         # 主程式入口
├─ configs/
│  ├─ defaults.json               # chiplet、network、power、RapidChiplet 設定
│  └─ models.json                 # ResNet / ShuffleNet workload 假設
├─ simple_rapidchiplet/           # 評估核心程式
├─ tests/                         # 單元測試
├─ tools/                         # 簡報圖表輔助工具，非必要
└─ results/                       # 執行後自動產生，GitHub 上傳時可不放
```

## 快速開始

在 PowerShell 進入資料夾：

```powershell
cd C:\Users\user\Desktop\rapidchiplet\simplified_rapidchiplet
```

如果電腦有安裝 Python：

```powershell
python .\run.py --legacy --models shufflenet_v2_x1_0 --ppa-goal balanced --workload-search pareto-dp --dp-top-k 12 --out .\results\demo
```

如果 Windows 使用 `py` launcher：

```powershell
py -3 .\run.py --legacy --models shufflenet_v2_x1_0 --ppa-goal balanced --workload-search pareto-dp --dp-top-k 12 --out .\results\demo
```

也可以直接跑 demo 腳本：

```powershell
.\scripts\run_demo.ps1
```

執行完成後會在 `results/demo/` 看到：

```text
best.csv              # 每個 model 的最佳結果
best.json             # best.csv 的 JSON 版本
summary.csv           # 每個 model + topology 的最佳結果
summary.json          # summary.csv 的 JSON 版本
chiplet_results.csv   # 所有候選設計的完整 sweep
report.html           # 可直接用瀏覽器打開的視覺化報告
```

## 常用指令

同時評估 ResNet50 與 ShuffleNetV2（固定 mesh）：

```powershell
python .\run.py --legacy --models resnet50 shufflenet_v2_x1_0 --ppa-goal balanced --workload-search pareto-dp --dp-top-k 12
```

切換 PPA 目標：

```powershell
python .\run.py --legacy --ppa-goal latency
python .\run.py --legacy --ppa-goal area
python .\run.py --legacy --ppa-goal power
python .\run.py --legacy --ppa-goal balanced
```

FPS 僅為 inverse-E2E 結果，不提供 FPS 目標輸入。

切換 workload search：

```powershell
python .\run.py --legacy --workload-search brute-force
python .\run.py --legacy --workload-search pareto-dp --dp-top-k 12
python .\run.py --legacy --workload-search pareto-dp --dp-top-k 0  # 僅用於完整 frontier / 研究驗證
```

## 預設設定

- 預設模型：`resnet50`、`shufflenet_v2_x1_0`
- 正式 topology：`mesh`（passive interposer 問題定義下不做 tree 比較）
- 最多可用 chiplets：`16`
- 每個 chiplet：`8 x 8` PE array、`200 MHz`
- 預設 workload search：`pareto-dp`（需要完整驗證時再使用 `brute-force`）
- 預設 PPA goal：`balanced`

## 官方 RapidChiplet 與本地 fallback

`configs/defaults.json` 內的 `rapidchiplet.root` 可以指向官方 RapidChiplet 專案路徑。如果該路徑存在，程式會呼叫官方 engine 取得 area、power、link、ICI latency、throughput 等指標。

正式 PPA latency 是 serial Batch=1 E2E，組成為 compute 加 canonical
communication event service。官方 route、bandwidth、per-flow latency 由
RapidChiplet 回傳；local proxy 使用自己的有效參數，JSON 會標示來源。
`rapid_avg_latency_ns` 仍是 traffic-weighted ICI diagnostic。

如果組員電腦沒有官方 RapidChiplet，程式會自動改用 `simple_rapidchiplet/rapid_proxy.py` 的本地近似模型，仍可直接跑完整流程並產生報告。這個 fallback 主要用於課堂展示、組內比較與快速驗證；若要和官方 RapidChiplet 結果嚴格對齊，請先下載官方專案，並修改：

```json
"rapidchiplet": {
  "root": "你的官方 RapidChiplet 專案路徑"
}
```

## 修改輸入資料

調整 chiplet、network、power、最大 chiplet 數：

```text
configs/defaults.json
```

新增或修改模型 workload：

```text
configs/models.json
```

每個 model 由 stage 組成，主要欄位是：

- `macs_g`：該 stage 的 MAC 數，單位為 Giga MACs
- `output_mb`：該 stage 輸出的 activation 大小，單位為 MB

## 測試

```powershell
python -m unittest discover -s tests
```

如果本機沒有 `python` 指令，可以改用：

```powershell
py -3 -m unittest discover -s tests
```

## GitHub 上傳建議

建議上傳 source code、config、README、tests，不要上傳 `results/`、`__pycache__/`、瀏覽器 profile 或 crash dump。這些檔案已經放進 `.gitignore`，用 Git 指令上傳時會自動排除。本工作區另外有 `.codex-pptx-build/`、`editable_outputs/` 等簡報產物，若不想放上 GitHub，請不要使用未檢查的 `git add .`。

```powershell
# 先確認目前 branch、未提交修改與 remote
git branch --show-current
git status
git remote -v

# 只 stage 程式、設定、README 與測試；不要把簡報產物一起加入
git add README.md `
  simplified_rapidchiplet/README.md `
  simplified_rapidchiplet/README_zh-TW.md `
  simplified_rapidchiplet/PREFERENCE_BLOCK_DSE.md `
  simplified_rapidchiplet/configs `
  simplified_rapidchiplet/run.py `
  simplified_rapidchiplet/preference_dse.py `
  simplified_rapidchiplet/simple_rapidchiplet `
  simplified_rapidchiplet/tests

# 確認 staged diff 後再 commit
git diff --cached --check
git diff --cached --stat
git commit -m "Use analytical E2E latency for block-level DSE"

# 只有尚未設定 origin 時才需要這行
git remote add origin https://github.com/<你的帳號>/<repo-name>.git
git branch -M main
git push -u origin main
```

若要保留實驗結果給組員看，建議只挑選少量代表性的 `best.csv`、`summary.csv` 或 `report.html`，不要整包 `results/` 都放進 GitHub。
