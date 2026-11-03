# RapidChiplet 專題資料

> **2026-11-04：兩個RNG的完整空間研究已完成。** RL新增141／204次完整PPA，兩組在共同查詢預算下優於本輪同域SA／Random；[完整數字、最佳mapping與逐步帳本](research_results/full_v05_completed/README.md)。原DP分組限制可能弱化起點，較強DP與新機制仍在本機驗證；未宣稱已打敗原版Tangram／Gemini。下面v0.5為目前已驗收程式。

> **完整系統新版：INMIN Full v0.5。** v3的1–16顆／50布局、Grouping、Partition、PE配置、獨立DRAM、Fusion／Tile／Cache與Dataflow已整合到新的多步Replay RL入口。雙擊 `start_inmin_full_gui.cmd` 使用新控制台，或 `run_inmin_full_overnight.cmd` 長跑。[完整設計空間及操作](docs/INMIN_FULL_SYSTEM_GUIDE.md) · [本輪進度](docs/inmin_v05_progress.md)。下方v0.4固定硬體實驗是歷史基線。

> **2026-11-03：新增 INMIN v0.4 本機 Mapping 搜尋。** Exact native calendar 保留完整 PPA／容量成本，結構化 Mapping + 內層 dataflow + replay-Q，並提供相同動作的 Random／SA 對照。新入口 `run_inmin_v04_overnight.ps1`；原 v0.3 工具與歷史結果保留。[新版操作說明](docs/INMIN_V04_LOCAL_GUIDE.md) · [本輪進度與證據](docs/inmin_v04_progress.md)。完整 PPA 與 SGD 更新分開記帳；不把候選生成收益直接歸功於 RL policy。

> **2026-11-01：新增 INMIN v0.2，GUI schema3 預設每 Compute 16 PE、共享 1 MiB L2、沒有實體 Core。** 使用獨立 profile 與 Python 顯式資源分析，保留舊版回放；報告僅文字與數字表格。103 項驗收已通過，代表模型的公平 RL 實驗仍進行中，尚未宣稱 RL 優勢。[技術設計／重現指令](docs/INMIN_V02_IMPLEMENTATION.md) · [目前進度與背景工作](docs/inmin_v02_progress.md)。下列日期內容為歷史版本紀錄。

> **2026-10-31：目前預設為 Gemini 2×2-core chiplet + Tangram 式起點 + 複合動作 RL。** 每顆四個完整 cores，獨立 IO／DRAM，RL 聯合選擇 chiplet 數量、合法 mesh 形狀、分割與資料流；Power／Area／Latency 偏好與硬限制共同決定最佳可行 archive。[完整操作與能力界線](docs/rl_dse_runbook.md) · [硬體契約](docs/gemini_hardware_contract.md)。啟動 `.\simplified_rapidchiplet\scripts\start_hybrid_gui.ps1`，開啟 `http://127.0.0.1:8765`。以下較早版本為歷史紀錄；V2 可用 `--engine legacy_v2` 重播。

> 2026-10-14：INMIN V2 已改為固定 4×4 PE 的相同 chiplet，片內與 interposer 均採 mesh，移除 I/O cluster。GUI 自動推薦 1／2／4／8／16 顆，保留 mapping／tile／fusion／reuse 搜尋與完整布局／traffic 輸出。[本次規格、論文假設與執行方式](docs/inmin_v2_fixed_mesh.md)。

> 2026-10-11：新增可執行的跨層 Tile／Fusion／Reuse DP；三種 overlap 模式、重算 MAC、UB cache-copy／容量帳本；每個 RL mapping 提案呼叫內層 DP。
> [本版架構與驗證](docs/hybrid_df_dse.md) · [最新架構圖](output/df_joint_20261011/system_architecture.svg)。CLI預設`--fusion dp`，`--fusion off`作對照；GUI提供內層開關與預算。

> 2026-10-09：新增七種真實模型、四種 Energy/Area/Delay 偏好與 Power 硬限制 GUI，whole-DAG 通訊感知 DP、片內 dataflow 搜尋、付費雙緩衝、每次結果的實際晶片布局／三级架構圖。
> [新架構](docs/hybrid_oct09_architecture.md) · [GUI／重跑方式](docs/hybrid_oct09_runbook.md) · [新驗證報告](docs/hybrid_oct09_validation.md)。執行 `.\simplified_rapidchiplet\scripts\start_hybrid_gui.ps1`，開啟 `http://127.0.0.1:8765`。

> 2026-10-06：本次融合 INDM IO-cluster mesh／multi-ring 與 Gemini layer pipeline，保留 RL 五個空間 operator，新增容量感知規劃與原生 RapidChiplet PPA 分工。
> [完整架構](docs/hybrid_architecture.md) · [執行方式](docs/hybrid_runbook.md) · [驗證結果](docs/hybrid_validation.md)。新入口：`simplified_rapidchiplet/hybrid_dse.py`。

> 2026-10-04：新預設已改成 communication-aware K mapping，實際 ResNet50 operator graph、
> 每 block N_active/ordered assignment、official canonical-flow network evaluation 與 Q-learning。
> [目前入口與完整操作說明](simplified_rapidchiplet/COMMUNICATION_AWARE_DSE.md)。
> `.\simplified_rapidchiplet\scripts\run_demo.ps1` 執行新流程；下文舊六模型比較均為 `--legacy`。


這個 repository 放的是 RapidChiplet 相關專題檔案，主要可執行版本在 `simplified_rapidchiplet/`。

目前上傳版本是新的 block-level preference-aware DSE 實作，不是
`260815_cnn-e2e-constrained-ppa` 的小幅修改版。本版本只移植 0815 專案的
Batch=1 analytical E2E latency 定義，其他 block、mapping、DAG scheduler、
preference RL 與專案結構都以本 repository 自己的實作為主。

## 下載後怎麼跑

```powershell
cd .\simplified_rapidchiplet
.\scripts\run_demo.ps1
```

或直接用 Python：

```powershell
cd .\simplified_rapidchiplet
python .\run.py --legacy --models shufflenet_v2_x1_0 --ppa-goal balanced --workload-search pareto-dp --dp-top-k 12 --out .\results\demo
```

詳細中文說明請看：

```text
simplified_rapidchiplet/README_zh-TW.md
```

## 主要資料夾

- `simplified_rapidchiplet/`：簡化版 DNN chiplet 評估工具，可直接執行。
- `presentation_work/`：簡報草稿、素材與投影片整理。
- `rapidchiplet.pdf`：RapidChiplet 相關參考資料。

新版 block-level preference-aware DSE 入口：

```powershell
cd .\simplified_rapidchiplet
py -3 .\preference_dse.py --legacy --models resnet50 --preference balanced --budget 64
```

它會依照 ResNet-50 的原生階層建立 18 個 semantic blocks，允許任意 block group 使用多顆 chiplet，並以 user preference 產生 constraint-first RL reward。詳細說明請看 `simplified_rapidchiplet/README_zh-TW.md`。

目前內建 6 個可直接拿來跑 DSE 的 CNN workload：

- `resnet18`：legacy stage metadata，作為較小的殘差網路基準。
- `resnet50`：18 個原生 Bottleneck semantic blocks。
- `shufflenet_v2_x1_0`、`shufflenet_v2_x2_0`：不同寬度的 channel-shuffle 網路。
- `mobilenet_v2`：由 Stem、InvertedResidual 與 Head 組成。
- `efficientnet_b0`：由 Stem、MBConv 與 Head 組成。

不指定 `--models` 時，preference DSE 會對這 6 個模型全部執行；也可以指定其中數個，例如：

```powershell
python .\preference_dse.py --legacy --models resnet18 resnet50 mobilenet_v2 efficientnet_b0 --preference latency --budget 8
```

這些模型目前是由 `configs/models.json` 提供的 analytical workload metadata，不需要安裝 PyTorch 或下載權重；因此適合先驗證 block/mapping/chiplet 搜尋流程。若之後要對接真實 ONNX/PyTorch graph，再將相同欄位替換成模型匯出的資料即可。

目前 `avg_latency_ns` 已改成 Batch=1 analytical end-to-end latency：
`sum(group compute) + sum(canonical-flow communication service)`。RapidChiplet
的 traffic-weighted ICI latency 會另外輸出為 `rapid_avg_latency_ns`，不再直接
拿來當 inference latency。

## 上傳 GitHub 注意事項

`simplified_rapidchiplet/results/` 是執行後產生的結果檔，檔案可能很大，不建議整包上傳。此 repository 已加入 `.gitignore`，用 Git 指令上傳時會自動排除結果檔、Python cache 與瀏覽器暫存檔。

Phase 1.5 correctness repair：詳見 [修復報告](docs/phase_1_5_correctness_repair.md)。FPS 僅為 inverse-E2E diagnostic；預設輸入沒有 FPS constraint/bonus/penalty。
