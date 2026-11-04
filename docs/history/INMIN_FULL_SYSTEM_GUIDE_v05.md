# INMIN Full v0.5：完整 v3 設計空間的多步 Replay RL

更新：2026-11-03。本版整合在同一個 `full_system_dse.py`，舊 v0.3／固定硬體 v0.4 的程式和結果保留。代表硬體仍使用 `chiplet_single_pe_array_v0_3`，每 Compute 16 PE、共享 1 MiB L2、沒有實體 Core；IO=2R。報告僅文字與數值表格。

## 先啟動

檔案總管雙擊 `start_inmin_full_gui.cmd`，開啟本機 `http://127.0.0.1:8766`。這是新完整引擎的控制台，可設定模型、方法、目標、完整 PPA 次數、時間、Fusion 深度與 PPA 硬限制，也能停止、續跑及查看數值結果。原 GUI04 仍對應原 v3 流程。

長時間執行可雙擊 `run_inmin_full_overnight.cmd`。預設 ResNet18，RNG 7／17／37，每個 RNG 比較 RL／SA／Random，各方法 512 次完整 PPA 上限與 3600 秒搜尋；共同架構篩選另有最多 3600 秒。九個方法搜尋上限九小時，另外加三次共同準備／篩選；不代表必然完成 512 次或已收斂。腳本暫時防止自動睡眠，退出後還原；電腦必須保持開機。

沿用既有 `.venv`。新電腦以 Python 3.12 建立 `.venv`，再安裝 `simplified_rapidchiplet/requirements-replay.txt`。重播隨附輸入不需要重新下載神經網路權重。

```powershell
Set-Location C:\Users\user\Desktop\rapidchiplet
.\run_inmin_full_overnight.ps1 -DryRun -Seeds 7 -Methods 'rl,sa'
.\run_inmin_full_overnight.ps1 -Seeds 7,17 -Methods 'rl,sa,random' -Queries 256 -SecondsPerMethod 2400 -OutputRoot output/full_two_rng -KeepAwake
.\run_inmin_full_overnight.ps1 -Resume -OutputRoot output/full_two_rng -KeepAwake
```

單一模型、直接 CLI：

```powershell
.venv/Scripts/python.exe -u -X utf8 simplified_rapidchiplet/full_system_dse.py --out output/full_r18 --queries 512 --seconds 7200 --eval-timeout 300 --methods rl,sa,random
.venv/Scripts/python.exe -u -X utf8 simplified_rapidchiplet/full_system_dse.py --out output/full_r18 --resume
.venv/Scripts/python.exe -X utf8 simplified_rapidchiplet/scripts/analyze_full_system.py --out output/full_r18
```

最後一個指令只分析已完成的數字，不做額外 PPA。結果保存在 `REPORT.md`、`summary.json`、`analysis/curves.csv`、`analysis/equal_budget.json`、各方法的 `recommendation.json`。

## 與 v3 的設計空間對照

|變數|本版實際搜尋|執行或驗證|
|---|---|---|
|Compute 數量／布局|1–16 全整數；50 完整矩形；方向分開|每次切換完整重建 IO／DRAM／NoP／面積與能量|
|Layer Group|拓撲合法的 Temporal／Spatial 合併、拆分、模式切換與重配置|原 v3 A10 交易與唯一 PE 所有權驗證|
|Spatial Partition|所有支援的非空整數 B/K/H/W；不限制 K/P/balanced 三種模板|A2；與 quota 相容的 `partition_options`，Batch=1 故 B=1|
|PE Quota|合法份額、idle acquire/release、同 Spatial Group 轉移|A4；包含非 2 次方份額；保留整數分割|
|PE Placement|有序 PE 交換、idle relocation、跨層 PE 交換|A5／A6，連續操作可探索排列；模板是額外捷徑|
|DRAM／Flow|逐 tensor 的 activation／materialized output，以及獨立權重端點|A8；producer／consumer 選擇一致；真實搬移付費|
|Fusion|closed、contiguous、single-exit DAG 子圖；可啟用／拆分|真實 DAG 不刪除；最大深度可調，預設4與v3相同|
|Fusion Tile／Cache|末層正整數 H/W tile；halve/double、recompute／horizontal／full cache|A3／A7；付費 lifetime、halo、cache copy、重算 MAC|
|On-die Dataflow|output_stationary／weight_reuse／activation_reuse|A12 保留；可由內層選擇三種有成本模式|
|Weight Reuse|cold／persistent，auto 提案由容量估計產生|完整 WL2 存活容量決定是否合法|
|Microbatch|1|v3 物理編譯器本來固定 Batch=1／Microbatch=1；無多值可搜尋|
|任意 routing、input-channel package reduction、vertical-only cache|未支援|與 v3 相同的物理契約；未假裝提供無成本擴充|

這裡的「完整」指 v3 已有可執行設計變數整合進同一個新搜尋流程。巨大組合空間使用有界提案，不代表每個組合都能在有限時間列舉完成，也不保證全域最佳。`--fixed-architecture single_array_c2_2x1 --architecture-screen off` 可做固定硬體對照；`--fusion-off` 可做 Fusion Off 對照。

原 v3 交易執行器與驗證器保持相同。新版以 lazy sampling 直接抽 layer、partition、PE 交換及合法 quota，省去「先建出大量動作，再丟掉大部分」的時間。`--legacy-generator` 可切回原候選產生器做控制。提案只保留有限 shortlist，每種 family 輪流取得名額，避免架構候選佔滿整池。

## 哪些問題交給哪個演算法

1. 讀取已有容量感知 Tangram-style segment-DP／beam 產生的起點，完整重驗原 reference 與最佳 seed 的物理數字。`inputs/replay_v04/<model>` 保留原來源／hash。可以使用原 v3 seed 工具重建新的 `--seed-case`；這不是原版 Tangram 執行檔。
2. 外層共同架構篩選：每個布局套用事先宣告的 quota=8、balanced partition、spread、interleave、auto retention，逐一完整 PPA。時間／查詢不足會留下逐點失敗、timeout、未完成狀態；50 點仍全部保留在後續搜尋域。
3. 所有方法從共同最佳已評估設計出發。架構篩選物理執行一次，PPA 次數與時間對每個方法同額計帳。先篩選得到的收益記在 `gain_before_rl_pct`；後續收益記在 `gain_after_shared_screen_pct`。RL 可以用這批共同完整 PPA 做 terminal warmstart；對照方法同樣取得共同最佳起點。
4. 內層處理有界 Fusion／tile／cache beam 和三種 Dataflow 的局部排序。Fusion beam 估計 boundary、權重重載、halo 重算及 cache footprint，只產生提案；獨立原子動作與隨機代表保留其他方向。這沒有重現整套 DeFiNES optimizer；最終效益必須經全模型 PPA。
5. RL 選擇完整交易；完整 Oracle 回傳實際 latency／energy／power／area、容量與診斷。每個可執行新 PPA 可供學習；容量失敗／timeout不會提供虛構物理 reward。

## RL 的定義

本版是 action-conditioned Double Q，`gamma=0.9`，有獨立 target network、經驗重播、每8個合法步驟回到已驗證 elite 的 episode，以及實際成本差的 dense reward：

```text
r = 10 * log(penalized_cost_before / penalized_cost_after)
y = r + gamma * Q_target(next_features[argmax Q_online(next_features)])
```

terminal 不 bootstrap。模型是148維候選／狀態特徵→128 Tanh→64 Tanh→兩個輸出；第一個為 Q，第二個為即時 reward 輔助預測。特徵包含目標layer的拓撲位置、形狀及branch fanout，使相同形狀卻處於不同圖位置的動作可被區分，即使關閉瓶頸特徵也保留。採 Adam／Huber loss／gradient clipping，target 每64次 SGD 同步。預設每個實際新樣本做24次 minibatch SGD；SGD 次數另外計帳。ε-greedy 保留探索，訓練成熟後可明確利用 Q 排序，避免小 reward 讓單位溫度 softmax 一直近似均勻。

RL 會從較差但可執行的設計繼續走；最佳可行設計另存 archive。預設不依賴 bottleneck feature 或複合動作假說，`--guidance` 可啟用診斷特徵／舊 generator 指引；`--no-composites` 關閉舊 C 家族，保留完整原子變數與結構化提案。不能因為開啟開關就聲稱其有效。

本版預設主要目標為 **EDP=Energy×Latency**，降低只因平均功率下降卻延遲變慢而被視為好解的機會。Balanced 仍保留為相對 latency／area／平均 power 等權 Cost；也可預先選 `latency`、`energy`。不同目標必須另開實驗，不改寫歷史結果。`--max-area`、`--max-power`、`--max-latency-ms` 為硬限制，超額使用顯式 penalty引導探索，但 `recommendation.json` 只提供 strict-feasible 設計；找不到時 `state=null`，closest 另列。

有硬限制時，GUI摘要的Cost可能是closest探索值；請以推薦JSON的`strict_feasible`與`state`確認有沒有合法推薦。請保留GUI啟動視窗／PowerShell終端至工作結束；可鎖定螢幕，重開機後使用續跑。

## 如何核對與續跑

```powershell
Get-Content output/full_r18/status.json
Get-Content output/full_r18/runner.log -Tail 8
New-Item output/full_r18/STOP_REQUESTED -ItemType File -Force
```

當前有界 Oracle 完成後儲存。不要直接刪檔或改 frozen source。每次啟動有 source／input hash、環境、設定、parent invocation帳本；resume 使用同一份凍結程式。已完成的 durable PPA 不重算。停止時保留 pending transition、候選清單、RNG、target／online network、optimizer 與 replay，續跑可以得到相同軌跡。

`events.jsonl` 保存動作、選前機率／Q、完整 mapping、PPA結果、實際 query／wall；`learning.jsonl` 保存哪個已完成 transition 更新了模型、reward、terminal、successor數量、loss與梯度。保留初始模型和最新兩代 checkpoint，避免把每一步完整 replay 的重複副本累積到數十 GB。

完整評估沿用 v0.4 exact native calendar、所有 NoC／NoP路由、重算／能量／buffer lifetime。Worker最多保留3個架構的 geometry cache，避免50布局掃描耗盡記憶體；被淘汰的描述重建，物理模型不變。未安裝相符 DLL 時退回 Python 後端。

## 論文與可宣稱的範圍

本版將 [Gemini 的架構／Mapping 分層概念](https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/gemini.hpca24.pdf)、[DeFiNES 的多層資料搬移／融合取捨](https://arxiv.org/abs/2212.05344)、[Double Q 的選擇與 target 評分分離](https://arxiv.org/abs/1509.06461)用於工程設計。三者不是我們已完整重現的原版系統。

Tangram-style 起點、相同 Oracle 的 SA、Random 都必須比較。`equal_budget.json` 提供同實際查詢／同實際耗時；joint gain須與固定硬體 Mapping gain分開。PPA仍是未校準解析估計；現階段不把我們的 SA 對照寫成已打敗原版 Gemini，也不保證有限搜尋能找到數學上的最佳 Mapping。
