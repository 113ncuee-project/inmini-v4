# INMIN：完整 chiplet／mapping 設計空間的本機 RL 搜尋

目前程式為 **Full v0.8**，包含原 v3 的有效搜尋變數：1–16 顆 Compute、50 種矩形布局、Layer Group、整數 B/K/H/W 切分、PE 份額／排序、DRAM Flow、Fusion／Tile／Cache，以及三種已定價的片內 Dataflow。

使用完整 compiler、通訊、資源排程、buffer lifetime 與能量模型評估候選；結果為解析 PPA。長時間搜尋由本機 Python 執行，可停止、續跑及保留已學到的 policy。報告使用文字、JSON 和數字表格。

|先看哪裡|用途|
|---|---|
|[完整操作與設計空間](docs/INMIN_FULL_SYSTEM_GUIDE.md)|安裝、GUI、長跑、停止／續跑、延長預算|
|[組員閱讀入口](docs/INMIN_TEAM_README.md)|程式地圖、實驗流程與論文結果判讀|
|[目前驗證報告](docs/INMIN_FULL_SYSTEM_VALIDATION.md)|188項回歸覆蓋、真實CLI／GUI及精確續跑測試|
|[搜尋技術與實驗協定](docs/INMIN_GRAPH_EDIT_DEVELOPMENT.md)|Q學習、張量邊界／PE指派特徵、資源守恆DP與公平對照|
|[新版分層探索與實驗進度](docs/INMIN_STRATIFIED_SEARCH.md)|完整V3空間、成本拆分Q、架構探索修正與待完成對照|
|[RL學習檢查](docs/INMIN_POLICY_LEARNING_RESULTS.md)|未訓練設計的成本排序、細部排序與退步案例|
|[成本拆分Q前版結果](docs/INMIN_COST_VALUE_RESULTS.md)|v0.7強DP後RL改善16.06%，Random24.42%；保留負面結果|
|[RACE-RL來源核對](docs/INMIN_RACE_REFERENCE_AUDIT.md)|組員方法、不同硬體與可移植範圍|
|[已完成實驗與限制](docs/INMIN_GRAPH_EDIT_RESULTS.md)|數字結果、負面案例、較強起點與新消融進度|
|[原完整空間兩RNG資料](research_results/full_v05_completed/README.md)|完整最佳mapping、壓縮原始事件、相同預算比較與來源hash|

## 啟動

在專案根目錄建立 Python 3.12 環境並安裝依賴；已有 `.venv` 可直接使用。

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r simplified_rapidchiplet/requirements-replay.txt
```

雙擊 `start_inmin_full_gui.cmd`，開啟 `http://127.0.0.1:8766`。完整命令列範例：

```powershell
.venv/Scripts/python.exe -u -X utf8 simplified_rapidchiplet/full_system_dse.py --out output/r18_long --methods rl,sa,random --queries 512 --seconds 7200 --eval-timeout 600 --cost-value-q --episode-cost-rise .25 --architecture-refresh-interval 4 --seed-seconds 3600 --seed-dp-seconds 1800
```

這是每方法最多512個**含共同階段**的完整查詢、7200秒搜尋；起點生成與共同篩選另行記帳。可直接回放隨附的 ResNet18／ResNet50／MobileNetV2 工作量，不需要下載權重。

夜間多個RNG可用 `run_inmin_full_overnight.ps1`。相同實驗目錄請用 `--resume`；**已完成**實驗需要增加預算，使用 [延長工具](research_tools/extend_full_run.py) 在新資料夾接續，保留原結果與既有成本。

## 結果應如何判讀

目前v0.8已通過操作驗收，加入對所有方法相同的顆數分層探索。已完成的種子7由Random取得較低EDP，種子37由RL勝出；其餘確認仍在進行，尚未證明穩定優勢。完整數字與學習退步案例均保留於上方報告。前版v0.7的RL改善16.06%、Random24.42%的負面對照完整保留。原v0.5兩RNG的RL在共同查詢預算下優於當輪對照；固定2顆硬體的另一組實驗則由Random勝出。不能只選擇勝出的表格，也不能把共同架構篩選收益全部歸給RL。

同硬體、同Oracle的Tangram-style adapter／SA／Gemini-style operators是本機對照；本專案未重現原版論文執行檔的完整實驗。較大的Compute方案可能增加面積與平均功率；請一併看Latency、Energy、Area、Power及硬限制。

主要程式仍位於 `simplified_rapidchiplet/`。歷史版本、profile與報告保留，較早入口說明見 [原始README歷史副本](docs/history/README_before_v06.md)。
