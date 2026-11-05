# INMIN：完整 chiplet／mapping 設計空間的本機 RL 搜尋

目前程式為 **Full v0.9**，包含原 v3 的有效搜尋變數：1–16 顆 Compute、50 種矩形布局、Layer Group、整數 B/K/H/W 切分、PE 份額／排序、DRAM Flow、Fusion／Tile／Cache，以及三種已定價的片內 Dataflow。

使用完整 compiler、通訊、資源排程、buffer lifetime 與能量模型評估候選；結果為解析 PPA。長時間搜尋由本機 Python 執行，可停止、續跑及保留已學到的 policy。報告使用文字、JSON 和數字表格。

|先看哪裡|用途|
|---|---|
|[完整操作與設計空間](docs/INMIN_FULL_SYSTEM_GUIDE.md)|安裝、GUI、長跑、停止／續跑、延長預算|
|[組員閱讀入口](docs/INMIN_TEAM_README.md)|程式地圖、實驗流程與論文結果判讀|
|[目前驗證報告](docs/INMIN_FULL_SYSTEM_VALIDATION.md)|204項回歸、明記來源的CLI驗收、實際GUI／並行與精確續跑|
|[搜尋技術與實驗協定](docs/INMIN_GRAPH_EDIT_DEVELOPMENT.md)|Q學習、張量邊界／PE指派特徵、資源守恆DP與公平對照|
|[新版分層探索與實驗進度](docs/INMIN_STRATIFIED_SEARCH.md)|完整V3空間、成本拆分Q、架構探索修正與待完成對照|
|[學習器修正與完整空間](docs/INMIN_RESIDUAL_RL_DESIGN.md)|固定共享先驗、自己量測殘差、normalized多步Q與因果控制|
|[修正後RL的實際開發結果](docs/INMIN_RESIDUAL_RL_RESULTS.md)|一個已完成種子、相同查詢／時間、固定起點排序與負面結果|
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
.venv/Scripts/python.exe -u -X utf8 simplified_rapidchiplet/full_system_dse.py --out output/r18_long --methods rl,sa,random --queries 512 --seconds 7200 --eval-timeout 600 --cost-value-q --residual-cost-q --episode-cost-rise .25 --architecture-refresh-interval 4 --seed-seconds 3600 --seed-dp-seconds 1800
```

這是每方法最多512個**含共同階段**的完整查詢、7200秒搜尋；起點生成與共同篩選另行記帳。可直接回放隨附的 ResNet18／ResNet50／MobileNetV2 工作量，不需要下載權重。

夜間多個RNG可用 `run_inmin_full_overnight.ps1`。相同實驗目錄請用 `--resume`；**已完成**實驗需要增加預算，使用 [延長工具](research_tools/extend_full_run.py) 在新資料夾接續，保留原結果與既有成本。

## 新版多種子長跑

GUI的RL成本表示選「固定先驗＋線上修正Q（實驗）」。指令列的新選項是 `--cost-value-q --residual-cost-q`；未指定的新舊介面預設值保留相容性。

```powershell
.\run_inmin_full_parallel.ps1 -Out output/r18_residual_long -Models resnet18 -Seeds "7,37,47" -Jobs 3 -Policy residual -Queries 612 -Seconds 7200 -ScreenSeconds 14400 -SeedSeconds 7200 -SeedDPSeconds 3600 -EvalTimeout 600
```

612是每方法含共同階段的查詢上限；若共同階段實際100次，則剩512次新PPA。7200秒是每個方法的搜尋時間，3方法約6小時，初始化另計；不同種子並行改善整批吞吐，不能當成單一搜尋的加速。相同模型共享已付費起點，保留各臂歷史成本，不交換搜尋標籤／policy。ResNet50可另開目錄以 `-Models resnet50` 執行；篩選及DP有獨立時間上限。

```powershell
.\run_inmin_full_parallel.ps1 -Out output/r18_residual_long -Stop
.\run_inmin_full_parallel.ps1 -Out output/r18_residual_long -Resume
```

`-DryRun`只核對設定。Resume使用原凍結程式、query／time／PPA限制與policy；已完成cohort的Resume不新增PPA。

## 結果應如何判讀

v0.9修復了部分種子多步價值分支近常數、共享暖身造成空Adam更新的問題，並以自己的完整PPA觀察修正共享成本先驗。第一組已審計ResNet18開發結果：新RL在共同強起點後降低26.48% EDP，較舊RL在相同查詢下改善6.09%、相同時間下改善11.98%；但舊Random7仍更好。新消融與fresh RNG確認尚未完成，不能宣稱穩定勝過Random／SA。

[完成主臂的原始證據與推論模型](research_results/full_v09_development/README.md)提供0新增PPA／SGD的重現方式，包含固定同一起點的細部成本排序，以及最佳控制方案預測名次退步的結果：

```powershell
.venv/Scripts/python.exe -X utf8 research_tools/replay_residual_development.py
```

v0.8的18項結果已完成。主方法同query floor的平均EDP，RL為0.040524、Random為0.043626、同域SA為0.047132；但加入相同horizon的均勻控制後，RL同時間平均優勢只有約1.7%。開發種子7由Random勝出，確認種子37／47由RL勝出。所有負面結果和完整帳本都保留，三個種子不足以宣稱穩定或統計顯著優勢。

PPA是同一未校準解析物理模型的結果，完整成本與資源守恆優先。Tangram-style起點及同域SA並非原論文binary；架構篩選、非學習提案和RL收益分開記錄。

