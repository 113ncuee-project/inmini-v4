# INMIN Full v0.8：本機操作與完整搜尋空間

更新：2026-11-04。程式來源 `9ee5237e2f5ab48bb98d6cb388f88427170faafec4e8c7218edf3e2cb3651562`。188項回歸覆蓋、完整CLI與實際GUI已通過；新RL技術的收益另由實驗對照驗證。[驗證證據](INMIN_FULL_SYSTEM_VALIDATION.md) · [技術與協定](INMIN_GRAPH_EDIT_DEVELOPMENT.md) · [新版分層探索](INMIN_STRATIFIED_SEARCH.md) · [前版成本Q結果](INMIN_COST_VALUE_RESULTS.md) · [歷史結果](INMIN_GRAPH_EDIT_RESULTS.md) · [RACE-RL參考核對](INMIN_RACE_REFERENCE_AUDIT.md)。

## 最短啟動方式

專案根目錄有 `.venv` 時，雙擊 `start_inmin_full_gui.cmd`；本機介面為 `http://127.0.0.1:8766`。可設定模型、目標、完整查詢上限、時間、Fusion深度與PPA硬限制，並停止／續跑。本版GUI保留舊Q作對照；要用新設定，選「起點校正多步Q」，Episode容許Cost惡化比例填0.25，重啟機率0.9。GUI顯示嚴格可行的整組設計數字；沒有合法推薦時會明確顯示。

新環境使用Python3.12：

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r simplified_rapidchiplet/requirements-replay.txt
```

隨附 `inputs/replay_v04/{resnet18,resnet50,mobilenet_v2}` 的工作量與已驗證初始mapping，重播不下載權重。新DNN可透過原seed工具建立相同格式的 `--seed-case`；原GUI04與歷史profile保留。

## 一次完整長跑

```powershell
.venv/Scripts/python.exe -u -X utf8 simplified_rapidchiplet/full_system_dse.py --seed-case inputs/replay_v04/resnet18 --out output/r18_long --methods rl,sa,random --queries 512 --seconds 7200 --eval-timeout 600 --seed 7 --cost-value-q --episode-cost-rise .25 --architecture-refresh-interval 4 --seed-seconds 3600 --seed-dp-seconds 1800
```

`--queries`是每方法**含共同階段**的完整PPA上限；`--seconds`是每方法搜尋時間上限，包含候選、特徵、SGD與Oracle。共同reference／seed、架構與起點強化實際只做一次，再對各方法同額計費。初始化另外有有界預算，總端到端時間會更長。7200秒不是保證完成512次或已收斂；報告列出實際查詢與時間。

```powershell
.\run_inmin_full_overnight.ps1 -Models resnet18 -Seeds 7,17,37 -Methods 'rl,sa,random' -Queries 512 -SecondsPerMethod 3600 -KeepAwake
.\run_inmin_full_overnight.ps1 -Resume -OutputRoot output/full_system_overnight -KeepAwake
```

腳本預設cost-Q、量測後0.25惡化reset／機率0.9、每4步顆數分層探索及600秒單次oracle；可用 -Policy transition、-DisableCostRestart、-ArchitectureRefreshInterval 0 做對照。預設三RNG×三方法，搜尋上限合計九小時，另加共同階段。每個模型的第一個RNG生成共同起點，後續RNG預設重用同一份已支付起點與共同觀察，重新初始化自己的網路、optimizer及RNG；不導入任何搜尋結果或訓練policy。每一方法仍繼承完整共同查詢與時間帳。若要重新生成不同起點，使用 -IndependentInitialization，並分開報告起點與隨機性。每個模型的第一個RNG生成共同起點，後續RNG預設重用同一份已支付起點與共同觀察，重新初始化自己的網路、optimizer及RNG；不導入任何搜尋結果或訓練policy。每一方法仍繼承完整共同查詢與時間帳。若要重新生成不同起點，使用 -IndependentInitialization，並分開報告起點與隨機性。每個模型的第一個RNG生成共同起點，後續RNG預設重用同一份已支付起點與共同觀察，重新初始化自己的網路、optimizer及RNG；不導入任何搜尋結果或訓練policy。每一方法仍繼承完整共同查詢與時間帳。若要重新生成不同起點，使用 -IndependentInitialization，並分開報告起點與隨機性。可指定多個模型；請先選擇符合可用時間的組合。腳本執行期間暫時避免自動睡眠；電腦要保持開機。一般本機執行不需要對話或LLM服務。

夜間腳本可另指定 `-SeedQueries 48 -SeedSeconds 3600 -SeedDPSeconds 1800` 延長初始化；`-ScreenSeconds` 控制共同架構篩選。這些是有界計算預算，不保證窮舉完成。硬限制可事前指定 `-MaxArea 150 -MaxPower 6.5 -MaxLatencyMs 5`，0表示不設定；不能看到輸贏後才加限制。`-DryRun`只列命令，不啟動PPA或建立實驗目錄。

GUI保留相容預設：顆數探索間隔為0；要使用本輪v0.8協定，將 `architecture_refresh_interval` 設為4，並選cost-value Q與成本惡化上限0.25。Python CLI同樣須明示 `--architecture-refresh-interval 4`。既有實驗按凍結來源和原設定續跑；更換程式後請開新輸出目錄。

## 停止、續跑與延長

```powershell
New-Item output/r18_long/STOP_REQUESTED -ItemType File -Force
.venv/Scripts/python.exe -u -X utf8 simplified_rapidchiplet/full_system_dse.py --out output/r18_long --resume
```

停止會在當前有界Oracle／DP checkpoint後保存。續跑使用原凍結程式與設定，恢復policy、target、optimizer、replay、候選、RNG與已完成PPA。已完成實驗的 `--resume` 不新增查詢。

需要更長時間時，在新目錄延長已完成實驗：

```powershell
.venv/Scripts/python.exe -u -X utf8 research_tools/extend_full_run.py --source output/r18_long --out output/r18_long_phase2 --extra-queries 512 --extra-seconds 7200
```

這會增加累計上限，保留已學到的policy與所有觀察，並從已驗證elite開始下一phase。原目錄不變，舊查詢／時間與實際複製成本繼續計帳；它是同一實驗的延長，不能算成另一個獨立RNG。新上限會影響後續epsilon退火。途中停止後，對新目錄正常使用 `--resume`。工具已以三方法各14→18次完整PPA驗證：共新增12次、0次共同重算、原資料不變，RL持續更新。

## 起點如何產生

1. 載入隨附容量感知Tangram-style segment-DP／beam的可行mapping，完整重驗reference與選中seed。兩次logical call若payload相同，只計一次真實PPA。
2. 共同架構篩選保留全部50布局，以事先宣告的q8／balanced／spread／interleave／auto mapping逐點完整評估。預算不足、timeout與未評估逐點記錄；其他布局仍可由RL一步提出。
3. 預設 `--seed-refinement wide`：允許全連續segment長度、beam16，同時保留／關閉保守whole-output容量剪枝；再測宣告的q8 P/K/balanced及q16/32 inner mapping。DP與內層分數只產生候選，真正排名仍由完整PPA決定。
4. 保留原始起點與所有已驗證seed；所有方法從共同最好設計出發，取得同一份已支付觀察。原seed→共同起點與共同起點→搜尋後改善分開報告。

Python CLI／GUI的額外強化預設最多48次PPA、1200秒評估／600秒DP段；夜間腳本改用3600秒評估／1800秒DP，進展中的DP可由prefix checkpoint延長；小query budget會保留搜尋份額而略過額外強化。可用 `--seed-queries`、`--seed-seconds`、`--seed-dp-seconds`調整新實驗。`--shared-case`只重用來源、輸入與數值環境等價的共同物理歷史，繼承原查詢與時間，重新計算當前特徵；不導入別的方法搜尋結果或訓練policy。

## 完整v3有效搜尋變數

|變數|範圍與執行方式|
|---|---|
|Compute／布局|1–16全整數，50完整矩形，旋轉分開；一步切換合法架構並重建mapping|
|IO／DRAM／NoP|IO=2R；每個布局重新計算資源、流量、面積、能量與功率|
|Layer Group|固定拓撲順序的連續Temporal／Spatial合併、拆分與模式切換|
|Partition|所有支援且非空的整數B/K/H/W；乘積等於PE quota；Batch1故B=1|
|PE quota|所有合法整數份額，包含非2次方；可保留idle、acquire/release、Spatial層間轉移|
|Placement|有序PE交換、idle relocation、同Spatial Group跨層交換；模板只是額外捷徑|
|DRAM Flow|獨立外部activation、權重及materialized output端點；付費interleave；遮罩當下無效內部FD，邊界啟用後仍可修改|
|Fusion|contiguous、closed、single-exit子圖；可啟用／拆分，最大深度可設定，預設4|
|Fusion tile／cache|全部有意義的正整數sink H/W，±1、halve/double及直接跳轉；recompute／horizontal／full cache|
|片內Dataflow|output_stationary／weight_reuse／activation_reuse三個已定價模式|
|Weight retention|cold／persistent；auto是候選，完整WL2 lifetime決定可行性|
|內部執行tile|由每個ownership region的實體W/A/O分池容量fitting產生完整input-channel tile|
|Microbatch|1，沿用v3實作物理契約；不宣稱支援未實作的多Batch、C-panel、任意routing或vertical-only cache|

「完整」指v3已實作的有效設計變數整合到同一系統。巨大組合使用有限、可旋轉的候選，不代表窮舉所有組合或保證全域最佳。`--fixed-architecture single_array_c2_2x1 --architecture-screen off`提供固定硬體控制，`--fusion-off`關閉Fusion；這些是明確的實驗選項。

硬體profile為 `chiplet_single_pe_array_v0_3`：每Compute16PE，共享1MiB L2，沒有實體Core。M=16輸出lane、T=16輸入向量為明記的實驗假設，每PE每拍256MAC；INT8 W/A、24-bit psum。PPA係數未經晶片校準，請依[硬體規格](specs/Chiplet_Hardware_Spec_v0.2_reviewed.md)及輸出假設解讀。

## RL及本版技術

建議使用 `--cost-value-q --episode-cost-rise .25`；未指定新旗標的原transition-Q仍保留相容性。新Q使用 `Q=phi(before)+V(after)+H(before,action)`、`phi=10 log(C)`。V僅取65個絕對after-design特徵，以實際完整PPA成本訓練；H以Double-Q target學習continuation。成本超過elite的1.25倍後以90%機率結束episode，仍保留實際負reward及10%較差中間狀態的探索。

外層為action-conditioned Double Q：gamma=.9、target network、replay、每8個合法步驟回到elite；每個實際新可行PPA預設24次minibatch SGD。完整物理成本差提供reward：

```text
r = 10 * log(cost_before / cost_after)
y = r + gamma * Q_target(next_features[argmax Q_online(next_features)])
```

terminal不bootstrap；timeout／容量失敗不提供虛構PPA reward。epsilon-greedy的低entropy本身不證明學習有效，須看同預算搜尋、held-out診斷與凍結／均勻控制。

149維共用狀態／候選描述外，預設加入111維：75個tensor-boundary編輯、14個實際before-schedule service／criticality／queue／pressure，以及22個ordered Part×PE ownership的邏輯局部性編輯。原transition-Q的graph residual分支零初始化，使其F on/off初始base Q相同；新成本拆分Q是另一種網路容量，不能將兩者差距直接當成純特徵消融。邏輯box需求不含完整tile、cache lifetime與排程，不能當PPA或reward。

額外Spatial整組整數DP對明示proxy求最佳份額，嚴守總quota及PE互斥；Temporal投影可重用序列stage的PE，以實際kernel／分池價格及當前量到的T/E產生候選。這些只是提案；完整Oracle仍決勝。有限family round-robin避免50架構候選擠掉其他變數。

`--no-boundary-features`、`--no-resource-projection`可分開消融；`--no-composites`保留完整原子變數。`--freeze-search-policy`、`--uniform-policy --freeze-search-policy --no-shared-warmstart`為因果控制。`--operators gemini5`是固定其他變數的五類LP語義對照，並非原版Gemini執行檔或預設搜尋域。

## v0.8的顆數分層探索

每4次決策用已完整驗證的elite作為mapping模板，依尚未嘗試顆數的maximin距離排定下一顆數；同顆數的所有合法矩形與保留／同比例PE份額一起交給選擇器。顆數排程是非學習的外層協調，RL學習其中mapping／layout選擇，普通步驟仍開放原完整動作。Random、SA、凍結Q與均勻策略使用相同機制。失敗或timeout只記嘗試，不能偽造PPA成本；不代表每個layout已成功完成評估。

這修正前版RL只實評少數架構的問題，但本身不證明線上RL有效。每步保存實際候選、Q分數、選擇機率、物理reward、學習更新、架構frame及checkpoint；同源同起點的對照才能區分協調器與學習器收益。可選的 `--service-features`、`--cost-rank-weight` 不是本輪主實驗設定；先前離線開發診斷未提供足夠採用理由。

## 結果、限制及可重現性

預設EDP=Energy×Latency；也支援latency、energy與原balanced。`--max-area`、`--max-power`、`--max-latency-ms`是硬限制；推薦JSON只提供strict-feasible設計，沒有時 `state=null`，closest另列。

```powershell
.venv/Scripts/python.exe -X utf8 simplified_rapidchiplet/scripts/analyze_full_system.py --out output/r18_long
```

分析不新增PPA。主要輸出：`REPORT.md`、`summary.json`、各方法`recommendation.json`、`events.jsonl`、`learning.jsonl`、`worker/accounting.json`、`analysis/equal_budget.json`、`analysis/curves.csv`。保存來源／輸入hash、有效設定、環境、checkpoint、真正PPA數、SGD數、覆蓋與負面結果。已完成的durable PPA不重複計費。

同一專案的昂貴實驗由全域鎖序列化；另一個工作持鎖時，新工作會明確失敗，請查看正在執行的實驗。新版來源與結果凍結後續跑，避免混入中途修改。native calendar與Python fallback保留相同物理定義。

原v0.5的兩RNG有正面結果，但固定2顆對照由Random勝出；一次可行的Fusion配對也顯示Fusion On流量較高、EDP稍差。v0.7完整開發對照由Random取得較低EDP；v0.8分層探索的匹配對照與獨立種子仍在進行，不能先宣稱穩定優勢或已打敗原版論文。請以[逐組數字](INMIN_GRAPH_EDIT_RESULTS.md)及完整mapping判讀。
