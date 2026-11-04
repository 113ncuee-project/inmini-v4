# INMIN Full v0.6：本機操作與完整搜尋空間

更新：2026-11-04。程式來源 `5f2fdce00493d95868b4b0837c03c283d590a0c5a274a34de6a6bc3ec37766de`。173項回歸、完整CLI與實際GUI已通過；新RL技術的收益另由實驗對照驗證。[驗證證據](INMIN_FULL_SYSTEM_VALIDATION.md) · [技術與協定](INMIN_GRAPH_EDIT_DEVELOPMENT.md) · [數字結果](INMIN_GRAPH_EDIT_RESULTS.md)。

## 最短啟動方式

專案根目錄有 `.venv` 時，雙擊 `start_inmin_full_gui.cmd`；本機介面為 `http://127.0.0.1:8766`。可設定模型、目標、完整查詢上限、時間、Fusion深度與PPA硬限制，並停止／續跑。GUI顯示嚴格可行的整組設計數字；沒有合法推薦時會明確顯示。

新環境使用Python3.12：

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r simplified_rapidchiplet/requirements-replay.txt
```

隨附 `inputs/replay_v04/{resnet18,resnet50,mobilenet_v2}` 的工作量與已驗證初始mapping，重播不下載權重。新DNN可透過原seed工具建立相同格式的 `--seed-case`；原GUI04與歷史profile保留。

## 一次完整長跑

```powershell
.venv/Scripts/python.exe -u -X utf8 simplified_rapidchiplet/full_system_dse.py --seed-case inputs/replay_v04/resnet18 --out output/r18_long --methods rl,sa,random --queries 512 --seconds 7200 --eval-timeout 600 --seed 7
```

`--queries`是每方法**含共同階段**的完整PPA上限；`--seconds`是每方法搜尋時間上限，包含候選、特徵、SGD與Oracle。共同reference／seed、架構與起點強化實際只做一次，再對各方法同額計費。初始化另外有有界預算，總端到端時間會更長。7200秒不是保證完成512次或已收斂；報告列出實際查詢與時間。

```powershell
.\run_inmin_full_overnight.ps1 -Models resnet18 -Seeds 7,17,37 -Methods 'rl,sa,random' -Queries 512 -SecondsPerMethod 3600 -KeepAwake
.\run_inmin_full_overnight.ps1 -Resume -OutputRoot output/full_system_overnight -KeepAwake
```

預設三RNG×三方法，搜尋上限合計九小時，另加共同階段。可指定多個模型；請先選擇符合可用時間的組合。腳本執行期間暫時避免自動睡眠；電腦要保持開機。一般本機執行不需要對話或LLM服務。

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

額外強化預設最多48次PPA、1200秒評估／600秒DP段，進展中的DP可由prefix checkpoint延長；小query budget會保留搜尋份額而略過額外強化。可用 `--seed-queries`、`--seed-seconds`、`--seed-dp-seconds`調整新實驗。`--shared-case`只重用來源、輸入與數值環境等價的共同物理歷史，繼承原查詢與時間，重新計算當前特徵；不導入別的方法搜尋結果或訓練policy。

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

外層為action-conditioned Double Q：gamma=.9、target network、replay、每8個合法步驟回到elite；每個實際新可行PPA預設24次minibatch SGD。完整物理成本差提供reward：

```text
r = 10 * log(cost_before / cost_after)
y = r + gamma * Q_target(next_features[argmax Q_online(next_features)])
```

terminal不bootstrap；timeout／容量失敗不提供虛構PPA reward。epsilon-greedy的低entropy本身不證明學習有效，須看同預算搜尋、held-out診斷與凍結／均勻控制。

149維共用狀態／候選描述外，預設加入111維：75個tensor-boundary編輯、14個實際before-schedule service／criticality／queue／pressure，以及22個ordered Part×PE ownership的邏輯局部性編輯。graph residual分支零初始化，使F on/off的初始base Q完全相同。邏輯box需求不含完整tile、cache lifetime與排程，不能當PPA或reward。

額外Spatial整組整數DP對明示proxy求最佳份額，嚴守總quota及PE互斥；Temporal投影可重用序列stage的PE，以實際kernel／分池價格及當前量到的T/E產生候選。這些只是提案；完整Oracle仍決勝。有限family round-robin避免50架構候選擠掉其他變數。

`--no-boundary-features`、`--no-resource-projection`可分開消融；`--no-composites`保留完整原子變數。`--freeze-search-policy`、`--uniform-policy --freeze-search-policy --no-shared-warmstart`為因果控制。`--operators gemini5`是固定其他變數的五類LP語義對照，並非原版Gemini執行檔或預設搜尋域。

## 結果、限制及可重現性

預設EDP=Energy×Latency；也支援latency、energy與原balanced。`--max-area`、`--max-power`、`--max-latency-ms`是硬限制；推薦JSON只提供strict-feasible設計，沒有時 `state=null`，closest另列。

```powershell
.venv/Scripts/python.exe -X utf8 simplified_rapidchiplet/scripts/analyze_full_system.py --out output/r18_long
```

分析不新增PPA。主要輸出：`REPORT.md`、`summary.json`、各方法`recommendation.json`、`events.jsonl`、`learning.jsonl`、`worker/accounting.json`、`analysis/equal_budget.json`、`analysis/curves.csv`。保存來源／輸入hash、有效設定、環境、checkpoint、真正PPA數、SGD數、覆蓋與負面結果。已完成的durable PPA不重複計費。

同一專案的昂貴實驗由全域鎖序列化；另一個工作持鎖時，新工作會明確失敗，請查看正在執行的實驗。新版來源與結果凍結後續跑，避免混入中途修改。native calendar與Python fallback保留相同物理定義。

原v0.5的兩RNG有正面結果，但固定2顆對照由Random勝出；一次可行的Fusion配對也顯示Fusion On流量較高、EDP稍差。新強起點／F/P／凍結policy研究尚在執行，不能先宣稱新特徵有益或已打敗原版論文。請以[逐組數字](INMIN_GRAPH_EDIT_RESULTS.md)及完整mapping判讀。
