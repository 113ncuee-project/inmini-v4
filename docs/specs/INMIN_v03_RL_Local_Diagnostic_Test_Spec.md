# INMIN v0.3｜RL Mapping 搜尋成效與失敗歸因實驗：Codex 實作及本機操作規格

版本：2026-11-02／目標基線：使用者提供的 INMIN_v03_Source_20261102.zip  
交付對象：Codex（實作）、研究者（Windows 本機執行）  
文件性質：**待實作規格，不代表本文所有測試已執行。**

## 0. 任務目標與最高優先原則

請在現有 INMIN v0.3 原始碼基礎上，新增**完全在研究者自己的電腦上運行、不需要 LLM/API/token 持續參與**的實驗啟動器、檢查器、資料分析器，以及可中斷續跑的研究流程。使用者用 PowerShell／Windows Terminal 啟動後，程式必須獨立完成後續評估、將完整證據保存於本機，即使沒有繼續和 Codex 對話也能執行。Codex 不得替使用者直接開啟數小時的正式實驗。

主要科學問題：

1. **RL 是否真的在相同硬體上找到比共同最佳可行 Seed 更好的 Mapping？** Seed 本身的改善不可算進 RL。
2. **在可比較的搜尋時間與完整 PPA 評估預算下，RL 是否優於 Random、Greedy 與 Atomic RL？**
3. **Bottleneck 資訊是否影響候選選擇，而且選出的動作是否實際改善對應瓶頸？** 要分離「特徵有效」、「瓶頸導向候選生成有效」、「複合動作有效」三種來源。
4. **當 RL 無法改善時，哪一階段造成失敗？** 初始 Mapping、動作可達性、候選池、Policy 學習、Fusion、容量、通訊、PPA oracle、Timeout 或報告統計。

**負面結果必須完整保留**。不能因演算法沒有獲勝就改寫／丟棄實驗。不能以未選中的候選或 Timeout 當成已驗證較差的方案；不能把分析 PPA 當成實體晶片量測；不能宣稱有限次搜尋得到全域最佳解。

## 1. 已核對的 v0.3 現況：實作時先重用，不要重造

這些資訊來自本次 ZIP 的實際檔案與內附研究紀錄，並不等於作者獨立重跑 v0.3 完整研究：

| 項目 | 來源與現況 | 實作含意 |
|---|---|---|
| 主要 CLI | simplified_rapidchiplet/single_array_dse.py | 優先包裝此入口，不複製 RL／oracle |
| 現有 Arms | random、greedy、atomic_rl、bottleneck_rl | 請保留原始語意與行為紀錄 |
| v0.3 硬體 | 16 PE／Compute；1～16 Compute；50 個有方向完整矩形；每列 2 IO | 固定 Mapping 實驗須鎖定同一完整架構 ID |
| RL | hybrid/rl_search.py：linear softmax Actor–Critic，checkpoint／trace、候選機率、normalized entropy | 這些欄位應成為必查診斷證據 |
| 候選 | hybrid/single_array_proposals.py、target_bottleneck.py；含 A12、C7/C8 與複合動作 | 必須逐項統計提案、入選、執行、被拒、成功 |
| Seed | hybrid/single_array_seed_v3.py；capacity-screened segment beam | Seed 生成、完整 PPA 與 RL 改善要分帳 |
| 報告 | summary.json、joint_search.json、checkpoint.json、shortlists/*.json.gz、bottleneck_trace.json、fusion_coverage.json、architecture_coverage.json、recommendation.json | 新分析器應從現有資料重建；缺漏才補欄位 |
| 溯源 | effective_config.json、source_identity.json、seed_attempts.json、worker/accounting.json | 完整設定、版本、查詢與時間要核對 |
| 已有工作 | scripts/run_single_array_research_suite.py、scripts/audit_single_array_results.py、scripts/test_single_array_acceptance.py | 重用其鎖、快照、帳本與審計邏輯 |

ZIP 中的 v0.3 測試資料聲稱 113 項驗收通過；ResNet18 的 DP seed probe 已有可行候選：stripes 約 22.6611 ms／26.1231 mJ，DP 約 20.5577 ms／17.3965 mJ。**這是 DP 種子的收益，不是 RL 的收益。** 不能以 v0.2 歷史負面結果代替 v0.3 正式結果。

重要區別：目前 CLI 的 --bottleneck-features off 會遮蔽 Actor／Critic 所見的動態瓶頸特徵，但瓶頸導向候選生成器仍然啟用。**不能把這個開關寫成「完全關閉 Bottleneck」消融。**

v0.3 原 CLI 預設 --seed-seconds 7200、--seconds 7200、--eval-timeout 1800，對日常檢查偏長。**本次新增的本機實驗器必須使用獨立、明確、有上限的 profile，絕不在未確認時直接採用上述長時間預設。**

## 2. 實驗設計：先隔離 Mapping，再逐步放開 Fusion 與 Architecture

### 2.1 共同公平條件（硬性）

同一比較組必須相同：

- 完整 source snapshot/hash、Python 與依賴版本、hardware profile、計算精度及 oracle 參數。
- DNN 原始來源、FX/DAG workload hash、輸入大小、Batch=1、Seed 數值、PPA preference=balanced、相同 PPA hard limits（預設無額外限制）。
- **同一個完成且可行的起點狀態（state fingerprint）、相同 reference metrics、共同 seed archive**；各 arm 共用 seed 的真實 PPA 成本必須在方法報告中各計一次，但實際獨立 CPU 使用量不能重複宣稱。
- 共同完整 PPA 上限、搜尋牆鐘上限、單次 oracle timeout、候選族規則及合法性檢查。若演算法自身的原生動作空間不同，須明確註明「不同空間的方法比較」，不可宣稱完全控制變因。
- 不同模型各自正規化與比較，**不直接比較不同模型的原始 normalized cost**。
- Greedy lookahead 的所有 PPA、Random 的所有評估、Cache hit、Timeout、Compiler reject 均計帳；不能只計最終選中步驟。

公平比較同時呈現兩種尺度：(a) best-so-far cost 對實際 wall-clock；(b) best-so-far cost 對已完成 Full PPA 次數。兩種結果可能不同，必須並列。

### 2.2 三個互不混淆的搜尋空間

**M：Mapping-only（主要論文實驗）**

- 固定完整硬體 architecture ID：Compute 數、R×C、IO 配置、帶寬、SRAM、PE 數全都不變。
- 固定 Fusion 關閉（或固定完全相同的 fusion stack），只搜尋 layer group／partition、PE quota／placement、K/P 等合法 Mapping、片間資料來源／路徑與獨立 dataflow。
- 阻擋 A9 等會改變架構的動作；複合交易只要內含超出 M 空間的 edit 就必須整筆拒絕，不能偷偷修正成另一個空間。
- 若現有 CLI 不支援嚴格動作空間遮罩，Codex 需先新增可稽核的 --search-space mapping_only，不可只在結果端過濾。

**F：Mapping + Fusion（效果歸因）**

- 與 M 使用完全相同的固定硬體，但允許合法的 fusion depth／tile／cache／overlap 與必要的聯合動作。
- 同一組 Mapping 的 Fusion On/Off 要做**配對 counterfactual**，區分搜尋後 best-to-best 比較與固定 Mapping 的因果比較。
- 對 Fusion On/Off 分別記錄 DRAM／L1／L2／NoC／NoP bytes、重算 MAC、峰值 SRAM、排程成本與不可行原因。

**J：Joint Hardware + Mapping + Fusion（最後才做）**

- 使用 v0.3 全部 1～16 Compute 與 50 種 Layout，允許 A9；每個 layout 的 IO 數、DRAM 頻寬／容量、功耗／面積均隨布局正確重算。
- 這一組不能直接證明「Mapping 演算法比較好」，因為運算資源與 IO 同時改變。與 M、F 分開成章。
- 保留架構覆蓋率、實際測到的 n_compute／R×C、未探索原因。不能要求全部 50 種都做昂貴 PPA。

### 2.3 對照組與消融

| Arm | 主要要驗證什麼 | 注意 |
|---|---|---|
| Seed-only | 僅完成共同 seed，不進行搜尋 | 必須每組都有 |
| Random | 不學習時的可達品質 | 使用相同合法候選規則與完整評估預算 |
| Greedy | 局部改良與候選質量 | lookahead 查詢全部計費 |
| Atomic RL | 一次單維改動的學習能力 | 與 composite 空間不同，需註記 |
| Bottleneck RL | 目前完整方法 | v0.3 原生 |
| Bottleneck RL／features off | 只關閉 Actor／Critic 的瓶頸特徵 | **候選生成器仍有 guidance** |
| Composite RL／generic proposals（新增） | 分離複合動作與瓶頸導向提案 | 須實作 generator ablation，保留同一基礎動作合法性 |

若要嚴格驗證瓶頸資訊，另建 2×2 消融（初期先用 ResNet18，不做全模型笛卡兒積）：

| 候選生成 | Actor 瓶頸特徵 Off | Actor 瓶頸特徵 On |
|---|---|---|
| Generic proposals | G0F0 | G0F1 |
| Guided proposals | G1F0 | G1F1（完整版） |

生成器開關必須清楚記錄是否**改變候選集合本身**；不能把不同候選池的結果誤解為單純 Policy features 的因果效果。另增固定 state＋固定候選集的一步決策 replay：只改 feature mask、不調用 PPA，比較選擇機率、排名和 entropy，作為最乾淨的直接檢查。

## 3. 分階段實驗：少量、代表性、能逐步定位

下表為**建議的本機預設上限**，不是已存在 CLI 旗標；Codex 可調整具體數字，但須在 manifest 留下有效值。每個 arm 的總牆鐘（含按公平原則分攤的 seed）在 diagnostic profile 預設不超過 30 分鐘；更長的 research profile 由使用者明確選取。

| Phase | 模型／硬體 | Arm 與目的 | 建議 PPA 上限 | 時間／停止條件 |
|---|---|---|---|---|
| P0：smoke | Toy，固定小架構 | 4 原生 arms；檢查 CLI、trace、resume、query 帳本 | 12～16／arm | 小於約 5～10 分鐘的目標；失敗即停止 |
| P1：Mapping 基本能力 | ResNet18，預檢後選 1 個**已證實可行**的固定 ≥2 Compute layout | Seed-only、Random、Greedy、Atomic RL、Bottleneck RL | 24～32／arm | 預設 seed ≤300s、search ≤1500s／arm；單 oracle ≤300s |
| P2：瓶頸特徵歸因 | 同一 ResNet18 固定硬體 | G1F0 vs G1F1；若成本允許加 G0F0/G0F1 | 24～32／arm | 僅啟用一組可重現的消融，避免一次全跑 |
| P3：Fusion 對照 | ResNet18，硬體固定 | Fusion Off/On；少量完全固定 Mapping paired tests | 6～12 組已選 state | 報告每個 stack 的 bytes、重算、容量 |
| P4：泛化測試 | ResNet50 v1.5、MobileNetV2；若後續需要再加 VGG16 | 只選前階段最有意義的 2～3 arms／模型 | 16～32／arm | 比較不同 DAG／記憶體瓶頸；不開啟全組合 |
| P5：確認結果 | 最重要模型、最佳 arm 與最強對照 | Seeds 7、17、37，重複不同起點與 policy RNG | 依 P1/P4 | 報告每個 seed、均值/離散度；不能僅 Seed7 宣稱普遍有效 |
| P6：Joint 擴充（可選） | 1～16 Compute 全域 | 在已驗證 Mapping/Fusion 能運作後再開 | 使用者指定 | 另列架構變更的貢獻與 PPA 成本 |

Phase P1 的固定架構不是任選後假設可行：先以相同 oracle 驗證候選（例如 2 Compute 的某個布局或 4 Compute 的 2×2），選**一個**可行且能產生跨 chiplet 傳輸的基線。如果該架構在限定時間內不可行，必須記錄並回到 P0／容量診斷，不能悄悄改用不同架構完成各 arm。

本機執行器要提供三種 profile：

- smoke：只做快速邏輯驗證，不得作研究結論。
- diagnostic（預設）：每 arm 約 30 分鐘以內的總預算，少數代表模型；可按階段啟動。
- research：明確指定更長 seed/search/oracle 上限及多 seed；需要使用者主動確認才啟動。

**不許在同一台電腦默認平行跑多個大型 Full PPA case**，以免 CPU/記憶體競爭讓 wall time 不可比較。預設 sequential。每個 case 自動保存進度；所有測試都可以停止／恢復／跳過已完成 case。

## 4. 必須保存的診斷資料與因果鏈

資料設計要回答「不改善時為何失敗」，不只寫最終 cost。建議事件採追加式 JSONL 或穩定 checkpoint，另匯出 CSV 與文字 Markdown 報告；每個事件有 schema_version、run_id、wall_time_monotonic、UTC、model、arm、seed、source identity、stage、event_id。

### 4.1 Run 與 Seed 資訊

- git/source SHA256、全部使用的 .py／硬體模型 hash、Python/PyTorch/numpy/torchvision 版本、OS、CPU 核心／RAM、可取得時的 thread 設定。
- engine v03、state schema3、feature schema、模型 DAG hash、batch、input size、固定硬體 ID、完整布局／IO 配置、Fusion 開關、PPA objective/limits。
- common seed：stripes/DP 的提案來源、DP 耗時、capacity pruning、候選去重、完整 PPA 是否完成、不可行原因、seed 評估耗時、最終最佳 seed hash、metrics、cost、reference hash。
- 真正獨立的 seed generator 耗時、common seed Full PPA 次數、search-only 耗時、每 arm 的 **charged seed+search** wall、整個套件的實際 unique backend wall 都分開。

### 4.2 每一步 Action 決策（不可缺欄）

| 類別 | 必存欄位／問題 |
|---|---|
| Before | before state fingerprint、當前 PPA／cost、current best、是否回到 bootstrap seed |
| Observation | 24 維 state features、283 維 chosen-candidate features（如對應 v0.3 schema）、每個數值的意義／正規化／來源及 evidence quality |
| 瓶頸 | critical-path layer/event、slack、MAC/L1/L2/VPU service、DMA／DRAM／NoC／NoP queue/busy、最高 link 利用率、peak live SRAM/capacity 與 pool id |
| Candidate pool | 全部已建立候選數、family、legal/rejected/filtered 數與原因；可恢復 shortlist gzip 路徑、hash、候選完整 ordered edits |
| 選擇 | selected family/targets、完整 action plan、候選概率、behavior probability、normalized entropy、提案來源（generic/memory-guided/critical-window） |
| 執行 | shadow transaction是否合法、修改前後完整 state、各 edit 是否真的生效；是否超出 Mapping-only 的遮罩 |
| PPA | oracle request/cache hit/full completed/timeout/compiler reject/export fail、單次提案/編譯/排程/評估/輸出耗時 |
| After | after state hash、latency/energy/area/power、cost、是否滿足限制、reward、advantage、policy_update、best-so-far 改善 |
| 物理效果 | DRAM R/W、L1/L2 services、NoC/NoP hop-bytes、MAC/recompute、各 bottleneck 前後量值、資料流與 Fusion stacks |

原本已保存的 trace 直接讀取，不要重複儲存數百 MB calendar；巨量逐 event calendar 允許以可稽核的壓縮檔和索引保存，摘要必須說明 truncation 規則，不能截掉重要錯誤來源。

### 4.3 另外建立四種可診斷指標

1. **搜尋品質曲線**：best-feasible cost vs completed Full PPA、best-feasible cost vs charged wall、best-feasible cost vs search-only wall；橫軸含 0=共同最佳 Seed。
2. **可達性／效率漏斗**：proposed → structurally legal → shortlist → selected → oracle attempted → completed PPA → feasible → positive reward → exceeds global seed → exceeds global best。依 action family、target layer、chiplet count、Fusion depth 分解。
3. **Policy 是否真的學習**：probability/entropy vs step、選擇家族分布、policy 更新數、對同一 state 的固定候選清單進行 features on/off 離線決策比較；不能用「有更新」就宣稱策略有效。
4. **診斷準確度**：選擇動作前的 target bottleneck、動作企圖改善的量、實際改善／惡化、是否成為新 bottleneck；依 target type 統計命中率，明註 whole-flow reservation 的等待分攤不是精確硬體因果量。

對 5～10 個具代表性的 chosen actions（成功、失敗、重算增加、timeout 各涵蓋），在**同一個 before state**做少量配對 counterfactual：原方案與候選／對照方案由相同 Full PPA evaluator 評估，保存逐資源差分；若超過預算跳過並記錄原因，不可偽造數字。

### 4.4 PPA 帳本驗證

- 完整 PPA 數必須與 oracle worker ledger 及 runner report 相符；cache hit 不增加新的完整評估。
- Timeout / compiler reject 無合法 PPA，不給虛構 reward、不進 best archive、不更新 Policy。
- P×T≈E（明列容許數值誤差）、能源分項總和、MAC 守恆與有效 SRAM peak ≤ capacity，或明確記錄 overflow。
- 所有 Best 必有完整已完成 PPA、state hash、完整 Mapping 與能重放的來源／設定；best depth 是最佳狀態的深度，**max evaluated depth 不能誤寫成 best depth**。
- 四方法未完成相同查詢數，報告只能如實比較「相同上限下實際完成數」，不能單獨宣稱 query efficiency 較佳。
- 同模型的不同實驗如果 source/oracle/profile 不一致，將標記 incomparable，拒絕合併成同一張勝負表。

## 5. 本機測試工具：Codex 必須實作的 CLI／使用者體驗

**建議檔名**（新檔，不取代舊入口）：

- simplified_rapidchiplet/scripts/run_local_rl_diagnostics.py
- simplified_rapidchiplet/scripts/analyze_local_rl_diagnostics.py
- simplified_rapidchiplet/scripts/check_local_rl_diagnostics.py
- scripts/run_inmin_rl_experiments.ps1（可選 Windows 啟動捷徑）
- docs/LOCAL_RL_EXPERIMENTS_README.md
- tests/test_local_rl_diagnostics.py

要求只依賴目前專案 Python 環境與標準函式庫，分析與操作不得使用 OpenAI/LLM/雲端推論 API，不得需要 Codex agent 保持連線、反覆輪詢或輸出訊息。允許本機 CPU 評估程序、PyTorch 模型解析與必要的既有套件。不下載 pretrained weights。

### 5.1 使用者功能／CLI 契約

| 子命令 | 行為 |
|---|---|
| help / list | 顯示可用 profiles、models、arms、預估最壞預算、限制及 CLI 範例 |
| doctor | 只檢查 Python/依賴、硬碟空間、來源 hash、是否已有同目錄任務、CPU/RAM、Windows 路徑；不跑 DSE |
| dry-run | 印出每個 case 的精確子程序命令、階段、查詢與最長時間，不啟動 PPA |
| run | 使用者明確指定 profile／phase，依序執行；每項輸出 log、checkpoint、manifest |
| status | 從已保存狀態顯示進度、已完成 Full PPA、最近改善、剩餘上限、最後錯誤；**不調用 oracle／模型** |
| stop | 對本 runner 所屬的工作程序發出安全的停止訊號；等待 checkpoint；不殺其他研究程序 |
| resume | 相同 manifest/source/config 接續，中斷／重開電腦可用；不重複完成的 Full PPA |
| summarize | 從既有 JSON/JSONL 整理 Markdown、CSV、診斷結果；**不調用 oracle／RL** |
| audit | 核對上述帳本／物理守恆／來源一致性；不修改 checkpoint |
| export | 產生一個含 report、有效設定、必要 traces／hash 的可攜 zip；不包含不必要的大型中間快取 |

使用者只要執行一次 run，就可以離開 Codex 對話讓本機程序自行跑。安裝前置與執行方式要有 **Windows PowerShell 可複製命令**，在 README 裡從 repo 根目錄算路徑，不能依賴寫死的 C:\Users\...。

### 5.2 使用例子：必須在文件中能直接複製

新增 runner 的**提議介面**（Codex 實作後才可視為真實命令）：

~~~powershell
& '.venv/Scripts/python.exe' -X utf8 simplified_rapidchiplet/scripts/run_local_rl_diagnostics.py doctor
& '.venv/Scripts/python.exe' -X utf8 simplified_rapidchiplet/scripts/run_local_rl_diagnostics.py dry-run --profile diagnostic --phase p1
& '.venv/Scripts/python.exe' -X utf8 simplified_rapidchiplet/scripts/run_local_rl_diagnostics.py run --profile diagnostic --phase p1 --out output/local_rl_v03_r18
& '.venv/Scripts/python.exe' -X utf8 simplified_rapidchiplet/scripts/run_local_rl_diagnostics.py status --out output/local_rl_v03_r18
& '.venv/Scripts/python.exe' -X utf8 simplified_rapidchiplet/scripts/run_local_rl_diagnostics.py resume --out output/local_rl_v03_r18
& '.venv/Scripts/python.exe' -X utf8 simplified_rapidchiplet/scripts/run_local_rl_diagnostics.py summarize --out output/local_rl_v03_r18
& '.venv/Scripts/python.exe' -X utf8 simplified_rapidchiplet/scripts/run_local_rl_diagnostics.py audit --out output/local_rl_v03_r18
~~~

**現在 v0.3 已存在且可作低層測試的 CLI 範例**（執行時間仍由實際機器決定）：

~~~powershell
& '.venv/Scripts/python.exe' -X utf8 simplified_rapidchiplet/single_array_dse.py --engine v03 --model resnet18 --seed 7 --preference balanced --fusion on --arms random,greedy,atomic_rl,bottleneck_rl --queries 24 --seed-seconds 300 --seed-queries 4 --seed-dp-seconds 60 --seconds 1500 --eval-timeout 300 --shortlist 64 --out output/manual_r18_v03
~~~

注意：上面是**現有全域搜尋**的示例，還不是嚴格 Mapping-only，因為目前需要新增動作遮罩與固定硬體功能。Codex 實作完成之前，不得把此指令的結果命名為 Mapping-only 實驗。恢復現有 CLI 時需用同參數加 --resume。

### 5.3 停止、斷線、重啟與資料保護

- 預設單程序／單大型 case 順序運行，不要求 Codex/ChatGPT 持續對話；stdout/stderr 一律寫到本機檔案，終端只顯示有限摘要。
- 實驗目錄有 OS lock，禁止重複啟動相同 run；已有工作進行中只可 status，不可啟動第二個 runner。
- 每個完成步驟 atomic checkpoint，外部 worker timeout 應記錄且清理**只屬於該 case**的子程序，不得誤殺其他使用者程序。
- Ctrl+C／使用者 stop 應盡可能完成目前 checkpoint，標記 interrupted；若在完整 oracle 中途終止，該評估只能記 interrupted/unknown，不可算完成。
- resume 必須核對 source hash、profile、完整有效設定、model/arch、seed、objective；不同就拒絕並要求新 output。**既有報告和正式實驗不可覆寫**。
- 可選 Windows 暫時保持喚醒模式，但提示使用者仍須留意闔蓋睡眠、Windows 更新與電源。不能擅自修改永久電源設定。
- 避免巨型 JSON 記憶體／匯出瓶頸：沿用 compact evidence，詳細數據壓縮按需載入；報告分析不重新跑 oracle。
- 如長時間進程失敗，本機至少保留最後完成的 trace、stderr、最後成功 case、resume 命令與 termination reason。

## 6. 機器可執行的結果診斷規則：負面結果也要能定位

每個 run 結束後，自動產生 diagnostics.md，至少判讀以下情況；只能寫「證據支持／尚待驗證」，不要直接自動下根因定論。

| 觀察到的現象 | 優先懷疑 | 下一個最小測試 |
|---|---|---|
| DP 無可行 seed／大量容量 reject | DP 容量近似、L1/L2 分池、缺少 C-panel／spilling | 固定 1 個候選，逐 pool peak/lifetime 與合法 tile 修復 |
| Seed 明顯優於 stripes，但 RL best 不再改善 | RL 動作可達性、候選覆蓋、Policy 學習不足 | 比較 seed-only、greedy、同 state 候選漏斗 |
| Greedy 改善，RL 不改善 | 候選質量尚可；Policy 抽樣與訓練訊號可能不足 | 比對同 shortlist 的機率／entropy／正 reward 率 |
| Random 和 RL 相近 | 無法歸因於 Policy 學習 | Features on/off、固定候選 replay、多 RNG seeds |
| Features on/off 幾乎無差／entropy 長期接近 1 | Feature 無區辨力、正規化飽和、更新太少、候選池不變 | 固定 state 不同 target 的 feature 差及 probability |
| 多數評估超時，完成 Full PPA 很少 | Oracle 性能、複雜狀態、query/wall 失衡 | 拆 compile/schedule/evaluate/export 耗時，profiling |
| Cache hit 高、unique state 少 | 搜尋停滯、重複動作、狀態去重／episode restart | 追查 state fingerprint、候選反向交易與覆蓋率 |
| 有正 reward 但 best 沒進步 | 從較差 current state 回復，不代表超越 Seed | 同列 current-cost delta 和 global-best delta |
| Fusion 後 DRAM／NoC 變高、重算 MAC 上升 | Tile/halo/recompute、cache policy、資料路徑 | 同 Mapping 的 Off/On 配對逐 tensor 分解 |
| Fusion Off 容量失敗但 On 可行 | 存活時間、O-L2 pool 限制、spill/tiling 空間 | Fixed Mapping、固定 buffer 容量、修復可行性 |
| A9 變更多但 PPA 不改善 | IO/DRAM 隨布局放大、D2D 代價、過度資源 | 固定架構 Mapping-only vs Joint 分開比較 |
| Critical bottleneck 與被選 target 不一致 | 診斷／proposal target mapping／掩碼 | 同 state 預期 action→target 受控測試 |
| Reference／seed hash／來源不同 | 不公平比較或重現性錯誤 | 停止合併報告，重新建立共同 seed manifest |

診斷報告最後必須列：

1. **可確定的事實**（數字、來源、影響範圍）。
2. **最可能的 1～3 個原因**（每個附對應證據與證據品質）。
3. **下一個最省時間的鑑別實驗**（固定哪些變因、需要多少 PPA、預期觀測量）。
4. **可能的修改位置**（實際程式檔案／函式，僅在已核對時列出）。
5. 目前證據不能推論的事情。

## 7. 實驗輸出目錄（建議）

~~~text
output/local_rl_v03_r18/
  manifest.json
  effective_config.json
  source_snapshot_manifest.json
  suite_status.json
  logs/
  seed/
  cases/<model>/<space>/<fusion>/<arm>/<rng_seed>/
    checkpoint.json
    joint_search.json
    shortlists/
    oracle_accounting.json
    events.jsonl
    best_mapping.json
  reports/
    summary.md
    compare_methods.csv
    best_vs_wall.csv
    best_vs_full_ppa.csv
    action_funnel.csv
    policy_learning.csv
    bottleneck_effects.csv
    failure_reasons.csv
    fusion_pairs.csv
    diagnostics.md
    audit.json
~~~

若已有原生路徑與資料結構，使用索引或 manifest 引用原始檔即可，不要為符合上表而不必要地複製大型工作檔。保留單一事實來源；任何 CSV 都要可由完成的 JSON/JSONL 重建。

## 8. Codex 實作順序與最低驗收

### Step A：先作既有程式核對與設計盤點

1. 讀取 v0.3 上述程式、113 項測試、source identity 與現存研究腳本。
2. 實際核對現在能否嚴格鎖住架構與 Fusion；若不能，先補「合法的搜尋空間遮罩／限制」，並有拒絕證據。
3. 確認現有 trace 的 schema/key、單位、wall time、best metrics 是否足以建曲線；只新增必要資料，不改 oracle 計算公式。
4. 不重新命名舊 v0.2/v0.3 實驗結果，不修改舊 frozen snapshot。

### Step B：本機 Runner 與資料分析

5. 新增 doctor/list/dry-run/run/status/stop/resume/summarize/audit/export。
6. 建立不可變 run manifest、分階段 runner、單例鎖、child PID 掌控、時間與 Full PPA 帳本。
7. 實作從現有 trace 生成兩種 best-so-far 曲線、行為漏斗、entropy、bottleneck before/after 與負面原因報告。
8. 實作完整的 Features on/off 與 Generic/Guided generator ablation 標記；所有候選生成差異明確呈現。

### Step C：只跑小型驗收，不啟動正式長跑

9. Unit：查詢守恆、reward/energy consistency、mapping-only 不可改 chiplet/fusion、失敗不能更新 Policy、同 reference、同源 hash、NoC/IO 面積變化、structured report、CSV/JSON units。
10. Smoke：Toy 各方法少量 PPA；確認 no API calls，離開對話後可繼續、本機 log 可查看。
11. Resume：強制中斷並接續 toy，不重算已完成 PPA，結果 hash 與未中斷測試可比較。
12. Status/summarize/audit：檢查**無任何 oracle/模型執行**。測試缺損 log、部分 timeout、invalid action、0 feasible、只有 1 個 arm 等情況，仍能寫出有用報告。
13. 本機全流程 smoke 成功後，**停止**。只交付工具、文件和 smoke 結果，讓使用者自行決定何時啟動 P1～P6。不要自動提交長跑、不要重啟暫停的排程。

### 最低驗收判準

- Windows PowerShell 在 repo 根目錄依 README 命令可用，無需持續詢問 Codex。
- 所有長跑都能中斷續跑；可無損保留失敗與負面結果。
- 能在固定硬體上真正禁用架構變更，並清楚區分 Mapping、Fusion、Joint 結論。
- 每個 RL 改善都能從完成的 Full PPA 與共同 Seed 比較得到；沒有完成評估就不能顯示勝出。
- 逐步記錄 bottleneck → proposed candidates → selected action → before/after PPA → reward/advantage。
- 同時得到 best-vs-time 與 best-vs-Full-PPA；兩種 budget、所有拒絕及 timeout 都可核對。
- 能自動在結果不好時列出下一個最小鑑別測試，不只是「RL 可能需要更多訓練」。
- 只有使用者明確要求才執行正式 P1～P6；此次 Codex 任務範圍為**實作工具、測試小型功能與交付使用指令**。

## 9. 最終交付給研究者

請 Codex 交付：

1. 新增／修改的檔案清單、每個檔案的功能。
2. Windows 本機安裝／啟動／查看進度／停止／續跑／分析的可複製命令。
3. 實際跑過的**小型驗收**及失敗項目（如有），不要產生虛構研究結果。
4. 已核對的輸出 schema 與一份 toy smoke 的 report 範例。
5. P1～P6 各自的預估最壞時間、完整 PPA 上限與模型清單；超出 30 分鐘的研究 profile 應明列並由使用者選擇。
6. 已知模型限制：分析式 PPA 非 silicon 測量、whole-flow reservation 保守、沒有完成 C-panel／silicon/SRAM macro 校準時需警示。

**最重要的研究結論格式**：

> 在 [模型／固定硬體／Fusion／來源]、相同 [Seed／PPA 參考與有效約束]、[時間／查詢] 預算下，Bottleneck RL 相對共同最佳 Seed 改善 X%，相對最強可比對照改善 Y%。完整 PPA 次數、Timeout、Cache hit、Policy 更新及主要瓶頸變化分別為 ...。若未勝出，證據最支持的失敗環節是 ...，下一個鑑別測試是 ...。本結論僅適用於本次分析模型與實驗域。

以上每一個 X、Y 與數值只能由本機輸出的完成驗證資料填入，禁止預先寫入研究結論。
