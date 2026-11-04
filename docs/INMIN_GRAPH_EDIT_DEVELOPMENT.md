> 歷史版本記錄：以下為v0.5／v0.6當時的來源、實驗與協定。現行程式及新結果請見 [成本拆分Q報告](INMIN_COST_VALUE_RESULTS.md) 與 [操作指南](INMIN_FULL_SYSTEM_GUIDE.md)。

# INMIN graph-edit RL：設計、歸因及事前實驗協定

## 目前實際版本 — 2026-11-04上午

程式／研究均使用**study_v8**，來源`5f2fdce00493d95868b4b0837c03c283d590a0c5a274a34de6a6bc3ec37766de`。173項回歸、完整CLI（含起點強化及停止續跑）與GUI 8次完整PPA通過。研究已於07:52開始先做強DP／inner共同起點，再執行原訂F/P與因果對照。下文較早時間的段落是事前方案的歷史修訂，實際凍結protocol／source manifest與本段為準；沒有把尚未完成的消融填成勝出結果。

F目前為111維：75個tensor-boundary編輯、14個實際before-schedule context，以及22個新增ordered Part×CG ownership局部性特徵；總輸入260維，149維base與零初始化graph分支保持F-only對照。新特徵以實際shard順序對應PE、required-input boxes精確交集、相同destination-die／region去重、真實路由incidence及MAC gateway邏輯需求計算。部分重疊、執行tile／generation／cache lifetime／queue仍由完整Oracle處理，絕不當PPA或reward。

加入這部分的依據是實際Toy反例：相同die histogram、PE集合與partition，交換consumer的ordered PE即可改變NoP流量，但舊graph sketch完全相同。新編碼能區分，向量化與scalar交集／流量守恆、group邊界及after-label隔離測試均通過。真實R18 64個候選F-only hashes完全相同；工程提案＋特徵耗時約0.71–2.40秒，0新DNN PPA／0SGD，不視為搜尋速度勝出證據。

一般CLI／GUI已整合wider DP起點，不依賴本機既有夜間輸出；共同PPA會以真實payload計數，reference與seed相同時只計1次。關閉架構篩選也會保存零次查詢帳本。新研究logging的Path序列化與核心整合失敗均在DNN研究前修正，保留先前版本與失敗日誌。

記錄時間：2026-11-04 00:35 Asia/Taipei。這是尚待 DNN 消融驗證的研發版本，不把機制名稱當成已證實的創新或效益。正在執行的 v0.5 正式研究維持來源 `336f979b54946ba967d3c16dcf56b9aa7fca3bb43dc6aa304b440540d3858933`；隔離副本位於 `tmp/inmin_v06_graph/app`，不修改已凍結研究。

## 已觀察到的問題與改進方向

v0.5 RNG7 的完整空間 RL 完成141個新搜尋 full PPA、共同階段52個，最佳 EDP=0.06109825。相對共同篩選後的0.07957726再改善23.2215%；同組 SA 的最佳0.07943955。Random及第二RNG未完成，不能先寫成穩定優於所有方法。最佳架構8顆Compute／2×4，Latency3.620577ms、Energy12.145677mJ，但面積106.221720mm²、平均Power3.354624W高於原2×1方案。

多個有效動作合併了真正的DAG分組，降低DRAM materialization與NoP流量。原148維狀態／動作描述主要是全域平均及目標層位置，沒有直接表示「本次移除了哪個tensor的哪些consumer cut」。這是新特徵的依據，不是預先假定所有合併必然有益。

## 候選技術：以DAG修改為條件的多步Q學習

1. **Tensor-boundary edit encoding**：明列每個tensor→consumer邊的跨group讀取、每個producer的一次materialization寫入、殘差fan-out、本地forwarding、partition差異、PE／die重疊、FD邏輯負載、group邏輯活躍資料量，以及修改前／後／差。共75個新增特徵。群組名稱改變不會產生假變化；有local與remote consumer的殘差保留兩種邊。
2. **有對照初始化的graph residual Q分支**：原149維基底Q網路保留，新增75→32→2的分支，最後一層初始為0。因此feature on/off在相同seed的初始Q／reward-head輸出完全相同；差異需由實際PPA標籤訓練後形成。主體仍為gamma=.9的Double Q、target network與replay，reward仍為 `10*log(cost_before/cost_after)`。
3. **Conserving integer group projection**：新增整組Spatial Group配額候選。對每層枚舉所有有非空整數partition的PE數，以DP選一個配額／partition，嚴守每層≥1、總配額≤實體PE、PE互斥。DP只對明示的理想shard service＋logical staging proxy求精確最小值；Batch1採stage service之和，沒有虛構穩態pipeline吞吐。保留idle PE，保留獨立FD、dataflow及retention。原A/C、global mapping、Fusion及50架構候選都保留。

這些邏輯量不等於實際流量、scheduled buffer peak或Critical Path。不能用proxy判定PPA改善，不能用它們製造reward；完整compiler／communication／calendar／capacity／energy oracle仍決定所有結果。

## 完整空間與對照空間

預設仍為完整schema3：Compute1–16的50種矩形、16PE／Compute、共享L2、Temporal／Spatial分組、B/K/H/W整數切分、任意合法PE配額與排序、DRAM FD、dataflow、retention、Fusion／tile／cache。Batch／Microbatch=1沿用原v3已實作的物理契約。proposal有限且可旋轉抽樣，不宣稱列舉所有Cartesian組合或保證全域最佳。

`--no-boundary-features`與`--no-resource-projection`分離消融。只改feature不改候選或RNG抽樣；已在Toy和R18靜態候選驗證。

`--operators gemini5`僅供明記的LP比較臂：OP1 partition、OP2同層順序swap、OP3同Spatial Group跨層swap、OP4單PE轉移＋兩層repartition、OP5外部FD。該對照固定架構、group、Fusion、retention及dataflow；它不是預設搜尋系統，也不是Gemini原始執行檔或完整paper再現。

## 比較與防止弱基線

- Wider Tangram-style基線：所有連續segment長度、beam16，分別保留／關閉保守whole-output容量剪枝；原始DP mapping和明示q8／balanced／spread／interleave／auto inner refinement皆接受完整PPA。輸出DP是否完成、實際評估數、不可行與超時。這仍是同硬體INMIN adapter，沒有執行原版Tangram BSD／ALLO。
- Same-domain SA／Random：同起點、完整相同變數與proposal generator。另測SA使用shared-start尺度的溫度與每horizon=8個接受狀態回到elite，避免只比較未調整的SA。
- Gemini-style LP5 SA：額外顯示五類操作的結果；不能將此限制空間臂與完整空間臂的差異全部算成RL策略效益。
- 不把architecture screening、seed DP、inner refinement的收益歸給RL；分別列出原seed→共同強起點→搜尋後的數字。

## 2026-11-04事前確認協定

待原夜間兩RNG研究及固定硬體／inner controls／跨模型檢查結束後，才啟動新DNN計時研究，避免重疊CPU負載干擾wall time。

1. 在不可修改的engine snapshot上重跑完整回歸、實際stop/resume／50布局／shared-history／LP5 workflow及GUI Toy。
2. ResNet18 RNG7為development：F0P0、F1P0、F0P1、F1P1四組RL，每組共同52＋最多64個新full PPA、900秒search、單次240秒。F=新增特徵，P=整數資源投影。
3. 只用RNG7的已測EDP選development最佳設定；選擇規則與所有落敗設定都保存。對最佳設定補同域SA／Random與LP5 SA。
4. RNG17為confirmation：重新初始化policy，對選定設定做RL／SA／Random，並保留plain RL對照；相同PPA上限、時間及起點。若最佳設定就是plain，不重複相同實驗。執行前00:53補充兩項核心因果控制：只凍結shared-warmstart Q（不做任何search SGD），以及完全不訓練／均勻選動作但維持相同RL horizon與接受規則。兩者都沿用選定的proposal設定，控制「好的候選」和「搜尋軌跡」的混淆。
5. 額外wider DP基線獨立計帳，不把它的標籤偷渡給RL。比較同完成full-query floor、同wall-time floor、Latency／Energy／Area／Power及traffic。結果不佳則記錄失敗歸因，沒有證據的元件不設為勝出技術。

共同歷史重用只讀已完成的原common／50布局screen，不讀任何RL／SA／Random搜尋結果或policy。必須核對seed輸入hash、所有物理oracle／compiler／硬體／route／native binary來源及數值環境一致；修改controller及seeder可明記，修改物理模型就拒絕重用。原52個full PPA與原共同wall time仍計入每臂，新增common實體查詢為0。新控制器重新計算自己的特徵，不沿用原148維向量。

實驗凍結：新增因果控制後使用 `output/inmin_v06_20261104/study_v2/engine`，來源 `4a46319fdd330f90ea674d8c01e0e1e12b0b10bbdecb7ad58600bc82c6182fb3`。先前study_v1在QA／研究執行前即被替換，新增研究PPA=0，engine／protocol保留。v2的凍結orchestrate.py、protocol.json、source_files是實際執行依據，執行後不改寫。

### 執行前修訂 — 2026-11-04 01:55

上述v2同樣在QA／研究開始前即保留並替換（新增研究PPA=0）。實際待執行版本為 `study_v3`，來源 `b8bccf07fb1cfcb60b4ca8da403654133dd65b776eda2872c4c1e5438acd4a85`。發現原RL最終合併為單一Temporal Group後，**先執行wider DP及inner refinement，將其完整PPA結果作為共享強起點，再做RL對照**。這防止把原本max-group bound造成的起點缺陷包裝成RL成效。

實際共同成本改為原2個common＋50布局＋新wider-DP／inner完整PPA，並計入整段新baseline的parent wall time。新增可行與不可行完整觀察都保留及計費；不可行／超時沒有reward。未完成的timeout仍可在後續搜尋重試。每方法再增加最多64個新full／900秒，總query ceiling依共享實際完成數動態計算。兩個RNG使用完全相同的共同物理觀察，不轉移任何控制組／搜尋結果或policy。

inner宣告範圍擴為q8的P／K方向及q16／32 balanced，所有候選仍要完整PPA；raw完整quota方案也保留，依預算完成則記錄，未完成不假稱最佳。DP若第一次時限到但prefix仍有進展，可從checkpoint最多再延長兩次；不將partial prefix當作全模型方案。

P額外加入Temporal成本投影：每一serial stage的quota≤實體PE數，可以重用同一批PE，不套Spatial總和限制。以實際fitting fragment、kernel及命名SRAM價格建立局部T/E與atomic-staging burst proxy，按目前完整評估的全域T/E做一階EDP投影。F新增14個實際before-schedule service／criticality／queue／pressure與明示quota edit交互特徵，因此新增總89維、完整238維；149維基底仍保持相同初始化。所有proxy依舊不當成PPA或reward。

同時修正schema3 A8內部tensor編輯不可移動consumer另一個外部輸入的問題，並遮罩目前不materialize的外部FD動作；邊界啟用後仍可選任何合法FD。原schema1／2歷史語義保留。這些controller／proposal修改與未修改的實際compiler／hardware／communication／calendar／native模型分開記來源；不把來源identifier變更冒充物理模型升級。

## 與文獻的關係

### 執行前修訂 — 2026-11-04 03:04

實際等候執行的是 `study_v5`，來源 `fcc0cc5d38fa5d22e36c21e48a4874eaf9b0144a1e873e145c838d9dd51cee93`。v4同樣只準備而未執行研究PPA，保留凍結來源及NOT_EXECUTED紀錄。v4已將共同強起點整合到一般CLI／GUI：預設wide初始化、最多48個額外起點完整查詢、1200秒起點評估、每段600秒DP並可checkpoint延長；總預算會保留搜尋查詢。研究共用既有strong archive時不重跑，所有已支付起點查詢與時間仍計入。實際stop/resume、完整回歸與GUI驗收仍必須先通過，才進行新DNN研究。

v5補上Fusion tile的可達性：原減半／加倍不能保證到達奇數尺寸，現在另外具有每軸±1及合法正整數H/W直接跳轉。交易測試確認(4,6)可合法到達(5,6)等非平方尺寸。非Fusion內部執行tile仍交由實體分池容量fitting決定，沒有把繼承的C-panel／microtile欄位宣稱為新硬體模式。保留原schema3完整有效變數，不將有限提案誤稱為所有Cartesian方案已窮舉。

|來源|採用或對照的內容|不能歸為本系統原創的內容|
|---|---|---|
|[Tangram, ASPLOS2019](https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/tangram.asplos19.pdf)|DAG分段／DP／beam的起點方法|分組與pipeline思想本身|
|[Gemini, HPCA2024](https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/gemini.hpca24.pdf)|Part／ordered CG／FD與五種LP operators、架構外層探索|五種operators及架構／mapping聯合探索本身|
|[Double DQN](https://arxiv.org/abs/1509.06461)|online argmax＋target network的Bellman目標|Double Q／replay本身|
|[ConfuciuX](https://arxiv.org/abs/2009.02010)|用cost model實際回饋進行RL資源搜尋的先例|「把RL用於加速器DSE」本身|
|[DeFiNES](https://arxiv.org/abs/2212.05344)|深度優先tile／cache／重算的成本建模參考|深度優先排程及解析模型本身|
|[CORE, 2025預印本](https://arxiv.org/abs/2506.03474)|依參數相依性解碼、以完整模擬回饋的critic-free one-step RL；僅作失敗時的替代方向，未實作或拿其數字作本機基線|單步RL、constraint-aware decoder或batch-relative policy更新本身|
|[Multi-Agent RL for DNN Mapping, 2025預印本](https://arxiv.org/abs/2507.16249)|依參數關聯分群、多agent改善sample efficiency的文獻；本輪沒有實作其演算法|「多agent／分層即可改善sample效率」不能當成我們已驗證的原創主張|

本輪要驗證的具體差異是：**張量邊界修改的可歸因表示＋真實成本訓練的多步動作選擇＋資源守恆的整組候選**。須以消融證明各自增益；目前沒有做完整novelty檢索，也不能宣稱已優於原論文不同硬體上的報告數字。所有本機PPA均是未校準的解析估計，不是晶片量測。

2026-11-04補讀上述兩篇原始預印本後，保留兩項判讀：one-step／critic-free也可以是如實定義的RL，不能僅憑gamma=0就否定整類方法；反過來，本版使用multi-step Double Q本身也不構成搜尋優勢證明。若完整oracle昂貴、強起點後Q不再提供收益，優先以凍結Q／相同horizon均勻選擇來辨認是表示、bootstrap、探索軌跡或候選品質的問題，再決定是否換policy。不能因為CORE或MARL在不同成本模型報告大倍率，就承諾本機也有相同幅度。
