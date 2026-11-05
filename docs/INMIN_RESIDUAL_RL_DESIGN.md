# 完整空間 RL：固定物理先驗、量測殘差與可訓練的多步價值

2026-11-05。此份文件對應來源 `b771758cbeba3788922270f05e4eb9f459b7336c0031f88e775151b45a1d89b4`、218個來源檔。204項回歸、CLI涵蓋證據、實際GUI與並行精確恢復皆通過；新DNN搜尋研究進行中。CLI以逐byte不變的driver／learner／harness沿用fc694實際319.013秒驗收，來源差異與原始證據另列，不冒稱又重新跑過。這是設計及診斷紀錄，沒有把重播訓練當成搜尋成效。

## 已找到的失敗原因

舊版確實更新權重，成本頭也能學到有用排序，但多步H分支不一定保持辨識力。直接檢查完成的ResNet18 checkpoint，而非只看entropy或梯度是否非零：

|RNG|真實訓練轉移|H標準差|當前bootstrap的H target標準差|第二Tanh狀態|
|---|---:|---:|---:|---|
|7|52|0.00000213|0.68335|64個單元全數飽和|
|37|74|1.49755|3.13227|仍有差異，並非全部退化|
|47|55|0.00000000333|3.89955|64個單元全數飽和|

H target由各自的當前target network產生，並非已知最優回報。這項診斷新增0次PPA、0次SGD。它支持「部分訓練的多步分支退化成近常數」，不能單靠它判定所有RL結果無效。

另有一個可重現的實作問題：共享起點只訓練V，但經由`stack(V+H,V)`反傳時，H會得到零值gradient而非`None`。Adam仍把這些步驟記入H的optimizer state；三個共享暖身checkpoint的H都已累積1992步。當第一個真正TD梯度到來時，其偏差校正已經按老optimizer處理。新版直接訓練V branch，H在暖身期間不進入計算圖，保留冷啟動的optimizer state。

## 三個分工明確的學習部分

令C為同一完整PPA evaluator回傳的目標成本，`phi(C)=10*ln(C)`。

```text
Q(s,a) = phi(C_before) + V0(after-design) + R(s,a) + H(s,a)
```

|部分|輸入及監督|用途|
|---|---|---|
|V0|絕對after-design特徵；已付費共享階段的真實成本|建立跨架構物理成本先驗，開始自身搜尋後固定權重|
|R|自己完成的新PPA；`-phi(C_after)-V0`|修正目前探索鄰域的系統性預測誤差|
|H|實際轉移、合法successor候選、terminal及Double-Q target|學習多步回報與尚未由成本估計器解釋的誤差|

V0及R是輔助物理成本估計器，不是PPA evaluator。任何候選成為best或推薦前，仍須通過完整compile、通訊、資源calendar、容量及能量評估。沒有拿預測值當作已量到的成本。

R以自己的實際量測標籤擬合ridge。預先固定full／local／after三種特徵基底，以及0.01、0.1、1、10、100、1000六種正則化。僅使用訓練標籤的leave-one-out誤差選擇；RMS正規化固定於本輪訓練輸入，並非每次leave-out重建的nested CV。R至少取得8筆自身量測才開始擬合。

完整自身歷史最多保存4096筆；每次擬合上限預設128筆，以固定時間覆蓋加上最新32筆選取，不消耗提案RNG。控制方法的搜尋標籤、已訓練policy及尚未評估的候選不進入R。線性候選擬合次數、神經網路SGD、PPA查詢各自記錄。

## 真正的多步更新

原始reward不變：

```text
r = 10*ln(C_before/C_after)
```

online network在實際合法successor候選中選最大Q，target network計算該候選價值。移除已知before potential後，更新目標為：

```text
y = r - phi(C_before)
    + gamma * [phi(C_after) + V0(next-after) + R_target(next) + H_target(next)]

prediction = V0(current-after) + R(current) + H(current)
loss = SmoothL1(prediction/10, y/10)
```

terminal的future項為0。gamma仍為0.9；target network及target residual每64個SGD步同步。除以10只改變loss的數值尺度，reward、Q、discount、PPA目標及最後報告值沒有更換。H使用兩層LayerNorm／SiLU，末層零初始化。V0共享暖身與H自身TD訓練分開，避免把暖身當成已學會多步決策。

可行的新完整PPA才加入真實轉移。timeout沒有虛構成本；容量失敗保留真實失敗帳本，不捏造reward。原有epsilon-greedy、target/replay、current與best分離，以及較差中間狀態的探索仍保留。

## 修正後的已保存轉移重播

這是開發診斷，使用RNG47原有83筆共享標籤及55筆自身真實轉移，重新訓練兩個版本。兩者共享V權重均與原checkpoint逐bit相同；各自3312次SGD、48次residual fit、864個ridge候選擬合，新增PPA均為0。

|項目|舊residual／Tanh|修正warmup與normalized H|
|---|---:|---:|
|共享暖身後已有optimizer state的H參數張量數|6|0|
|訓練轉移上H範圍|0.0000|34.2927|
|H標準差|約0|5.4638|
|各自當前bootstrap target的擬合MAE|1.5754|0.5018|

這確認多步分支恢復區分輸入的能力。兩個版本的bootstrap target不同，MAE不是相同已知最優target上的準確度比較。它不能證明新搜尋比Random／SA更好，也不是新的DSE樣本。

## 完整設計空間保留

78個物理來源檔與已驗證9ee完全相同。原FullDesignSpace的提案／執行方法保留，新增的只是殘差特徵索引；沒有用四個profile替換整個空間。

|設計變數|保留的範圍|
|---|---|
|架構|1–16顆Compute、50個完整矩形，旋轉分別計價；IO=2R|
|硬體契約|每Compute16PE、共享L2、無實體Core；M/T假設沿用版本化profile|
|Layer Group|Temporal／Spatial合併、拆分與模式修改|
|Partition／quota|合法整數B/K/H/W及PE份額，包含非2次方、idle及層間轉移|
|Placement|有序PE配置、交換、idle relocation及跨層配置|
|DRAM Flow|外部activation、weight及materialized output的獨立端點、付費interleave|
|Dataflow／retention|output_stationary、weight_reuse、activation_reuse；cold／persistent|
|Fusion|合法封閉單出口DAG、可拆／合、可設定深度、整數sink tile|
|Cache|recompute、horizontal_cache、full_cache，完整生命週期及成本|
|內層tile|由實體W/A/O分池容量fitting處理，外層仍可改dataflow／fusion tile|

沿用V3的Batch1；尚未實作的C-panel、任意routing／floorplan、vertical-only cache不聲稱已支援。「完整」指已實作的有效V3變數，有限提案不等於窮舉或全域最佳保證。架構refresh是各方法共用的非學習協調器，其收益不能全算在RL身上。

## 新DNN驗證與採用條件

|階段|預先固定的做法|
|---|---|
|正確性|204項回歸；真實CLI的3方法、殘差擬合及停止續跑；GUI8個Toy PPA；並行2案例與串行逐步結果相等|
|開發|固定RNG7／共享起點；新RL、關R、gamma0但horizon8，各96次新PPA／1800秒|
|採用門檻|新RL必須在相同累計查詢及相同累計時間兩者都比原9ee RL降低至少5% EDP|
|新確認|通過後才跑197／207／217：RL／SA／Random各192新PPA／3600秒；myopic與凍結Q另96新PPA／1800秒|
|未通過|保留負面結果；單獨測原CostQ的cost-rise 0對照，不把兩個介入混成一項優勢|
|固定硬體|最後跑2×1／FusionOff，fresh397／407／417，160新PPA／2700秒，區分mapping收益與硬體擴張|

三個獨立RNG案例可受控並行；每個案例內仍維持原序列決策。共享的只有同一份已付費、可核對的pre-search觀察，不交換搜尋標籤或policy。保留每案例及整批的時間帳本；整批吞吐改善不是單次RL搜尋加速。

跨論文的量測硬體、輸入、成本模型不同。本系統SA是同域對照，不能直接稱為原Gemini binary；新PPA仍是未經晶片校準的解析模型。新版本的優勢必須由上述新DNN對照證明。

## 其他重播種子與目前研究界線

原有RNG7轉移重播時，舊residual的自身bootstrap擬合MAE為0.501，新版為0.822；RNG37為2.119與1.714。三個種子的原共享V均逐bit相同。這些target隨learner而不同，不能用來宣稱預測了已知最優回報。全部重播共20640次SGD、320次residual fit、5760個ridge候選，新增0次PPA。它們是開發診斷，沒有混入新DNN查詢。

v08完整18項資料及五種方法對照保留。與均勻horizon控制相比，v08同時間平均優勢約1.7%，不是已解決的穩定學習優勢。新版本的殘差與gamma0／凍結控制仍須由新DNN結果判定。
