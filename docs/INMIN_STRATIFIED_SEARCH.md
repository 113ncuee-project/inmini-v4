# Full v0.8：分層探索、實際學習與目前證據

更新：2026-11-04T22:33:15.050635+08:00。程式來源 `9ee5237e2f5ab48bb98d6cb388f88427170faafec4e8c7218edf3e2cb3651562`，212個來源檔；軟體驗收已通過，研究仍在執行。這份報告只使用文字與數字；開發結果不等於跨種子或原論文優勢。

## 為什麼修正探索協調

前版v0.7的成本拆分Q在強DP起點之後降低16.06% EDP，但同輪Random降低24.42%。RL的91個新完整PPA只涉及兩個搜尋架構，Random涉及八個；在125個未參與訓練的控制設計上，最佳Random方案被最終成本估計器排在第43。問題包含跨架構探索不足，不能單靠增加SGD或宣稱policy entropy降低就算解決。

本次維持完整V3有效變數與同一物理模型，在RL之外加入顆數分層協調。它每4步提供一次從已驗證elite轉移到另一顆數的完整mapping候選，保留其餘步驟的原完整動作。所有控制方法使用相同協調器，才能判斷收益是否超過提案機制本身。

## 完整流程與各部分責任

|階段|實際方法|可聲稱的角色|
|---|---|---|
|初始mapping|容量感知Tangram-style連續segment DP／beam、Spatial／Temporal、K/P與quota變體|提供強起點；proxy僅提案，完整PPA排名|
|共同架構篩選|50種完整矩形，IO=2R，各自重算DRAM／NoP／完整成本|所有方法相同的前置觀察，收益獨立列帳|
|顆數分層協調|maximin未訪顆數，從真實elite轉移；同顆數全部合法矩形×兩類quota|非學習的探索安排；不能算成RL學會選顆數|
|細部mapping提案|完整原子／複合交易、資源守恆整數DP、Temporal投影、Fusion局部候選|提供合法候選，未評估候選沒有PPA標籤|
|RL選擇|action-conditioned成本拆分Double Q，ε-greedy|根據已支付物理觀察學習mapping交易排序及後續價值|
|完整PPA|實際編譯、資料傳輸、有限資源calendar、buffer lifetime與能量核算|唯一真實成本／reward來源；解析模型，未經晶片校準|
|線上更新|replay、online／target Double Q、實際物理成本輔助學習|每個可行新PPA預設24次minibatch SGD，SGD不等於PPA次數|
|保存與恢復|來源／輸入hash、模型兩代、optimizer／replay／RNG、frame／query／時間帳|同來源精確續跑；延長保留累計成本，不能算獨立種子|

有效空間仍包含1–16顆Compute、50矩形、所有支援整數B/K/H/W、PE份額／排序／交換、Temporal／Spatial分組、外部activation／weight／materialized-output FD、三種已定價dataflow、retention、合法Fusion／tile／cache。每顆Compute為16PE、共享1MiB L2、無實體Core。完整支援變數不表示窮舉所有組合、所有方案皆可行或保證全域最佳。

顆數frame依與elite顆數及已嘗試顆數的最小距離取最大，固定tie-break。以8顆起點為例，最先探索16、1、12；elite改變時後續次序也會變。每個選中mapping仍必須支付完整PPA；一顆轉移失敗不代表該顆數全部mapping無效，也不會給它虛構成本。前版的普通跨架構動作仍然存在。

## Q學習的精確定義

令 `phi(C)=10*log(C)`，物理reward為 `r=phi(C_before)-phi(C_after)`。新網路表示：

```text
Q(before, action) = phi(C_before) + V(after_design) + H(before, action)

V_target = -phi(C_after)
H_target (含V殘差) = r - phi(C_before)
                       + gamma * Q_target(next_state, argmax Q_online(next_state))
```

V只讀65個絕對after-design特徵，不讀before成本、progress或action family，避免起點變強時沿用舊相對收益。V由真實物理成本輔助損失更新；H承擔Double-Q continuation與V誤差。`gamma=.9`，target每64次SGD同步，replay最多4096，預設horizon8。共享初始化只預訓練V，不能捏造未觀察的多步軌跡。

RL可以走到較差但可行的current；best archive獨立保留。當實際成本高於elite的1.25倍，90%機率結束episode，10%保留較差中間狀態探索；實際負reward照記。容量失敗、timeout及缺少PPA皆不以假成本更新。每步保存Q、機率、梯度／loss、terminal原因及完整選中交易，可檢查是否真的更新。

本輪主實驗使用既有260維描述，`service_features=off`、`cost_rank_weight=0`。額外局部service、排序loss、值尺度及雙PPA-head只做過獨立開發診斷，尚未構成主RL或搜尋收益；不得把診斷相關性改寫成已實作的新key technology。

## 本輪已完成與待完成

數字更新：2026-11-04T22:33:15.050635+08:00；原研究狀態 `completed`。下表只包含已完成並通過審計的case。方法尚在進行或尚未執行時，不以局部數字冒充完成端點。

共同階段已支付100次完整PPA、2230.010633秒，其中83個可行觀察，強起點EDP為0.051496084306。每方法首輪搜尋上限96次新完整PPA／1800秒、單次oracle上限600秒。全部使用Batch1／224×224 ResNet18、EDP、相同完整空間，未加事後PPA硬限制。

|Case／RNG|方法|新／累計完整PPA|EDP|強起點後改善%|搜尋秒|T ms|E mJ|Area mm²|平均W|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|development/controls/7|sa|46／146|0.050603473|1.7334|1802.90|2.397929|15.188496|231.031|6.334005|
|development/controls/7|random|96／196|0.034905992|32.2162|1646.38|1.994401|12.596746|161.458|6.316055|
|development/frozen/7|rl|46／146|0.044697498|13.2021|1803.10|2.663623|12.077617|120.031|4.534282|
|development/stratified/7|rl|56／156|0.043013014|16.4732|1803.86|2.458148|12.593973|133.840|5.123359|
|development/uniform/7|rl|96／196|0.035371769|31.3117|1737.55|1.889071|13.476565|212.443|7.133963|
|confirmation/frozen/37|rl|64／164|0.044404875|13.7704|1802.80|2.328620|13.724697|180.046|5.893918|
|confirmation/frozen/47|rl|65／165|0.044290426|13.9926|1802.98|2.646027|12.047208|120.031|4.552942|
|confirmation/stratified/37|rl|78／178|0.036402697|29.3098|1803.01|2.160158|12.128830|159.333|5.614789|
|confirmation/stratified/37|sa|96／196|0.047666239|7.4372|1031.19|2.347924|14.611597|231.031|6.223198|
|confirmation/stratified/37|random|96／196|0.047065591|8.6036|1722.29|2.329522|14.541446|231.031|6.242246|
|confirmation/stratified/47|rl|58／158|0.042156854|18.1358|1803.12|2.430629|12.483040|133.840|5.135724|
|confirmation/stratified/47|sa|64／164|0.042815727|16.8563|1803.48|2.444999|12.603624|133.840|5.154858|
|confirmation/stratified/47|random|65／165|0.048132212|6.5323|1803.86|2.663524|13.006188|133.840|4.883075|
|confirmation/uniform/37|rl|91／191|0.045022468|12.5711|1801.18|2.247461|14.418097|231.031|6.415282|
|confirmation/uniform/47|rl|77／177|0.042493447|17.4822|1802.91|2.434337|12.563544|133.840|5.160972|
|extended_development/controls_7|sa|113／213|0.047133260|8.4721|3601.62|2.592956|13.082872|159.333|5.045543|
|extended_development/controls_7|random|192／292|0.032964994|35.9854|2744.13|1.744963|13.596827|314.414|7.792042|
|extended_development/stratified_7|rl|159／259|0.039315256|23.6539|3601.99|2.255260|12.546867|159.333|5.563379|

各方法實際完成查詢數不同，必須以 `EQUAL_BUDGET.csv` 的相同查詢／相同時間比較判讀；不能只拿端點宣稱query效率。`QUALITY_THRESHOLDS.csv`保留事先固定的5%、10%、20%品質門檻，未達成照記。

`extended_development`是種子7的延長，繼承全部原查詢、時間、policy與RNG，不能算另一個獨立種子。原協定在任何主方法不足16個顆數frame時，三方法一起延長96次新查詢／1800秒。

主方法3個RNG已完成：7的Random超過RL，37及47由RL取得較低EDP；其他凍結／均勻控制若尚未完成仍標為pending。所有結果和較大架構的面積、能量、平均功率取捨都保留。本文沒有假定RL穩定勝出。

以下同查詢與同時間分析，先在每個RNG內把三個主方法截至相同預算，再對3個RNG作描述統計。它們共用已支付的起點，RNG7屬開發，37／47屬確認；不能把此表當作獨立大型實驗或統計顯著性。查詢floor分別為146、178、158次累計完整PPA；時間floor含共同2230.010633秒。全部原始數值見 `main_matched_3seeds.json`。

|比較範圍|方法|3個RNG平均EDP|樣本標準差|中位數|
|---|---|---:|---:|---:|
|completed endpoint; unequal actual queries/time|rl|0.040524189|0.003594895|0.042156854|
|completed endpoint; unequal actual queries/time|sa|0.047028480|0.003932849|0.047666239|
|completed endpoint; unequal actual queries/time|random|0.043367932|0.007347635|0.047065591|
|same q budget across all3 main methods|rl|0.040524189|0.003594895|0.042156854|
|same q budget across all3 main methods|sa|0.047132344|0.003962119|0.047977833|
|same q budget across all3 main methods|random|0.043626093|0.006921547|0.047082861|
|same seconds budget across all3 main methods|rl|0.042022522|0.001064038|0.042156854|
|same seconds budget across all3 main methods|sa|0.047028480|0.003932849|0.047666239|
|same seconds budget across all3 main methods|random|0.044376694|0.008260189|0.048132212|

凍結與均勻控制完成後，還要看五方法一起截齊的結果。加入控制會改變共同query floor，不能把不同floor的平均值混用。`frozen`保留共享成本模型但不作自身更新；`uniform`不訓練，沿用與RL相同的horizon及重啟動態，與每步從elite重新取樣的Random不同。

|方法|5方法共同查詢floor平均EDP|5方法共同時間floor平均EDP|各自完成端點平均EDP|
|---|---:|---:|---:|
|rl|0.042020423|0.042022522|0.040524189|
|sa|0.047132344|0.047028480|0.047028480|
|random|0.044354346|0.044376694|0.043367932|
|frozen|0.044464266|0.046445839|0.044464266|
|uniform|0.044435060|0.042752487|0.040962561|

完整控制提醒我們：RL相對均勻horizon策略的平均端點優勢僅約1.1%，同時間floor約1.7%；即使同query floor約5.4%，仍不足以宣稱穩定的多步學習優勢。這也是繼續修正critic與新增gamma0消融的原因，不能只報相對SA／Random的有利數字。

首個RL在第12次新PPA／267.198559秒，藉12顆2×6比例份額轉移找到0.043298463272；第27步普通group_projection再到0.043013014425。前者的顆數由非學習協調器選定，不能把整段收益全部歸於RL。最終設計沒有Fusion，面積與平均功率高於8顆共同起點。

新追加的[學習檢查](INMIN_POLICY_LEARNING_RESULTS.md)使用相同未訓練控制設計，比較共享初始化與最終模型，移除已知before成本項並檢查同架構／分組內的排序。它能檢查泛化與退步，仍不能替代搜尋控制或證明future-Q因果收益。

後續原型另修正實際發現的continuation飽和與暖身Adam空更新：固定共享先驗、線上殘差、LayerNorm／SiLU多步Q；已過202項回歸，實際CLI／GUI／並行精確恢復與新DNN尚未全部完成，不混入本表。若未達預先固定的5%雙預算改善門檻，再單獨測試重啟條件。固定2Compute／FusionOff保留為最後控制切片。不同RNG案例的受控並行只改善整批吞吐，不代表單一搜尋加速或學習成功。

## RACE-RL參考與論文界線

同學的原生RACE-RL是關係編碼器、conditional Actor與REINFORCE，並不是本系統Double Q；硬體為256PE／chiplet、主要64×64輸入、不同memory／area／roofline模型，不能直接比較絕對PPA。參考其關係表示、完整transaction與顆數探索思想，並未偷偷換掉16PE物理後端或縮成四個profile。完整核對見 [RACE來源報告](INMIN_RACE_REFERENCE_AUDIT.md)。

現有證據可以支持「強起點後找到實際改善設計」，尚不能支持「穩定擊敗Random／原生Gemini／其他論文」或「全域最佳」。SA／Gemini5與Tangram-style皆為同域改編對照，非原論文執行檔。

## 驗收與失敗保存

188項回歸實際在7d98來源執行，與9ee的被測核心／測試／native bytes相同，原結果與reuse證書保留；唯一改動的CLI harness已於9ee實跑313.599857秒通過，GUI16欄位與8次實際Toy PPA通過。沒有將同一回歸計為兩次新增測試。

study_v1因CLI測試的手動中斷命令漏傳可選旗標，造成308／260維設定不同，保留失敗；study_v2的精確續跑已通過，但短測全terminal，未覆蓋非terminal分支，故保留失敗並在v3加入真正的物理continuation測試。這兩次均沒有新DNN研究PPA。

原始研究位於 `output/inmin_v08_20261104/study_v3`，固定硬體後續位於 `fixed_study_v1`；凍結engine、protocol、orchestrator不可中途修改。每次交付保留來源hash、查詢與時間、checkpoint、可行與不可行、timeout及缺失結果。自動每小時續作排程維持暫停；目前本機一次性實驗佇列自行執行，不需要LLM在線。
