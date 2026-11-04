> 歷史版本記錄：以下為v0.5／v0.6當時的來源、實驗與協定。現行程式及新結果請見 [成本拆分Q報告](INMIN_COST_VALUE_RESULTS.md) 與 [操作指南](INMIN_FULL_SYSTEM_GUIDE.md)。

# INMIN 完整空間 RL：已完成結果與待驗證項目

更新：2026-11-04上午，Asia/Taipei。這份文件區分已完成的來源336實驗與正在執行的graph-edit研發版本，避免將候選機制當成已證實的效果。所有PPA是同一套未校準解析估計，報告不含圖片。

目前交付程式來源為 `5f2fdce00493d95868b4b0837c03c283d590a0c5a274a34de6a6bc3ec37766de`，173項回歸、完整CLI與GUI 8次PPA通過。`study_v8`已完成強DP／inner起點與四組F/P開發實驗；因改善仍小且發現暖身起點轉換偏差，在確認對照前明確中止並保留結果，優先驗證另一份凍結的成本拆分Q版本。下面原v0.5數字的source仍是336，沒有轉掛成新版成效。

## 新完成：較強共同起點，收益不歸給RL

全長contiguous segment-DP／beam16、保留及關閉whole-output剪枝，加上事先宣告的quota／Partition／placement／FD／retention內層候選，於2顆與8顆架構各新增24個完整PPA。48次全部計費，包含15個不可行結果；共同歷史合計100次完整PPA、2230.011秒，83個可行觀察。兩個RNG與所有方法取得完全相同的物理觀察。

|方案|正規化EDP|Latency ms|Energy mJ|Area mm²|平均Power W|
|---|---:|---:|---:|---:|---:|
|原50布局共同起點|0.07957726|4.191008|13.665996|106.221720|3.260789|
|舊版RL最佳（RNG17）|0.06012576|3.603077|12.010410|106.221720|3.333376|
|新DP／inner共同起點|0.05149608|3.111251|11.912693|106.221720|3.828907|

新起點比舊RL最佳EDP低14.354%。這證明先前起點仍有可直接由DP／內層處理的改善空間，不能把這部分當成RL技術收益。平均功率增加，是較短執行時間與能量共同決定的數值，不是所有PPA均下降。

四組開發實驗皆完成、audit通過；F為新增特徵、P為資源投影。共同100個PPA皆計費，下面是起點之後的新查詢。

|設定／RNG7|新增完整PPA|最佳EDP|起點後改善%|搜尋秒|最佳出現階段|
|---|---:|---:|---:|---:|---|
|F0P0|21|0.051452048|0.08551|901.594|uniform warmup，第1步|
|F1P0|14|0.051452048|0.08551|903.326|uniform warmup，第1步|
|F0P1|13|0.051078444|0.81101|903.483|uniform warmup，第5步|
|F1P1|33|0.050291041|2.34007|905.222|learning，第18步|

F1P1第18步選中Q排名第1的A10（機率0.674695），合併成單一Temporal Group；最終也包含5層mapping變更，Fusion為0。T=3.067680ms、E=11.799167mJ、A=106.221720mm²、平均P=3.846283W。這是有用的已測動作，但缺少同域／凍結／均勻控制，尚不能證明收益來自學習。

其他三組最佳都在uniform warmup產生，不歸因於Policy。900秒內的低查詢數主要由實際Oracle耗時與timeout主導；F0P0兩個高quota方案各約242秒逾時。另用83個既有共同觀察檢查，F1的即時reward預測從原起點約0.5 MAE惡化到強起點約6.6；此校準偏差本身不證明動作排序變差，新的成本拆分表示仍須用新PPA驗證。

四組完成後，按事先明記的開發切換條件停止原協調程序；尚未執行的SA／Random／LP5／凍結／均勻與RNG17確認組沒有填入預期結果。凍結來源與protocol不變，原始數字、timeout與中止理由完整保留。[四組原始證據](../research_results/full_v06_development/README.md) · [共同起點原始數字](../research_results/full_v06_strong_start/README.md)。

## 已完成：完整空間多步Q搜尋，兩個RNG

工作負載ResNet18、Batch1、EDP目標。三方法各自使用相同原始Tangram-style adapter起點及50布局共同完整篩選，原有2次common加50次screen全部計入每個方法。硬體變數為Compute1–16、50完整矩形、每Compute16PE、共享L2、IO=2R；不是固定峰值算力的架構比較。

本輪未設定Area／平均Power／Latency硬上限；不能把8顆方案當成符合原本2W或100mm²限制的推薦。這些硬限制可由CLI／GUI明確設定，受限實驗必須另開輸出重新比較，不能混用本表數字。

|RNG|方法|新增完整PPA|含共同階段完整PPA|最佳EDP|共同篩選後改善%|搜尋秒|
|---:|---|---:|---:|---:|---:|---:|
|7|RL|141|193|0.06109825|23.2215|2403.502|
|7|SA|107|159|0.07943955|0.1731|2401.913|
|7|Random|197|249|0.06634240|16.6315|2401.328|
|17|RL|204|256|0.06012576|24.4435|2086.290|
|17|SA|126|178|0.07690129|3.3627|2402.978|
|17|Random|204|256|0.06324071|20.5292|2037.859|

各組均通過來源、完整PPA、reward與學習帳本審查。RNG17在時間用完前達到query上限；不同方法完成數不相等，因此另外作共同預算比較。

|RNG|共同預算|RL EDP|Random EDP|SA EDP|RL相對Random下降%|
|---:|---|---:|---:|---:|---:|
|7|159個含共同階段完整查詢|0.06136426|0.07468540|0.07943955|17.837|
|17|178個含共同階段完整查詢|0.06047397|0.06945064|0.07690129|12.925|
|7|3579.559秒含共同階段時間|0.06109825|0.06634240|0.07943955|7.905|
|17|2609.696秒含共同階段時間|0.06012576|0.06324071|0.07690129|4.926|

共同起點EDP=0.07957726。相對它改善10%的首次已驗證時間：RNG7 RL搜尋563.037秒／Random1914.944秒，RNG17 RL332.031秒／Random509.942秒；兩組SA均未到達。這是已觀察到的到達時間，尚不足以建立一般化的統計優勢。

## 真正改善了哪些物理量

|方案|Latency ms|Energy mJ|Area mm²|平均Power W|Compute布局|
|---|---:|---:|---:|---:|---|
|原始較佳adapter seed|11.818170|18.438092|64.794232|1.560148|2顆／2×1|
|共同50布局篩選起點|4.191008|13.665996|106.221720|3.260789|8顆／2×4|
|RNG7 RL最佳|3.620577|12.145677|106.221720|3.354624|8顆／2×4|
|RNG17 RL最佳|3.603077|12.010410|106.221720|3.333376|8顆／2×4|

原始seed到共同起點的EDP收益約73.72%，發生在RL前；不能歸給policy。面積相對原2顆方案增加63.94%，平均功率也增加。共同起點到RL最佳則保持8顆架構，得到額外的latency／energy改善，仍須與較強起點及固定硬體控制比較。

重要限制：兩個RL最佳都大幅改變group邊界。RNG7最終為單一Temporal Group，Fusion為0。原始Tangram-style seed有group長度限制，較強、允許全長segment的DP可能取得相近結果。因此目前不把這些數字宣稱為打敗強Tangram或原版Gemini。

## Policy學習的證據與尚未證明的部分

|項目|RNG7|RNG17|
|---|---:|---:|
|實際可行新PPA／學習transition|138|200|
|其中非terminal多步transition|114|168|
|含共同暖身的SGD更新|4512|6000|
|學習階段決策正規化entropy中位數|0.423|0.338|
|未用於RL訓練的控制臂候選數|251|276|
|reward輔助head：共同暖身後Spearman|0.113|0.323|
|reward輔助head：搜尋後Spearman|0.642|0.584|

這已不同於先前v0.3近乎均勻的Actor。模型使用gamma=.9的Double Q、target與replay，確實具有非terminal Bellman更新。獨立控制候選的即時reward排序改善，也包含固定架構且不改group的子集。但這是輔助reward-head診斷，不能代替future-Q或「學習造成搜尋優勢」的因果驗證；控制臂沒有提供單一group內fine-mapping的held-out樣本。

ε-greedy本來就會讓候選分布較集中，即使Q尚未學好也會如此；因此不能只用低entropy證明policy有效。搜尋成效、未訓練候選的預測排序及凍結／均勻控制須一起判讀。

另有可直接修正的浪費：舊RL的A8中，RNG7有14次、RNG17有40次修改「當下完全在group內、沒有外部materialization」的tensor DRAM端點。所有54次完整物理PPA tuple都與選前相同，分別花155.126、359.445秒。新來源已遮罩這類當下無作用的A8；當group邊界啟用時，端點仍可選，沒有刪掉有效FD搜尋域。這是已確認的舊版問題及對應程式修正，實際新系統的搜尋收益仍待對照，不能先把節省時間寫成已測成效。

## 下一輪：較強起點與可歸因控制

實際凍結來源為study_v8／`5f2fdce00493d95868b4b0837c03c283d590a0c5a274a34de6a6bc3ec37766de`。完整回歸、實際CLI stop/resume、共同歷史重用及GUI均已通過，目前執行wider Tangram-style DP和宣告的inner mapping。這些完整PPA、失敗與耗時都提供給所有比較臂並共同計費，沒有使用上一輪RL／SA／Random結果訓練新policy。

之後才測：新增張量邊界特徵F與守恆資源投影P的四組development消融，同域較強SA／Random、Gemini五類操作的語義對照，以及獨立RNG17確認。凍結共同暖身Q與均勻選擇、相同horizon兩項控制分離「候選好」與「學習有效」。所有失敗與落敗設定保留。新一輪尚未完成，不填入預期收益。

## 已完成的負面控制：固定2顆Compute

ResNet18、2×1、同一Tangram-style起點、允許Mapping／Grouping／Fusion、各128個完整查詢（其中2共同）。這組沒有架構數量收益，三方法均完成及audit通過。

|方法|EDP|相對seed改善%|Latency ms|Energy mJ|搜尋秒|最佳Fusion深度|
|---|---:|---:|---:|---:|---:|---:|
|RL|0.17023650|43.7714|9.439949|12.979377|1004.209|0|
|SA|0.16958012|43.9882|9.411832|12.967959|976.497|0|
|Random|0.16374609|45.9152|9.152150|12.877117|933.069|2|

固定硬體由Random勝出，不能宣稱原RL在純Mapping必定較好。另做inner-only配對：只開auto weight retention就可改善原seed EDP約10.176%，三種Dataflow的局部選擇則沒有額外改善。這是內層演算法的收益，不是policy學習證據。

## 真正可行的Fusion On／Off配對

對上述Random最佳，完整保留架構、group、PE、Part、dataflow、retention與FD，只移除FusionStack。原On結果重用已測物理標籤，Off獨立新增1個完整PPA，均可行；沒有導入RL訓練或共同起點。

Fusion為 `layer1_0_conv2 → add`，sink tile16×8，recompute模式。

|指標|Fusion On|Fusion Off|
|---|---:|---:|
|EDP|0.163746090|0.163006939|
|Latency ms|9.152150|9.156854|
|Energy mJ|12.877117|12.812404|
|DRAM Read B|18,882,212|18,082,724|
|實際Weight Read B|11,688,320|11,688,320|
|Activation Read B（總讀取減weight）|7,193,892|6,394,404|
|NoC hop-B|99,329,408|99,520,640|
|NoP hop-B|38,809,328|37,705,504|
|Executed MAC|1,814,073,344|1,814,073,344|
|Recompute MAC|0|0|
|Executable units|2500|2408|

此例Fusion讓EDP變差0.4534%。額外799,488B DRAM讀取全來自activation，沒有增加實際權重讀取或MAC；不能歸因於重算。輸出切成更多tile後，Conv所需輸入region／halo的重複需求及當前跨tile去重／存活策略是待進一步拆解的部分。兩方案硬體完全相同，也沒有新增buffer，故不能寫成「新硬體一律讓Fusion增加流量」。同group的Off本來已有L2 forwarding，Fusion未必還能省掉一次DRAM materialization。

這個配對證明當前compiler／scheduler／cache模型下，這個Fusion方案沒有收益；沒有證明該layer組合在所有合法cache與排程下都不適合融合。也不能把Random最佳的45.9152%收益歸給Fusion，因為相同mapping的Off反而略好。

## 跨模型執行檢查

|模型|方法／RNG|含共同階段完整PPA|Seed相對EDP改善%|搜尋秒|
|---|---|---:|---:|---:|
|ResNet50 v1.5|RL／7|12|58.2953|569.057|
|MobileNetV2|RL／7|12|11.0706|80.598|

兩者完成並audit通過；這是小預算執行檢查，沒有SA／Random配對或多RNG，不作RL優勢結論。R50 reference與seed相同，實際共同PPA為1，新增搜尋為11；不能硬寫成2＋10。

## 組員查核入口

1. 操作及完整設計空間：`docs/INMIN_FULL_SYSTEM_GUIDE.md`；新機制／事前協定：`docs/INMIN_GRAPH_EDIT_DEVELOPMENT.md`。
2. 已完成原始研究：`output/inmin_v05_20261103/night_edp/resnet18/{7,17}`。各方法`recommendation.json`提供完整可重放mapping及物理數字，`analysis/audit.json`提供驗證結果。
3. 新研究：`output/inmin_v06_20261104/study_v8`。`protocol.json`、`engine`、`orchestrate.py`凍結；`status.json`及`commands.log`記錄進度。v1–v5只準備，v6／v7於研究前整合檢查失敗，原來源與負面紀錄保留。
4. 額外數字抽取：`output/inmin_v06_20261104/completed_v05_evidence`，可由`tmp/inmin_v06_graph/summarize_completed_v05.py`重建；不執行新PPA或SGD。

方法名稱須如實寫成「INMIN Tangram-style adapter」和「同一物理Oracle下的SA／Gemini-style OP1–5」，不能將它們寫成原版論文執行檔再現。本輪主要回答同硬體、同成本定義及實際預算下的搜尋成效。
