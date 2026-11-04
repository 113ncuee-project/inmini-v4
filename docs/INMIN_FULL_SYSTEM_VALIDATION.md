# Full v0.6 驗證

2026-11-04；來源 `5f2fdce00493d95868b4b0837c03c283d590a0c5a274a34de6a6bc3ec37766de`。驗收與本輪凍結研究使用相同程式bytes；物理compiler、hardware、通訊及排程成本沿用相同版本。新特徵／投影是否改善搜尋，須另看研究對照，不能由Toy測試推論。

|驗證|結果|原始證據|
|---|---|---|
|完整回歸|173項，0 failure／error，來源穩定|`output/inmin_v06_20261104/study_v8/acceptance/result.json`、`tests.log`|
|真實CLI流程|通過；約365秒|`study_v8/workflow/result.json`與各CLI原始輸出|
|GUI到PPA|通過，8次真正完整PPA，11個欄位／token／嚴格推薦API驗證|`study_v8/gui_validation.json`與GUI案例|
|完成實驗延長|三方法各14→18；新增12個Toy PPA、0共同重算；原資料不變|`continuation_toy_validation/continuation_validation.json`|
|已公開R18最佳設計重放|1個完整PPA、0SGD；10個物理量吻合，78個物理來源檔案核對|`study_v6/published_best_replay_validation.json`|
|R50共同cache計費|reference與seed相同，正確繼承1個真實查詢，新增0|`r50_single_common_import_preflight/result.json`|

上述相對路徑未寫完整時，位於 `output/inmin_v06_20261104/`。夜間舊版研究與原驗證報告保留。

## CLI實測內容

- RL／SA／Random相同完整查詢上限，逐項source／input／reward／MAC／功率能量一致性審查。
- 正常完成與中途停止／續跑取得相同動作順序、最佳mapping與Q更新數；完成後再resume不新增PPA。
- 實際50個架構PPA，所有方法共同計費一致；等價共同歷史重用不重跑reference／screen。
- 關閉架構篩選時，一般CLI仍真正執行全長DP起點強化，保留搜尋查詢份額。
- 初始化階段可合作停止／續跑，不重複追加共同觀察。
- Gemini-style LP5控制不越出其宣告變數；SA elite restart；凍結Q沒有online SGD，uniform控制沒有隱藏學習。
- 不可能的Area上限產生 `strict_feasible=false`、`state=null`，沒有假推薦。

## 張量／資源正確性

新增測試確認：producer一次materialization與多consumer讀取分開、殘差fan-out保留；群組名稱不影響PPA；整數DP對宣告proxy與窮舉相符、允許idle並守恆；Temporal可重用PE、沒有誤套Spatial總和；內部FD修改不移動不相關外部輸入。

ordered Part／PE特徵以向量化box交集與獨立scalar結果核對；identical destination／region去重；保留真實required-input channels／halo。實際Toy證明，舊graph統計相同的兩個PE排序有不同NoP流量，新特徵可辨識。新增特徵不讀after-PPA報告，沒有把邏輯需求當成物理reward。

相同seed的F on/off，149維base與初始Q完全相同；只開關特徵不改候選或RNG抽樣。Graph Q的save/load及後續replay更新一致。Fusion整數tile測試涵蓋奇數及非平方形狀。

## 發現並修正的整合錯誤

執行器曾因Path物件的JSON日誌中止，未啟動新DNN研究；已轉成明確字串並加入QA-only閘門。CLI關閉架構篩選時曾缺零次查詢欄位；已修正及實測。GUI第一次驗證受執行沙箱禁止localhost連線，保留失敗日誌後，在允許本機連線的環境對同一份來源完成驗證。

`study_v1`至`study_v5`為研究開始前保留的prepared-only來源；v6在日誌階段失敗，v7於整合驗證失敗。這些版本沒有本輪DNN搜尋結果，未把它們計入有效RL收益。實際新研究使用v8凍結來源。原資料與負面結果保留。

## 科學結論的界線

工程驗收證明執行、計帳及物理模型的一致性。它不證明晶片量測準確、全域最優、RL統計優勢或已打敗原版論文。舊版兩RNG完整空間RL勝出，固定2顆對照卻由Random勝出；新強起點與因果對照仍在執行。詳見[數字報告](INMIN_GRAPH_EDIT_RESULTS.md)。
