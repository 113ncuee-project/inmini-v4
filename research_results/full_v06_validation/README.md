# Full v0.6：已通過的工程驗收

目前程式來源：`5f2fdce00493d95868b4b0837c03c283d590a0c5a274a34de6a6bc3ec37766de`。這裡是正確性及操作證據；新強起點的RL效益仍由獨立研究對照，不從Toy收益推論。

|檔案|結果|
|---|---|
|`acceptance_result.json`|173項，0 failure／error，source-stable|
|`workflow_result.json`|實際CLI、50布局、共享歷史、精確停止／續跑、強起點及因果控制通過|
|`gui_result.json`|GUI端到端8個完整PPA、token／嚴格推薦API通過|
|`continuation_result.json`|三方法14→18各新增4次，共12次；0次共同重算，舊資料不變，Q更新96→128|
|`promotion.json`|本機193→205個程式來源檔案，保留修改前副本；新來源與驗收snapshot完全相同|

延長工具測試使用其檔案內記錄的shipping Toy來源`6294492...`，並由新launcher續跑凍結的原程式；不是將舊測試掛成新的DNN結果。工程重放及[舊版完整空間實驗](../full_v05_completed/README.md)另保留來源及計費。

新研究的原始protocol與已觀察結果見[技術協定](../../docs/INMIN_GRAPH_EDIT_DEVELOPMENT.md)、[數字結果](../../docs/INMIN_GRAPH_EDIT_RESULTS.md)。未完成消融不填入預期改善率。
