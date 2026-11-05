# Full v0.9功能驗收與研究界線

2026-11-05；來源 `b771758cbeba3788922270f05e4eb9f459b7336c0031f88e775151b45a1d89b4`，218個來源檔。原v08驗收已保存在history，原profiles及報告均保留。

|項目|實際證據|結果|
|---|---|---|
|整套回歸|204項，72.938秒，前後來源相同|通過|
|Serial CLI|fc694實際319.013秒；本版全部driver／learner／物理／CLI harness逐byte相同，僅coordinator與其unit checks改變|明記來源沿用，不聲稱新重跑|
|CLI涵蓋|3方法、停止／續跑、共同初始化／原成本沿用、50個Toy布局、嚴格可行推薦、真正nonterminal、真正殘差fit及V0逐bit固定|通過|
|實際GUI|16fields，8個完整ToyPPA，residual選項／status API／token檢查|通過|
|實際並行|兩種子真實PPA時間區間重疊；與串行逐步action／cost／reward／SGD一致|通過|
|並行續跑|cooperative stop／Resume逐步一致；重複啟動不改活躍owner；完成後Resume0新PPA|通過|
|物理／空間相容|78個物理來源逐byte相同，11個原FullDesignSpace方法AST相同|通過|
|本機PowerShell|6個實際preflight情境；硬限制、參數、原設定續跑、stop、路徑驗證|0PPA／SGD；saved resume fixture明記為synthetic|

以上DNN以外的Tiny／Toy測試是功能驗收。新DNN尚未完成控制與確認，不能把功能驗收或舊資料重播訓練當成搜尋勝利。Root仍保存每個實驗的frozen source／query／time／policy／checkpoint。

v12的舊representation斷言失敗、v13的screen-off等待問題，原始來源及失敗紀錄均保留。v14只修正host readiness和unit checks；學習器與物理模型未再次變更。CLI沿用明細在`workflow/UNCHANGED_SOURCE_REUSE.json`，原實測在v13；實際並行結果在`parallel_workflow/result.json`。

完整本機證據：`output/inmin_v14_20261105/readiness_study_v1`。每種子獨立訓練，共享起點查詢與時間仍計入每臂；SGD、ridge fits、cache hit、timeout和完整PPA各自列帳。PPA仍為未經晶片校準的解析模型。
