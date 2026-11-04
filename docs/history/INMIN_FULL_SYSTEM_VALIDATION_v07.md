# Full v0.7 驗證

2026-11-04；來源 `b6bf8e77f95b1e47fff56491b248fd9ac55a8884aa505d5e12dcac75bdda2f05`，208個來源檔。根目錄程式與本輪已驗證凍結engine逐byte一致；舊來源與歷史報告完整保留。

|驗證|結果|本機原始證據|
|---|---|---|
|完整回歸|182項，零failure/error；來源穩定|`output/inmin_v07_20261104/study_v4/acceptance/result.json`、`tests.log`|
|實際CLI|201.035秒通過；三方法、真實50布局、嚴格限制、共享計費|`study_v4/workflow/result.json`、`commands.log`|
|成本拆分Q|絕對after觀察對before/progress/action family不變；實際非terminal Bellman；錯誤reward拒絕；exact save/resume|`tests/test_cost_value.py`與CLI checkpoint|
|量測後reset|真實Toy PPA觸發、負reward保留、後續回到verified elite；續跑軌跡及Q更新一致|`tests/test_episode_cost.py`、workflow/measured_restart|
|架構轉移|全部50布局資源、IO、比例份額及idle守恆|`tests/test_full_system.py`|
|實際GUI|15欄位、token/狀態/嚴格推薦API，8個完整Toy PPA|`study_v4/gui_result.json`|
|安全更新|從逐byte5f2升級；12個變更檔原版備份；新來源208檔核對|`output/releases/20261104_cost_value_release/promotion.json`|

`study_v4`相對路徑位於 `output/inmin_v07_20261104`。不修改實驗期間的engine、protocol或orchestrator。共享歷史只在78個物理來源、輸入與數值環境相符時重用；query及時間原額繼承。非神經控制只計算選中候選的描述，避免無用途的整批feature成本。

既有實驗延長、舊最佳mapping完整PPA重放、R50相同reference/seed僅計1次共同PPA等歷史驗證保留在 [v0.6報告](history/INMIN_FULL_SYSTEM_VALIDATION_v06.md)。未把這些舊測試重新命名為新來源的新增PPA。

驗收證明軟體流程、物理資源與計帳一致性，不證明晶片量測準確、全域最佳或統計優勢。新開發組RL改善16.06%，Random24.42%，凍結Q0.89%；完整數字與限制見 [新結果報告](INMIN_COST_VALUE_RESULTS.md)。另外的架構refresh修正版仍需自身QA與新PPA，沒有混入這份通過結果。
