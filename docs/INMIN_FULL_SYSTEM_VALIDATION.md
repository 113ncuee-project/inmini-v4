# Full v0.8 驗證

2026-11-04；目前來源 `9ee5237e2f5ab48bb98d6cb388f88427170faafec4e8c7218edf3e2cb3651562`，212個來源檔。根目錄程式與已驗證凍結engine逐byte一致；舊來源與歷史報告完整保留。

|驗證|實際結果|本機原始證據|
|---|---|---|
|回歸覆蓋|188項，零failure/error。實際執行於來源7d98；該次測試涵蓋的核心／測試／native檔與9ee逐byte相同，原結果原樣保留|`output/inmin_v08_20261104/study_v3/acceptance/result.json`、`UNCHANGED_SOURCE_REUSE.json`|
|新版實際CLI|313.599857秒通過；本次以9ee重新執行，包含全部可選service/rank/refresh分支、真實50布局、嚴格限制、共享計費及三方法精確stop/resume|`study_v3/workflow/result.json`、`commands.log`|
|真實非terminal學習|額外實評Toy continuation，檢查Double-Q replay與正確bootstrap；未將全terminal短測冒充覆蓋|`study_v3/workflow/physical_continuation`|
|顆數frame|maximin排程、全部同顆數矩形／兩類quota、失敗計帳與精確保存恢復|`tests/test_architecture_refresh.py`與CLI checkpoint|
|可選成本排序／service|排序輔助損失與exact save/load；service是局部描述而非PPA。主研究均關閉|`tests/test_cost_ranking.py`、CLI可選分支|
|實際GUI|16欄位、嚴格推薦API，8個完整Toy PPA|`study_v3/gui_result.json`|
|安全更新|從逐byteb6升級；13個變更／新增檔原版備份，新來源212檔核對|`output/releases/20261104_stratified_release/promotion.json`|
|夜間啟動器|實際DryRun通過：初始化延長、refresh4、事前硬限制及跨RNG共享初始化命令；0新PPA|`run_inmin_full_overnight.ps1`，另見下述界線|

上表 `study_v3` 相對路徑位於 `output/inmin_v08_20261104`。188項結果不是宣稱在9ee再執行一次；兩版本唯一不同是獨立CLI測試harness，而該harness已在9ee真正重跑。前兩個CLI驗收失敗及來源保留，詳見新版設計說明。夜間啟動器位於來源包之外，DryRun驗證了命令組裝；未假稱已另跑整晚多模型研究。

主要物理compiler、communication、hardware、calendar及能量計算與前版一致。研究只重用78個物理來源、輸入與數值環境相符的共同歷史；原查詢與時間完整繼承，不重用控制方法搜尋標籤或policy。相同來源的凍結checkpoint續跑不更改演算法設定。

軟體驗收證明流程、資源與計帳一致性，不證明晶片量測精度、全域最佳或統計優勢。第一個已審計開發RL arm完成56新完整PPA，在強共同起點之後降低16.47% EDP；匹配控制及獨立種子尚未全部完成。請以[分層探索說明](INMIN_STRATIFIED_SEARCH.md)與原始數字為準。

前版實際182項／201.035秒CLI／GUI驗證保留於 [v0.7歷史報告](history/INMIN_FULL_SYSTEM_VALIDATION_v07.md)；歷史長跑、完整PPA回放及延長測試也保留，未重新記為新版新增PPA。
