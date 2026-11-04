# INMIN Full v0.5：驗證與今夜研究協定

更新：2026-11-03 22:50（Asia/Taipei）。完整系統程式已實作；今夜 DNN 對照仍在執行。本文將工程驗證與尚未完成的研究勝負分開記錄。

## 已完成的驗證

|檢查|結果|原始證據（專案根目錄下）|
|---|---|---|
|歷史 v3／v4、完整域、交易、物理成本、GUI4、seed、native calendar與Double Q回歸|154項，0 failure，0 error，來源穩定；82.782秒|`output/inmin_v05_20261103/acceptance_shipping/result.json`與`tests.log`|
|完整空間各種整數矩形|全部50種結構合法；IO=2R、目標PE與memory端點守恆|`tests/test_full_system.py`|
|三方法共同架構階段|Toy上50個布局均完成實際PPA；各方法共同計52次（原reference／seed兩次＋50布局）|`workflow_shipping/shared_screen/screening.json`|
|查詢帳本與三方法端到端|各58次charged full PPA；實際共享只執行一次|`workflow_shipping/shared_screen/*/summary.json`與`analysis/audit.json`|
|中斷與續跑|相同選中mapping序列、best與SGD數；無重複durable PPA；完成resume零新PPA|`workflow_shipping/result.json`，186.613秒|
|硬性條件沒有可行解|推薦`state=null`；closest另列，沒有把違規設計當推薦|`workflow_shipping/infeasible/rl/recommendation.json`|
|新GUI HTTP與實際工作|6個控制欄位、status API、token驗證；從GUI成功完成8次實際Toy PPA；審計0 error/0 warning|`output/full_system_gui/20261103_224040_1ec992/`|
|長跑輸出／多模型參數／自動報告|PowerShell5.1 dry-run通過兩個RNG與三模型；已完成的三方法fixture可生成數值總表|`helper_checks/FULL_NIGHT_REPORT.md`|

此表中 `workflow_shipping/`、`helper_checks/` 皆位於 `output/inmin_v05_20261103/`。Toy用於正確性；其改善百分比不作為論文DNN收益。

新GUI的實際執行快照、交付版本與今夜研究快照，以下四個核心檔案逐一SHA256相同：`full_system_dse.py`、`hybrid/full_space.py`、`hybrid/multi_step_q.py`、`hybrid/single_array_runtime.py`。最終完整APP來源identity為 `336f979b54946ba967d3c16dcf56b9aa7fca3bb43dc6aa304b440540d3858933`；各工程測試另保存其當時完整source manifest。部分報告／打包輔助腳本在測試後修復並另做實際輸出檢查，沒有改寫舊測試結果。

## 正式夜間協定

|項目|設定|
|---|---|
|模型／工作量|ResNet18，Batch=1，原始完整DAG與MAC|
|RNG|7、17；相同原始最佳Tangram-style起點|
|方法|multi-step Replay RL、相同空間的SA、Random|
|主要目標|事先指定EDP=Energy×Latency；同列Latency／Energy／Area／Power|
|設計域|v3全部50個架構＋完整Mapping／Grouping／Fusion／Tile／Cache／Dataflow／獨立FD|
|每方法查詢上限|256次完整PPA，含同額共同reference／seed及架構篩選；SGD／proxy／cache不算full PPA|
|每方法搜尋上限|2400秒；每次Oracle180秒；共同架構篩選1800秒|
|來源|每案例凍結程式、輸入及環境；兩RNG拒絕混用不同live source|
|實際路徑|`output/inmin_v05_20261103/night_edp/resnet18/7`與`17`|

架構順序先對每個1～16顆數各取一個較方正布局，再測其餘方向，降低時間上限只涵蓋小顆數的偏差。有限預算未完成者保留狀態，並不刪除其後續可達性。任何「已完成50種」的說法都必須由該案例的`screening.json`實際證明。

初步篩選已找到比原最佳Tangram-style seed同時降低Latency、Energy、Area的設計。此時尚未完成RL精修，不能把共同架構／內層收益歸因於RL。`gain_before_rl_pct`、`gain_after_shared_screen_pct`、`equal_budget.json`及held-out policy報告用來做歸因。

主研究結束後，獨立佇列會對ResNet50／MobileNetV2各執行最多12次full PPA的固定2×1相容性檢查，保留成功與失敗；這是跨模型執行檢查，不能當作跨模型統計勝出。最後會建立新程式／數值結果ZIP並做CRC與逐檔SHA256驗證。

## 明早查看

- 主研究總表：`output/inmin_v05_20261103/night_edp/FULL_NIGHT_REPORT.md`。
- 每個RNG的完整表：`resnet18/<seed>/REPORT.md`，公平預算：`analysis/equal_budget.json`。
- 每個方法已驗證的最佳設計：`<method>/recommendation.json`；全局Pareto與每架構最佳：`analysis/measured_pareto.json`、`architecture_best.json`。
- Q輔助reward head的未見控制樣本排序：`analysis/held_out_policy.json`。該排序不是已證明future-Q準確或RL搜尋勝出。
- 主工作日誌：`output/inmin_v05_20261103/night_launcher.out.log`、`.err.log`。
- 全部後處理完成狀態：`output/inmin_v05_20261103/OVERNIGHT_COMPLETE.json`。
- 明早下載包清單：`output/uploads/INMIN_Full_20261104_morning_package.json`。

完成狀態與檔案必須實際存在才可當成完成。電腦需保持開機；可鎖定螢幕。這是已授權的一次性本機工作，原每小時自動排程仍暫停，不需Codex持續操作，也未使用重置券。

## 已知範圍與保留的負面記錄

Balanced開發probe `r18_joint_128` 在架構階段合作停止，沒有完成三方法研究。它顯示單顆晶片可能以Latency增加換取平均Power／Area降低，因此主研究預先改用EDP，並保留該probe的來源／實際PPA與中斷記錄。

第一次workflow發現Spatial Group已滿時的idle quota抽樣邊界錯誤，已修復並加入多狀態domain walk；最早失敗目錄仍保留。新GUI曾因在case內預先建立launcher log而被CLI拒絕，已改到case外，且後續GUI真實8次PPA驗證通過。自動總表曾對錯誤層級／重複method鍵失敗，已修復並用已完成三方法資料生成表格。

所有PPA仍是未校準解析估計。比較對象是專案的bounded Tangram-style seed與同Oracle的SA；沒有宣稱直接打敗原版Gemini二進位或保證全域最佳。RL是否更快／更好，必須等相同實際query與wall budget結果，不能由程式已實作或entropy已下降推論。
