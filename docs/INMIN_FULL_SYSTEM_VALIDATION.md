# Full v0.10：PPA偏好功能驗收與研究界線

2026-11-05；來源 `f434af44f03a95d0860f4f8ebfa94e248548f98a090613d952fe74884cf7b5f5`，220檔。舊版報告及profile保存在history，正在執行的EDP研究仍使用原凍結b771來源。

|項目|實際證據|結果|
|---|---|---|
|整套回歸|210項，108.906秒；0錯誤／失敗，來源前後相同|通過|
|V3偏好一致|Balanced／低延遲／低功率／小面積權重與原V3一致；四個舊研究公式逐bit保留|通過|
|真實物理PPA|偏好單元測試使用實際Toy calendar結果；功率與能量分開、硬上限獨立|通過|
|實際child-CLI|四偏好×RL／SA／Random，另低功率完整與停止續跑兩案例；303.415秒|通過|
|child-CLI實際成本|六個案例共170次真實Toy PPA、632次SGD；逐案audit及原始帳本保留|不是0PPA，也不是DNN效益樣本|
|child-CLI精確恢復|相同偏好：action／cost／reward／SGD一致；改偏好：新增PPA前拒絕|通過|
|偏好保存|objective.json及checkpoint／summary／recommendation保存精確權重、硬限制、固定reference|通過|
|實際GUI HTTP預覽|七目標正確轉成CLI參數；V3 preference別名、相斥目標、NaN／Inf硬限制、token、原job不變|0新增PPA／SGD，通過|
|實際PowerShell|七個並行wrapper目標＋串行低功率preflight；偏好／硬限制正確，未建立實驗目錄|0新增PPA／SGD，通過|
|物理／RL／空間相容|78物理檔逐byte一致，learner／FullDesignSpace／架構refresh未改|通過|

實際internal child-CLI測試是小型功能fixture，執行完整compiler／communication／calendar／buffer／reward／learner，但沒有聲稱重新跑過外層parent的global admission或整批cohort。b771的實際外層並行／串行／停止續跑證據保留原始來源與時間，詳見history的v09驗收；不把它標記成f434的新執行。功能測試與三個DNN確認案例同機執行，時間不可當作隔離的搜尋加速。

第一次可變原型驗收的210個個別測試全通過，但我在測試期間仍修改了來源，來源穩定檢查因此正確失敗。這次不採用；保存原log與失敗說明後，凍結完成來源，再取得上表108.906秒的實際通過結果。第一次localhost檢查被沙箱阻擋，後來以實際HTTP重新測通；沒有把未執行當成通過。

完整本機證據：`output/inmin_v17_20261105/preference_readiness_v1`。`VALIDATION_RECEIPT.json`逐案例列出實際PPA／SGD，`HTTP_preflight.json`、`wrappers/result.json`記錄0搜尋的預覽。原完整b771工作流在`output/inmin_v14_20261105/readiness_study_v1`。

目前長跑目標為EDP。開發主臂和無殘差、gamma0消融已完成；fresh197／207／217與固定硬體對照尚未完成。不能以這些EDP結果宣稱LowPower／SmallArea／Balanced的DNN搜尋已改善，也不能把低平均功率當成低能量。所有PPA為未經晶片校準的解析模型；完整成本與資源守恆優先，不宣稱穩定勝過Random／SA、原論文binary或晶片量測。
