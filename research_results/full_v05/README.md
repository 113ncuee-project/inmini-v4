# Full v0.5 工程驗證

這一層是完整系統的正確性與操作證據。Toy結果不作為ResNet18或論文收益。

|檔案|用途|
|---|---|
|`acceptance_result.json`|154項regression，0 failure／error，source-stable|
|`workflow_result.json`|三方法、真實50布局PPA、相同共同計費、完整停止／續跑|
|`toy_shared_screen_summary.json`|每方法58次charged full PPA，其中52共同，實際共享執行一次|
|`gui_toy_summary.json`／`gui_toy_audit.json`|新GUI端到端8次完整PPA，審計通過|
|`night_protocol.json`|正式ResNet18兩RNG／RL、SA、Random的預先宣告EDP協定；不是已完成的結果|

完整說明：[驗證報告](../../docs/INMIN_FULL_SYSTEM_VALIDATION.md)、[操作及設計空間](../../docs/INMIN_FULL_SYSTEM_GUIDE.md)。原始事件、最佳mapping與帳本包含在 [測試證據ZIP](../../release_packages/INMIN_Full_Results_20261103_full_release.zip)。

此公開快照建立時，正式DNN對照仍在本機執行。RL是否優於同預算SA／Random、是否接近原版Gemini，不能由這些工程測試推論。

## ResNet18已完成的共享前置階段

[50布局數值報告](R18_SHARED_SCREEN.md) · [全部50點CSV](r18_shared_screen.csv) · [完整選中mapping／指標JSON](r18_shared_screen_summary.json)。這部分已完成52個full PPA（reference／seed兩次＋50布局），屬架構與固定內層搜尋，與尚在執行的RL精修分開。報告同時列出EDP最佳及事後滿足原PPA上限的whole-design點，沒有拼湊不同設計的最佳指標。
