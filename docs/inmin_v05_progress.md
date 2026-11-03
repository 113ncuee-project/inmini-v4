# INMIN Full v0.5 進度

## ACTIVE HANDOFF — 2026-11-03 22:50 Asia/Taipei

完整系統已完成；最終154項回歸、實際GUI8個Toy PPA、50布局共同實際PPA及三方法stop/resume通過。詳見 `INMIN_FULL_SYSTEM_VALIDATION.md`。目前開始正式DNN研究，不可把「程式完成」混成「RL穩定勝出」。

- Live／night APP來源identity：`336f979b54946ba967d3c16dcf56b9aa7fca3bb43dc6aa304b440540d3858933`。**夜間佇列期間不要改APP或wrapper**，避免第二RNG的cohort檢查拒絕。可以更新根目錄docs與打包；不可改frozen source。
- 主PowerShell launcher PID32956（2026-11-03 22:45:28啟動）；`night_edp` 已進架構篩選，RNG7→17，RL／SA／Random，各256full上限／2400s，EDP，oracle180s，共同screen1800s。日誌：`output/inmin_v05_20261103/night_launcher.out.log`和`.err.log`。不要重複啟動同一工作。
- 後置一次性工作PID33416（22:56:15）正在等主launcher結束；腳本 `tmp/inmin_v05/crosscheck_and_package_after_night.ps1`。原僅等待的29296已按啟動時間核對後替換，主實驗未停。之後R50／Mobile各最多12full固定2×1相容性檢查，再自動輸出 `INMIN_Full_20261104_morning_package.json`與`OVERNIGHT_COMPLETE.json`；完成標記會核對六個主研究方法與審計，有失敗則明記completed_with_failures。它不恢復每小時automation。
- 主要新入口 `full_system_dse.py`＋`full_system_gui.py`；MultiStepQ gamma=.9、target network／replay／148維含目標DAG位置，真實cost delta reward。完整v3變數在同一流程，Batch/Microbatch仍按v3物理契約=1。
- `acceptance_shipping`154/0/0/source-stable；`workflow_shipping`186.613s passed（含50實際布局與共同計費）；GUI `output/full_system_gui/20261103_224040_1ec992` 8full、審計0/0。GUI與交付及night四個核心源碼hash相同。
- Balanced開發probe `r18_joint_128`已在共同screen階段合作停止（return130）；它不是完成的研究，不要resume。其negative tradeoff與durable PPA保留。
- GitHub目前準備中的隔離checkout：`tmp/github_inmini_v3_v04`，branch `codex/full-system-v05`，目標remote `inmini-v4`，遠端main原84dac0b。新版尚需封裝最終docs／測試證據再推送；不可改原workspace舊origin0917或破壞dirty worktree。

下一步：監看night正常／錯誤、整理可公開程式與QA ZIP並推送inmini-v4；在實際完成後分析同預算RL／SA／Random、保存負面結果和來源。主要研究與後置包若失敗，先讀stderr／status／原frozen snapshot；正常工作時不要再起重複實驗。尚不能宣稱比原版Gemini快／好。

使用者2026-11-03授權：恢復 v3 全可執行設計空間，RL仍為搜尋核心；瓶頸感知／複合動作可替換，以實際 PPA 改善及搜尋效率驗證；今夜可長跑，報告不用圖片。原每小時 automation 維持暫停，沒有使用重置券介面。

新入口：`simplified_rapidchiplet/full_system_dse.py`、`start_inmin_full_gui.cmd`、`run_inmin_full_overnight.cmd`。操作與空間表見 `INMIN_FULL_SYSTEM_GUIDE.md`。

已完成的工程檢查：50種架構結構合法／IO守恆、lazy v3交易、Fusion/tile/cache實際Oracle、Double Q學習／target／精確模型存取，以及三方法Toy端到端。第一次workflow找到飽和Spatial Group的idle數量邊界錯誤，已修復；失敗輸出保留於 `output/inmin_v05_20261103/workflow_release`。修復後 `workflow_fixed` 驗證中斷前後相同動作序列、相同best與SGD、無重複PPA、完成resume零新PPA、無可行解時推薦為null，91.902秒通過。

新版本繼續驗證：共用架構篩選只計一次物理執行，對各方法同額計帳；checkpoint保留初始與最新兩代模型；完整來源穩定regression與正式DNN實驗需讀同目錄後續驗證結果。任何開發Toy改善不能寫成正式DNN收益。

所有實驗位於 `output/inmin_v05_20261103/`。舊 v0.4 結果與凍結來源不修改。主研究尚未產生足夠證據前，不宣稱RL穩定優於SA或原版Gemini。
