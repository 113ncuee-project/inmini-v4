# 組員閱讀入口：程式、實驗與論文證據

目前可執行完整空間版本為Full v0.8；新學習器原型另外凍結驗證。先依根目錄README安裝Python3.12依賴，再使用GUI或夜間腳本。實驗由本機程式執行，不需要Codex或LLM持續連線。所有報告使用文字、JSON、數字表格。

## 建議閱讀順序

1. [操作與完整設計空間](INMIN_FULL_SYSTEM_GUIDE.md)：輸入、安裝、長跑、停止／續跑、硬限制、可支援與未實作項目。
2. [驗收證據](INMIN_FULL_SYSTEM_VALIDATION.md)：原始來源hash、測試覆蓋、實際CLI／GUI與精確恢復。
3. [分層搜尋及完整數字](INMIN_STRATIFIED_SEARCH.md)：共享起點、RL／Random／SA／凍結／均勻控制與面積功率取捨。
4. [RL有沒有學到](INMIN_POLICY_LEARNING_RESULTS.md)：共享後與線上訓練後的相同未訓練設計排序；同架構／分組內檢查。
5. [RACE-RL參考核對](INMIN_RACE_REFERENCE_AUDIT.md)：與同學原生RACE的演算法／硬體差異。對方完整ZIP／Word並未隨專案公開。

## 主要程式在哪裡

下列路徑以儲存庫根目錄為起點。

|檔案|責任|
|---|---|
|`simplified_rapidchiplet/full_system_dse.py`|共同初始化、選候選、完整評估、reward、replay、checkpoint及各方法預算|
|`simplified_rapidchiplet/full_system_gui.py`|本機GUI與API，將使用者設定傳入同一搜尋引擎|
|`run_inmin_full_overnight.ps1`|多模型／多RNG串行執行、共用起點、保持喚醒、原設定續跑|
|`simple_rapidchiplet/hybrid/full_space.py`|所有V3有效設計變數的提案／特徵與完整交易|
|`hybrid/architecture_refresh.py`|顆數分層轉移；這是外層非學習協調，所有方法共用|
|`hybrid/cost_value_q.py`|成本拆分Double Q、物理成本輔助學習、target／replay|
|`hybrid/single_array_seed_v3.py`|容量感知Tangram-style連續segment DP／beam起點|
|`scripts/refine_full_start.py`、`build_strong_shared_start.py`|較寬DP／內層mapping候選，完整PPA重新排名與共同計帳|
|`hybrid/single_array_compiler.py`|16PE／共享L2／精度／ownership／Fusion的實體編譯|
|`hybrid/single_array_communication.py`|按實際消費cohort拆分讀取，避免免費跨群組／tile駐留|
|`hybrid/single_array_evaluator.py`|資源排程、buffer lifetime、完整延遲／能量／功率／面積|
|`hybrid/native_calendar.cpp`|相同物理定義的可選C++calendar計算核心|
|`scripts/analyze_full_system.py`|來源／功率能量／reward／查詢／機率／推薦可行性審計|
|`research_tools/analyze_policy_learning.py`|獨立診斷成本排序，0新增PPA／SGD|
|`research_tools/extend_full_run.py`|在新目錄延長已完成研究，保留所有原查詢／時間／policy|

表中`hybrid/`均位於`simplified_rapidchiplet/simple_rapidchiplet/`；`scripts/`位於`simplified_rapidchiplet/`。硬體profile與歷史入口保留，不應把舊4Core數值當成新16PE架構。

## 一次實驗在做什麼

初始化先用Tangram-style DP／beam、架構與inner候選建立共同已評估archive；RL、SA、Random從同一個最強可行起點開始。這些共同改善全部算前置收益，不算RL收益。

每步先提出有限合法候選。RL依Q與探索機率選擇，Random在同一池均勻抽取，SA用完整成本決定接受與否。候選經compiler、communication、calendar、buffer capacity檢查得到完整PPA；只有完整可行結果才有物理cost reward。timeout不等於完整不可行，也沒有虛構成本。

RL保留current與best，允許有限較差中間狀態；Random每次回到best。這個軌跡差異會影響結果，必須由同軌跡均勻控制與重啟條件對照處理。候選生成器本身包含強啟發式／DP，因此Random也可能找到很好的設計。

共同與搜尋完整PPA、失敗、timeout、cache hit、候選數、SGD、線性擬合數是不同帳目。每個實驗保存凍結來源、輸入hash、環境、逐步記錄、完整最佳mapping與兩代policy；恢復使用原來源。延長同一RNG不能當成另一個種子。

## 論文目前可以怎麼說

本系統在完整V3有效設計空間內確實找到比共同強DP起點更低的EDP；結果同時涉及mapping與架構，應一併列出面積、平均功率、Latency與Energy。已完成研究包含RL勝出與Random勝出的種子，尚不能宣稱普遍穩定優勢。

「網路有更新」、「未訓練成本排序變好」、「比無學習的策略更有效」、「比原版論文更好／快」是四種不同層級的主張。前兩者有部分證據與退步案例；後兩者需要相同來源／同預算的完整控制及更多確認。

目前硬體為未經晶片校準的解析模型，M／T假設見規格與輸出profile；Tangram-style與Gemini-style／SA是同域改編，沒有重現原論文全部二進位實驗。Fusion也有流量較多、EDP較差的實評配對，不能當成必然收益。

新的固定先驗／線上殘差Double Q已完成小型測試，正在排隊進行完整驗收與新PPA。請依對應來源與完成標記判讀；不要將原型或排入佇列寫成已完成的論文結果。
