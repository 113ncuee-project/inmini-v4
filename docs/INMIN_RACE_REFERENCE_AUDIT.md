# RACE-RL 參考檔核對與可移植範圍

2026-11-04。這份核對只讀取組員提供的程式與報告，沒有執行其訓練、安裝套件或將對方數字當成本系統成果。對方完整程式及報告原文不會隨本專案打包公開。

## 來源

|檔案|SHA256|
|---|---|
|20261012_RACE-RL-research-race-rl (2).zip，44,687,940 bytes，2484 entries|6a5cf4409c0d4349ae54c5d6606bc50c7996b9a60285ff39c572d209973ce1d1|
|RACE-RL_實驗研究報告_實際PPA與四偏好.docx|3e1b0fdac24c9744a787795a20b54de9294b60656703cbed51b27f21df888557|

主要核對位置：`src/race_rl/policy.py`、`accelerators/cnn_policy.py`、`training.py`、`docs/research/rl_architecture.md`、`hypothesis_results.md`、`cnn_hierarchy.md`、`tools/inmin_comparison/README.md`。ZIP為11月4日版本；Word文字內的整理日期為10月30日，不能假設兩者是同一次實驗。

## 演算法與物理模型不同

|項目|組員的正式 CNN RACE-RL|目前 INMIN 完整空間版本|
|---|---|---|
|學習器|Relational Encoder + conditional Actor + REINFORCE；沒有 Critic/PPO|成本拆分 Double Q，target network、replay、非terminal Bellman更新|
|一次決策|一次graph forward後，抽顆數、anchor及最多兩個相依operator的參數，組成完整transaction|從合法完整mapping交易中選一個；可含原子、複合、全域mapping及架構切換|
|訓練節奏|預設60 episodes ×64 interactions；每回合一次Adam更新|每個真實可行新PPA後24次replay SGD；SGD不是PPA次數|
|探索|每個categorical有5% uniform mixture；每次transaction都抽chiplet count|ε-greedy候選排序；已發現架構探索不足，正驗證分層refresh|
|Reward|由best的有界merit改善形成log reward；失敗與缺少成本分開處理|10 log(C_before/C_after)；完整PPA才給cost與reward，保留best和current|
|Compute|每chiplet 256 PE，16×16|每chiplet 16 PE，4×4，共享L2，無實體Core|
|Memory/mesh|外部記憶體不當作compute mesh中的IO router|每列兩IO，IO=2R；DRAM、NoP、路由與平均距離隨布局改變|
|Area|主要為compute dies|Compute、IO、外部DRAM等完整版本化成本|
|時間模型|每層node/DRAM/link需求roofline，layer barrier|實際資源calendar、競爭與buffer生命週期|
|細部mapping|四個軟體profile、固定支援的placement/residency|完整V3整數partition、PE配置/順序、FD、群組、dataflow、Fusion控制|
|Fusion|正式CNN後端沒有跨層Fusion|支援合法封閉DAG Fusion、tile與cache方案，收益須實評|
|主要報告輸入|64×64|目前ResNet18正式比較224×224|

因此不能直接比較兩邊的毫秒、W或mm²，也不能把新Double-Q改名就稱為組員原版RACE-RL。可移植的是關係表示、完整耦合transaction及明確顆數決策的設計思想；若移植，仍需在同一硬體、workload、目標與完整PPA預算下重新比較。

## 對方結果提供的證據與限制

Word第二輪包含3模型×4偏好×5種子，共60設定。RACE-RL達標60/60；但例如MobileNetV2對全域Random僅8/20配對成本較低、對Chiplet-Gym局部改編4/20較低。ResNet18對Gemini改編16/20較低。這支持可行解搜尋的研究方向，沒有支持全面勝過所有方法。

同一五個種子重複用於四個偏好，不能當作20個獨立種子。無效提案及cache hit也計入64個request，與本系統「完整PPA／timeout／cache hit／proposal／SGD分開計數」不同，須先對齊帳本定義。

ZIP中的小型H1/H2/H3消融本身明記尚未成立，沒有證實attribution或coupled edit的獨立因果收益。其INMIN比較器改用了簡化提案、特徵及reward，省略原生完整generator、DP seed與physical calendar；不能把那個適配器的落敗當成原生INMIN或論文演算法落敗。

## 本專案的實作決策

保留16PE與完整V3有效變數，不用較小的四profile空間替換。保留同一完整物理oracle，將高層architecture探索與細部mapping選擇分層；完整交易仍可同時修改相關變數。所有新增heuristic/proxy僅產生或描述候選，不能產生RL的PPA標籤。

當前開發組已測到RL比強DP起點降低EDP16.06%，但Random降低24.42%。這個負面對照必須保留。後續要驗證的是：在相同architecture refresh與相同完整PPA／時間預算下，學習後的選擇是否比Random、SA、凍結Q及均勻策略更有效。

所有PPA仍為未經RTL／晶片量測校準的解析模型；同域改編Gemini對照也不能替代原論文跨硬體效能宣稱。
