# 完整空間RL：已完成的兩RNG研究

[數字報告](REPORT.md) · [表格CSV](results.csv) · [結構化JSON](summary.json)。RNG7、17的RL／SA／Random六方法都已完成，來源相同，各組audit通過。這是來源336的開發證據；新graph-edit／強起點實驗尚待完成，這裡沒有預填其收益。

|RNG|RL EDP|Random EDP|SA EDP|同查詢預算RL比Random下降%|
|---:|---:|---:|---:|---:|
|7|0.06109825|0.06634240|0.07943955|17.837|
|17|0.06012576|0.06324071|0.07690129|12.925|

RL分別新增141、204次完整PPA；共同階段52次同額計費。相同wall-time改善與到達品質門檻的時間另見報告。最終總EDP不能直接與原論文不同硬體數字比較。

**較弱起點的風險仍待排除。** RL改變了大量group邊界，原始DP有max-group限制；下一輪會先給所有方法同一個允許全長segment的DP／inner強起點，再驗證額外收益。原seed到8顆Compute共同起點的約73.7%收益發生在RL前，且增加面積與平均功率，不能歸給RL。

每個`7`／`17`資料夾包含共同起點、來源／輸入hash、有效設定、equal-budget審查、完整最佳mapping及三個壓縮的原始事件帳本。`rl/learning.jsonl`是實際更新紀錄，SGD不計成新PPA。壓縮事件可用Python標準庫`gzip.open(..., 'rt', encoding='utf-8')`讀取，包含當時候選機率、Q、選中交易、物理reward與計費。沒有額外PPA或policy更新用於產生本匯出。

`held_out_policy*`檢查未用於RL訓練的控制臂候選，其結果僅支持即時reward輔助head排序改善，不證明future-Q或學習造成搜尋優勢。兩RNG不足以建立統計上的普遍優勢。PPA均是未校準解析估計；SA是同域INMIN對照，沒有執行原版Gemini二進位程式。

`manifest.json`提供逐檔SHA256。`export_completed.py <workspace>`從完成且audit通過的本機原始輸出重建這個匯出；不操作實驗。
