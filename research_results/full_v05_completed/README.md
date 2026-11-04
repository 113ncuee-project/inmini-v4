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

另外保留完整負面對照與小型跨模型檢查，避免只上傳勝出實驗：

|資料夾|範圍與結果|
|---|---|
|`fixed_c2/7`|固定2顆、各128完整PPA；Random EDP0.16374609、SA0.16958012、RL0.17023650，Random勝出|
|`inner_only_controls`|固定原mapping，auto weight retention單獨改善約10.176%；不歸給policy|
|`fusion_pair`|Random最佳只移除Fusion，Off仍可行且EDP0.16300694；On額外讀取799,488B activation，權重／MAC相同，EDP反而高0.4534%|
|`crossmodel_smoke`|R50及Mobile各12完整PPA的執行檢查；沒有公平多方法／多RNG收益結論|

所有方法均另有 `mapping_review.json`：保留完整mapping與主要物理數字，省去重複的大型diagnostics；原 `recommendation.json`仍完整保留。原兩RNGjoint研究未設Area／平均Power／Latency硬上限，不能把較大的架構當成符合原2W／100mm²限制的推薦。

需要獨立重算最佳設計，可使用已安裝的Python環境，從repository根目錄執行下列指令。它先檢查物理來源與工作負載，取得實驗鎖，再做**一次真正完整PPA**，逐項核對Latency／Energy／Area／Power／MAC與流量；不訓練RL。請在沒有其他計時實驗時執行，輸出目錄必須是新的。

```powershell
.venv/Scripts/python.exe -X utf8 research_results/full_v05_completed/replay_best.py --seed 7 --method rl --out output/replayed_r18_rng7_rl
```

`--seed 17`、`--method sa`或`--method random`可選另外五個最佳設計。換行或物理來源已改變時會拒絕；本資料對應的完整已驗證原始程式可由commit `82fd95e`取回。[實際重放驗證](replay_validation.json)已通過：1個新增完整PPA、0SGD，78個物理來源檔案及10個物理量吻合。這個額外評估沒有算進原搜尋收益。
