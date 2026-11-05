# 使用者 PPA preference：搜尋目標與續跑契約

使用者偏好會決定種子排名、架構／mapping篩選、RL訓練的cost與reward、最佳設計及最終推薦。EDP是額外的研究目標，不能代替四種PPA偏好。低平均功率與低能量分開處理。

|偏好|命令值|Latency權重|Area權重|平均Power權重|
|---|---|---:|---:|---:|
|均衡|balanced|1/3|1/3|1/3|
|低延遲|low_latency|0.8|0.1|0.1|
|低平均功率|low_power|0.1|0.1|0.8|
|小面積|small_area|0.1|0.8|0.1|

這些預設與原V3 `rl_dse.PPA_PREFERENCES`相同。成本為三個指標相對固定共同reference的加權和。每個架構使用同一reference，不會個別重新正規化。Balanced及舊EDP／純Latency／純Energy公式逐bit保留，避免改寫歷史結果。

|額外研究目標|成本|
|---|---|
|edp|Energy比值×Latency比值|
|latency|Latency比值|
|energy|Energy比值|

RL reward仍為 `10×ln(Cbefore/Cafter)`，使用當次實驗的成本目標。候選仍經同一完整物理compiler、communication、calendar與buffer檢查；沒有因偏好而虛構PPA。元件權重不需要加進每個policy輸入：目前每次實驗只訓練一個固定偏好，並沒有聲稱一個policy可以即時切換所有權重。

GUI列出四個PPA預設及三個研究目標；命令列可使用：

```powershell
.venv/Scripts/python.exe -u -X utf8 simplified_rapidchiplet/full_system_dse.py --out output/r18_low_power --preference low_power --cost-value-q --residual-cost-q --queries 612 --seconds 7200 --eval-timeout 600 --max-power 3.5 --max-area 120
```

`--objective low_power`與`--preference low_power`等價；兩者若互相矛盾會拒絕。多種子wrapper以 `-Objective low_power` 選擇；需要不同偏好時，分別使用新的輸出目錄。為了舊腳本相容，未指定的CLI研究目標仍是EDP；新版GUI預設顯示Balanced。

Power／Area／Latency硬上限獨立於權重。搜尋可以把超標設計當成帶懲罰的中間狀態，但`recommendation.json`只能推薦完整可行且符合所有硬上限的設計；找不到時清楚記錄沒有合法推薦。低延遲權重不會取消功率或面積上限。

新實驗的`objective.json`記錄版本、精確權重、固定reference和硬上限。checkpoint、summary及recommendation保存同一份契約；續跑拒絕不相符的目標、上限或reference。改偏好必須開新實驗，不能把已學EDP的checkpoint重新命名成低功率policy。共同PPA archive也核對目標與硬限制，不會直接混用不同偏好的成本標籤。

目前長時間確認研究使用EDP，保留原凍結來源及協定。其EDP改善不代表低功率、小面積或Balanced搜尋也已證明改善；特別是現有最佳EDP解有面積與功率增加的取捨。四種PPA偏好的功能驗收與未完成的DNN實驗必須分開呈現，驗收進度見`INMIN_FULL_SYSTEM_VALIDATION.md`。
