# ResNet18 完整架構篩選：已完成的前置階段

這是 RL 精修前的共享架構／內層結果。主研究 RL／SA／Random 當時仍在執行；此表不能當作 RL policy 的收益。

完整架構 PPA：50；可行：50；另有 2 次原始 reference／最佳 seed 重驗。

|Point|Architecture|Latency ms|Energy mJ|Area mm²|Power W|EDP改善%|Latency改善%|Energy改善%|Area改善%|Power改善%|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Original best Tangram-style seed|single_array_c2_2x1|11.818170|18.438092|64.794232|1.560148|0|0|0|0|0|
|Best shared EDP|single_array_c8_2x4|4.191008|13.665996|106.221720|3.260789|73.7159|64.5376|25.8817|-63.9370|-109.0052|
|Post-hoc area/latency/energy <= baseline|single_array_c4_1x4|6.017990|11.972297|53.110860|1.989418|66.9354|49.0785|35.0676|18.0315|-27.5147|
|Post-hoc all four metrics <= baseline|single_array_c3_1x3|7.696513|12.005567|46.206278|1.559871|57.5956|34.8756|34.8872|28.6877|0.0177|

後兩點是從已測 archive 事後選出、滿足明列指標上限的完整設計，沒有額外訓練或額外 PPA，不能稱為另一次有硬限制的RL訓練。第一點主要目標只有 EDP，面積及Power會增加，必須同表呈現。
全部是未校準的解析 PPA。MAC守恆已核對；完整50點見CSV，各點完整mapping見JSON。
比較起點是專案的bounded Tangram-style DP/beam，不是原版Tangram或Gemini二進位。來源與研究設定見夜間協定。
