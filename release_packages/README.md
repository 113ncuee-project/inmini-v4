# INMIN 完整系統下載包

最新完整引擎（Full v0.6，2026-11-04）：

- [程式 ZIP](INMIN_Full_Code_20261104_v06_validated.zip)：205個驗證來源、完整v3有效搜尋域、多步Double Q、強起點DP／inner、GUI／夜間入口與延長工具。
- [測試與數字報告 ZIP](INMIN_Full_Results_20261104_v06_validated.zip)：173項回歸、完整CLI／GUI與續跑；原v0.5兩RNG、固定硬體負面結果、可行Fusion配對、跨模型執行檢查，以及新強起點48次PPA。
- [SHA256及CRC驗證](INMIN_Full_20261104_v06_validated_package.json)。程式2.33MB，數字證據6.28MB，均不含圖片。

新DP／inner起點已完成，最佳EDP=0.05149608，不能歸給RL。新版F/P、SA／Random、凍結／均勻控制仍在執行；不以未完成對照宣稱新技術有效。[結果與限制](../docs/INMIN_GRAPH_EDIT_RESULTS.md)及[操作指南](../docs/INMIN_FULL_SYSTEM_GUIDE.md)提供完整說明。

以下是歷史Full v0.5（2026-11-03）：

- [原v0.5 Policy重放補充包](INMIN_v05_Policy_Replay_20261104.zip)及[SHA清單](INMIN_v05_Policy_Replay_20261104_package.json)：原兩RNG的初始／共同暖身後／最終模型參數、壓縮事件與預測重算工具。先解壓上面的完整程式包，或使用Git clone；不做PPA、不訓練，也不是可續訓checkpoint。兩組已實際重算通過，輔助reward排序Spearman分別0.113→0.642、0.323→0.584，不能當成future-Q或新v0.6的收益證據。

- [程式 ZIP](INMIN_Full_Code_20261103_full_release.zip)：完整 v3 設計變數＋多步 Double Q／Replay／target network；含新GUI與夜間入口。
- [測試與數值證據 ZIP](INMIN_Full_Results_20261103_full_release.zip)：154項回歸、50個Toy布局實際PPA、三方法續跑與計帳、GUI實際8次PPA。
- [SHA256清單](INMIN_Full_20261103_full_release_package.json)。

先讀 [完整系統指南](../docs/INMIN_FULL_SYSTEM_GUIDE.md) 與 [驗證及夜間研究協定](../docs/INMIN_FULL_SYSTEM_VALIDATION.md)。此次打包時正式ResNet18多RNG對照尚在執行，不能把工程Toy結果當成DNN論文收益；後續實際結果由協定所列本機報告提供。

以下是歷史 v0.4 固定硬體基線：

- `INMIN_v04_Code_20261103.zip`：可攜的 v0.4 程式快照。
- `INMIN_v04_Results_20261103.zip`：文字、JSON 與 CSV 的數值結果摘要。
- `INMIN_v04_20261103_package.json`：完整交付清單及 SHA-256。

完整證據壓縮檔約 94 MB，包含所有中間候選、checkpoint 與歷史執行資料；為避免將不必要的大型本機資料放進公開 Git 歷史，本公開庫不包含它。可重跑程式、可攜輸入、兩個已驗證的設計建議，以及完整文字報告都已在本庫中。

請先閱讀 `../docs/INMIN_V04_LOCAL_GUIDE.md`。所有 PPA 數字都是未校準的解析模型估計，並非晶片量測。
