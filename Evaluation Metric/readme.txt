Evaluation Metric Module 說明

本資料夾包含誤差評估指標相關程式，主要用於計算各種 rPPG/HR 預測演算法之性能表現。
本資料與原先ToolBox相同，不同點皆在需要NIR的其他部分。

[主要檔案]
- metric.py : 提供多種標準指標計算函式，包括 MAE（平均絕對誤差）、RMSE（均方根誤差）、SUCI（成功率）、MER（平均誤差率）。
- readme.txt : 目前說明文件。

[功能簡介]
本模組可針對模型預測的心率序列（y_pred）與標準答案（y_true）進行誤差分析，協助使用者客觀比較算法表現、優化模型。

[可用指標說明]
- MAE：預測值與真實值間的絕對差值平均
- RMSE：預測誤差的均方根，對大/異常值敏感
- SUCI：預測誤差落在指定範圍（如 6 bpm）以內的比例
- MER：平均誤差率，反映誤差相對於真實值

[使用方式]
1. 在 main.py 或其他程式，import metric.py 的各函式：
   from metric import mean_absolute_error, root_mean_squared_error, success_rate, mean_error_rate
2. 呼叫函式並傳入 y_true, y_pred，即可取得各指標分數。

[範例]
mae = mean_absolute_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
suci = success_rate(y_true, y_pred, threshold=6)
mer = mean_error_rate(y_true, y_pred)

[注意事項]
- y_true 與 y_pred 必須格式與長度對齊，通常為一維時間序列 numpy array。
- 本模組可支援 RGB/NIR/深度等多模態 rPPG 方法的評估。

如需 Toolbox 整體說明或安裝教學，請參考上層目錄之 README.md。
