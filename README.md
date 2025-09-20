# IPPG_C3
專題第三組_IPPG資料

架構與對應資料夾
1.資料庫讀取，RGB、NIR、PULSEOX的資料讀取讓程式知道    ------> dataprocess
2.臉部辨識提取臉孔、去掉其餘部分                       ------> face_capture
3.演算法                                             ------> Algorithm
4.計算心跳、評估指標                                  ------> Evaluation Metric

代辦事項
可以幫我下載資料庫變成一個完整的檔案在隨身碟之類的地方，這樣開學可以拿給其他人直接複製到電腦就好


8/30
郭

添加兩個程式NIR_IPPG_outpulseox、NIR_IPPG_outpulseox_fil
路徑為
"C:\Users\ag133\OneDrive\文件\GitHub\IPPG_C3\dataprocess\NIR_IPPG_outpulseox.py"

有fil是添加了濾波器的版本，用在檢查心跳訊號是否正常，同時添加放大確認波型是否正常

添加了
C:\Users\ag133\OneDrive\文件\GitHub\IPPG_C3\dataprocess\read_RGB_NIR_OX.py
可以做到讀取RGB跟NIR影像資料的資料

列印如下
C:\Users\ag133\OneDrive\文件\GitHub\IPPG_C3\dataprocess>python read_RGB_NIR_OX.py
[NIR] 數量=7289, 解析度=640x640
[NIR] 首幀=Frame00001.pgm, 末幀=Frame07289.pgm
[RGB] 數量=3650, 解析度=640x640
[RGB] 首幀=Frame00000.pgm, 末幀=Frame03649.pgm
[NIR] 起始檔名索引 ≈ 1, [RGB] 起始檔名索引 ≈ 0
[MAT] 檔案：pulseOx.mat
[MAT] 訊號長度=3897, 值域=(128.0,255.0)
[MAT] 估計取樣率 fs≈38.855 Hz（依 pulseOxTime）
[INFO] 播放索引範圍：2 ~ 3649（offset: NIR=-1, RGB=0）

可以看到即時影像的資料，發現NIR跟RGB資料量不同，可能需要調整，後續還需要對好心跳資料
可以查找論文看有沒有寫或是閱讀說明文件看看





9/20

# IPPG_C3：專題第三組 - 遠端光電容積脈搏波(rPPG)專案

本專案實現基於 RGB/NIR 臉部影像和 PPG 心率標註之遠端光電容積脈搏訊號與心率估測，並可方便擴充多種演算法模組，適用於研究、教學和工具箱開發之用。

---
## 專案目錄結構與說明

- **dataprocess/**  
  主要為「資料讀取與前處理模組」。涵蓋 RGB/NIR 影像與脈搏標註(PulseOx)的檔案解析、序列對齊和格式轉換基礎工具。如 `read_RGB_NIR_OX.py` 支援影像與心率資料的批次讀取，後續輸出供演算法/評估模組使用。

- **face_capture/**  
  臉部偵測與 ROI 擷取相關程式。自原始影像分離出臉區或特定感興趣區（如只保留皮膚區），以提升後續脈搏訊號品質、增強演算法實效。

- **Algorithm/**  
  各種 rPPG/rHR 訊號萃取核心演算法總集合。內含 ICA、CHROM、POS、GREEN、LGI 等彩色或單一頻道信號處理方法，每種演算法以獨立 .py 管理、可直接引用。

- **Evaluation Metric/**  
  模型/演算法預測輸出誤差與指標計算模組。支援 MAE、RMSE、SUCI、MER 等標準指標，方便自動化性能比較與模型優化。

- **test/**  
  資料讀取與資料結構正確性測試腳本，涵蓋 NIR、RGB 影像及 PulseOx 標註滿足載入解析需求，利於新手及後加工流程前的基本驗證。

---
## 典型運作流程

1. **資料前處理：**  
   以 dataprocess 中之工具讀取 NIR/RGB 影像與心率標註(PulseOx)，進行格式轉換與同步對齊。  
   *注意：若 NIR/RGB 數量不一致或影像缺失，需進行時間軸插值與資料截斷對齊，以確保與心率資料長度一致。*

2. **臉部擷取與 ROI 分析：**  
   用 face_capture 模組將資料中非皮膚干擾部分剔除，產生乾淨且ROI對齊之輸入資料。

3. **演算法推理：**  
   調用 Algorithm 目錄下各算法進行 rPPG 信號萃取。可針對不同資料批次進行多組算法交互驗證。

4. **成效評估與誤差計算：**  
   透過 Evaluation Metric 模組對預測（如估測心率）與標註資料進行指標計算，比較各方法優劣。

5. **測試與驗證：**  
   新手可先於 test 目錄下執行基礎資料讀取驗證程式，排除資料路徑、格式、缺漏等基礎問題。

---
## 注意事項
- 各模組支援 NIR 與 RGB 單獨、混合輸入，亦可擴充至多模態輸出。
- 全流程預設模式為 Python 相對路徑，保證在各環境下可重現執行。
- 資料輸入前請務必完成「影像—心率信號」配對（時間軸插值/對齊）處理，否則誤差指標無意義。

---
## 參考文獻與優良實作
- 相關方法設計與同步流程是參考多篇 rPPG 領域高被引論文以及 MR-NIRP Car dataset 官方 protocol。
- 如需進階技術細節，建議查閱對應資料夾內 readme 與註釋。

---

本說明隨專案發展適時調整，請搭配各資料夾說明檔閱讀。如有任何問題請至 [GitHub Issues](https://github.com/Kuohsuanyu/IPPG_C3/issues) 發問。
