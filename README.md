---
9/21
依:目前資料壓縮可能只能壓縮到第二個subject，其他的解壓縮可能需要換地發解壓縮或把壓縮完的資料先存在別的電腦裡面!!!

目前有做subject1的資料對齊，但需要再做調整，可能需要用冠廷的電腦操作看看，因為現在要調查偵率是在哪時候是一致的，目前遇到的問題是有更改參數但還是不對齊的狀態，因電腦性能關係沒辦法很準確地去檢驗到哪一偵的NIR和RGB是相同的

9/22
冠:我是從subject19逆回去做，目前空間還夠，剩下對齊。如果我完成後會和依說，但因為每個subject對齊的參數都不一樣，要反覆試才能成功。
>>到晚上23:59前都會持續的調整


---

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




