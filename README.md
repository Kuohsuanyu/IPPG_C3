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
