# IPPG_C3
專題第三組_IPPG資料

架構與對應資料夾
1.資料庫讀取，RGB、NIR、PULSEOX的資料讀取讓程式知道    ------> dataprocess
2.臉部辨識提取臉孔、去掉其餘部分                       ------> face_capture
3.演算法                                             ------> Algorithm
4.計算心跳、評估指標                                  ------> Evaluation Metric

8/30
添加兩個程式NIR_IPPG_outpulseox、NIR_IPPG_outpulseox_fil
路徑為
"C:\Users\ag133\OneDrive\文件\GitHub\IPPG_C3\dataprocess\NIR_IPPG_outpulseox.py"

有fil是添加了濾波器的版本，用在檢查心跳訊號是否正常，同時添加放大確認波型是否正常

