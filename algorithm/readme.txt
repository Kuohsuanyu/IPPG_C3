本目錄存放 rPPG 相關經典演算法，每個 .py 檔代表一種脈搏信號萃取方法，適用於生理訊號估算與性能比較。

[主要檔案簡介]
- GREEN.py      : 以綠光通道為基礎的 rPPG 信號萃取。
- ICA_POH.py    : Poh 等提出之獨立成分分析（ICA）處理方法。
- LGI.py        : 基於 Local Gradient Information 的 rPPG 方法。
- PBV.py        : 基於 Pulse Band Variation 信號的演算法。
- POS_WANG.py   : Wang 等提出之 POS（Plane-Orthogonal-to-Skin）模型。
- yi chrom.py   : CHROM 演算法實作。

[功能說明]
本目錄演算法可與資料前處理（dataprocess）搭配，將臉部 ROI 及通道值傳入演算法函式，萃取 rPPG 原始信號，再進行心率或其他生理訊號分析。

[使用方式]
1. 在主程式(import)調用：
   from algorithm.GREEN import green_method
   rppg_signal = green_method(rgb_channels)

2. 可依需求批次測試不同演算法，呼叫performance評估模組計算誤差指標。
