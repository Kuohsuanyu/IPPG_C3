"""
LGI：Local Group Invariance 方法

來源：
Pilz et al. 的 CVPR 工作坊論文，利用局部梯度信息提升心率估計的穩定性。

方法概述：
通過奇異值分解(SVD)將資料降維，並基於梯度不變特性重建信號以增強脈搏訊號。

"""

import math
import numpy as np
from scipy import linalg
from scipy import signal
from unsupervised_methods import utils

def LGI(frames):
    """
    LGI 方法核心流程：

    參數：
    - frames：多幀影像數組。

    返回：
    - bvp：增強處理後的脈搏血容量信號序列（一維 numpy array）。

    詳細步驟：
    1. 以 utils.process_video 將影像轉換成 RGB 時間序列。
    2. 對時序執行奇異值分解(SVD)，取得主要特徵向量 U。
    3. 根據矩陣投影和局部梯度 invariance 理論，計算對偶矩陣 P。
    4. 投影原始數據至消除干擾的子空間，強化脈搏訊號。
    5. 從投影結果 Y 中選擇脈搏信號通道並展平成一維輸出。

    輸出波形適合後續心率估計。
    """

    precessed_data = utils.process_video(frames)  # 時間序列 RGB 數據
    U, _, _ = np.linalg.svd(precessed_data)       # 奇異值分解，分解出特徵矩陣

    S = U[:, :, 0]                                # 取得主特徵向量
    S = np.expand_dims(S, 2)                      # 擴維以便矩陣乘法

    SST = np.matmul(S, np.swapaxes(S, 1, 2))      # 計算 S*S^T

    p = np.tile(np.identity(3), (S.shape[0], 1, 1)) # 主張單位矩陣
    
    P = p - SST                                    # 計算投影矩陣

    Y = np.matmul(P, precessed_data)               # 投影至消除干擾的空間，強化訊號

    bvp = Y[:, 1, :]                               # 取脈搏信號通道
    bvp = bvp.reshape(-1)                          # 打平成一維訊號

    return bvp
