import numpy as np

def chrom_method(rgb_signal):
    """
    根據 CHROM 方法計算 rPPG 信號
    參數:
      rgb_signal: numpy array, shape=(N,3), 每行為 [r, g, b] 平均值
    返回:
      rppg_signal: numpy array, shape=(N,), 計算後的 rPPG 信號
    """
    r = rgb_signal[:, 0]
    g = rgb_signal[:, 1]
    b = rgb_signal[:, 2]
    
    X = 3 * r - 2 * g
    Y = 1.5 * r + g - 1.5 * b
    
    # 中心化及標準化
    X_norm = (X - np.mean(X)) / np.std(X)
    Y_norm = (Y - np.mean(Y)) / np.std(Y)
    
    # rPPG 信號合成
    S = X_norm - Y_norm
    
    return S
