"""POS
Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). 
Algorithmic principles of remote PPG. 
IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
"""
import math
import numpy as np
import math
from scipy import signal

def simple_detrend(signal, Lambda=100):
    return signal - np.mean(signal)

def _process_video(frames):
    RGB = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
    return np.asarray(RGB)

def POS_WANG(frames, fs):
    WinSec = 1.6  # 設定滑動視窗長度（秒）
    # 1. 影像預處理：計算每幀 RGB 平均值，得到 (N, 3)
    RGB = _process_video(frames)
    N = RGB.shape[0]  # 幀數
    l = math.ceil(WinSec * fs)  # 視窗長度（幀數）
    BVP = []
    # 2. 滑動視窗處理
    for n in range(l, N):
        # 3. 對視窗內的 RGB 做標準化（消除光照影響）
        Cn = np.true_divide(RGB[n-l:n, :], np.mean(RGB[n-l:n, :], axis=0))
        Cn = np.mat(Cn).H  # 轉置為 (3, l)
        # 4. 矩陣線性組合，強化脈搏訊號
        S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
        # 5. 動態權重組合兩組訊號
        h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
        mean_h = np.mean(h)
        h = h - mean_h  # 去除均值
        BVP.append(h[-1])  # 只取每個視窗最後一點

    BVP = np.array(BVP)  # 轉為一維陣列
    BVP = simple_detrend(BVP)  # 6. 去趨勢處理
    # 7. 設定帶通濾波器（0.75~3 Hz，心率範圍）
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))  # 8. 濾波
    return BVP  # 輸出一維 BVP 波形


if __name__ == "__main__":
    frames = np.random.randint(0, 255, (300, 36, 36, 3), dtype=np.uint8)
    fs = 30
    result = POS_WANG(frames, fs)
    print("POS_WANG結果", result)

# 加在這裡，畫出 BVP 波形
    import matplotlib.pyplot as plt
    plt.plot(result[0][0])
    plt.title("POS_WANG 提取的 BVP 波形")
    plt.xlabel("Frame")
    plt.ylabel("Amplitude")
    plt.show()


