"""PBV
Improved motion robustness of remote-ppg by using the blood volume pulse signature.
De Haan, G. & Van Leest, A.
Physiol. measurement 35, 1913 (2014)
"""

import math
import numpy as np
from scipy import linalg
from scipy import signal

class utils:
    @staticmethod
    def process_video(frames):
        # 將 frames: (num_frames, H, W, 3) 轉為 (1, 3, num_frames)
        # 平均每一幀的 ROI，回傳 shape (1, 3, num_frames)
        num_frames = frames.shape[0]
        rgb_means = []
        for i in range(num_frames):
            frame = frames[i]
            rgb_means.append(np.mean(frame, axis=(0, 1)))
        rgb_means = np.array(rgb_means)  # (num_frames, 3)
        rgb_means = rgb_means.T  # (3, num_frames)
        return rgb_means[np.newaxis, :, :]  # (1, 3, num_frames)

def PBV(frames):
    # 1. 影像預處理：將每一幀的 ROI 計算 RGB 平均值，回傳 shape (1, 3, T)
    precessed_data = utils.process_video(frames)
    # 2. 計算每個通道的平均值 (1, 3)
    sig_mean = np.mean(precessed_data, axis=2)

    # 3. 各通道標準化 (1, T)
    signal_norm_r = precessed_data[:, 0, :] / np.expand_dims(sig_mean[:, 0], axis=1)
    signal_norm_g = precessed_data[:, 1, :] / np.expand_dims(sig_mean[:, 1], axis=1)
    signal_norm_b = precessed_data[:, 2, :] / np.expand_dims(sig_mean[:, 2], axis=1)

    # 4. 計算每個通道的標準差 (3, 1)
    pbv_n = np.array([
        np.std(signal_norm_r, axis=1),
        np.std(signal_norm_g, axis=1),
        np.std(signal_norm_b, axis=1)
    ])
    # 5. 計算三通道變異量平方和的平方根 (1,)
    pbv_d = np.sqrt(
        np.var(signal_norm_r, axis=1) +
        np.var(signal_norm_g, axis=1) +
        np.var(signal_norm_b, axis=1)
    )
    # 6. 組合 PBV 特徵 (3, 1)
    pbv = pbv_n / pbv_d

    # 7. 將標準化後的 RGB 組成矩陣 (1, 3, T)
    C = np.swapaxes(np.array([signal_norm_r, signal_norm_g, signal_norm_b]), 0, 1)
    # 8. 矩陣轉置 (1, T, 3)
    Ct = np.swapaxes(np.swapaxes(np.transpose(C), 0, 2), 1, 2)
    # 9. 計算協方差矩陣 (1, 3, 3)
    Q = np.matmul(C, Ct)
    # 10. 解線性方程，取得權重 (1, 3)
    W = np.linalg.solve(Q, np.swapaxes(pbv, 0, 1))

    # 11. 利用權重與標準化資料融合脈搏波形 (1, T, 1)
    A = np.matmul(Ct, np.expand_dims(W, axis=2))
    B = np.matmul(
        np.swapaxes(np.expand_dims(pbv.T, axis=2), 1, 2),
        np.expand_dims(W, axis=2)
    )
    bvp = A / B
    # 12. 輸出一維 BVP 波形 (T,)
    return bvp.squeeze(axis=2).reshape(-1)

if __name__ == "__main__":
    # 產生隨機影像資料 (100 幀, 36x36, 3通道)
    frames = np.random.randint(0, 255, (100, 36, 36, 3), dtype=np.uint8)
    result = PBV(frames)
    print(result)

    # 繪製 BVP 波形
    import matplotlib.pyplot as plt
    plt.plot(result)
    plt.title("PBV 提取的 BVP 波形")
    plt.xlabel("Frame")
    plt.ylabel("Amplitude")
    plt.show()
