"""
ICA_POH.py - ICA (Independent Component Analysis, 獨立成分分析) 實作

參考論文：
Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010).
"Non-contact, automated cardiac pulse measurements using video imaging and blind source separation."
Optics express, 18(10), 10762-10774.

此程式碼目標是使用 ICA 方法，從RGB影像序列中分離獨立的脈搏信號成分（BVP）。

運算流程含：
- 影像轉時序RGB信號
- 去趨勢正規化
- ICA 分離
- 頻譜功率判斷選擇脈搏成分
- 帶通濾波清理訊號
"""


import math
import numpy as np
from scipy import linalg, signal
from unsupervised_methods import utils  # 依賴自定義函式庫，含去趨勢等


def ICA_POH(frames, FS):
    """
    ICA_POH 主入口函式。

    參數：
    frames - (ndarray) RGB 影像序列（時間序列）
    FS     - (float) 影像取樣頻率（Hz / fps）

    返回：
    BVP    - 處理後的血容積脈搏信號（numpy array）

    Step:
    1. 頻率範圍設定（0.7~2.5 Hz）對應心臟生理訊號頻率
    2. 將影像序列轉為RGB三通道平均強度時間序列
    3. 對三時間序列進行去趨勢與標準化
    4. ICA分離信號為3個獨立成分
    5. 利用功率譜判定脈搏相關成分
    6. 使用帶通濾波器濾除雜訊得到最終 BVP
    """

    # 定義心率合理頻率範圍Hz
    LPF = 0.7
    HPF = 2.5

    # 轉影像至RGB時間序列 (N 幀 x 3 通道)
    RGB = process_video(frames)

    NyquistF = FS / 2  # 奈奎斯特頻率 = 取樣率一半

    BGRNorm = np.zeros(RGB.shape)  # 儲存正規化資料
    Lambda = 100  # 去趨勢參數，控制平滑程度

    # 對每個通道單獨去趨勢並標準化
    for c in range(3):
        BGRDetrend = utils.detrend(RGB[:, c], Lambda)
        BGRNorm[:, c] = (BGRDetrend - np.mean(BGRDetrend)) / np.std(BGRDetrend)

    # 對標準化後資料執行ICA，取得分離成分矩陣 S (3 x N)
    _, S = ica(np.mat(BGRNorm).H, 3)

    # 判斷哪個分量是脈搏信號
    MaxPx = np.zeros((1, 3))  # 紀錄3個分量的最大功率

    for c in range(3):
        FF = np.fft.fft(S[c, :])  # 傅立葉頻譜
        FF = FF[1:]  # 去直流分量
        N = FF.shape[0]

        Px = np.abs(FF[:math.floor(N / 2)]) ** 2  # 功率譜前半部分
        Px = Px / np.sum(Px)  # 正規化功率譜

        MaxPx[0, c] = np.max(Px)  # 取最大功率值

    # 選擇最大功率成分索引，認為該分量代表血容積訊號
    MaxComp = np.argmax(MaxPx)
    BVP_I = S[MaxComp, :]

    # 設計3階巴特沃斯帶通濾波器，篩選合理心率頻率範圍
    B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], btype='bandpass')

    # 零相位濾波，避免相位延遲扭曲
    BVP_F = signal.filtfilt(B, A, np.real(BVP_I).astype(np.double))

    BVP = BVP_F[0]  # 回傳濾波後的脈搏信號

    return BVP


def process_video(frames):
    """
    將影像序列轉換為每一幀三個頻道的平均 RGB 強度。

    參數：
    frames - 輸入影像序列 (List[ndarray] 或 ndarray)

    返回：
    RGB - numpy array，形狀 (幀數, 3)，每個通道的平均值時間序列
    """
    RGB = []
    for frame in frames:
        sum_val = np.sum(np.sum(frame, axis=0), axis=0)
        avg_val = sum_val / (frame.shape[0] * frame.shape[1])
        RGB.append(avg_val)
    return np.asarray(RGB)


def ica(X, Nsources, Wprev=0):
    """
    ICA 執行函數，對輸入矩陣執行盲信號分離。

    參數：
    X - 輸入觀察矩陣，尺寸 (channels x samples)
    Nsources - 欲分離信號數
    Wprev - 上一次 ICA 分離矩陣初始值，0 為從頭開始

    返回：
    W - 分離矩陣
    Zhat - 分離信號矩陣
    """
    nRows = X.shape[0]
    nCols = X.shape[1]

    if nRows > nCols:
        print("Warning - The number of rows is cannot be greater than the number of columns.")
        print("Please transpose input.")

    if Nsources > min(nRows, nCols):
        Nsources = min(nRows, nCols)
        print("Warning - The number of sources cannot exceed number of observation channels.")
        print("The number of sources will be reduced to the number of observation channels ", Nsources)

    Winv, Zhat = jade(X, Nsources, Wprev)  # 呼叫 jade 進行 ICA 計算
    W = np.linalg.pinv(Winv)

    return W, Zhat


def jade(X, m, Wprev):
    """
    JADE ICA 算法，Joint Approximate Diagonalization of Eigen-matrices。

    用於盲信號分離，藉由對四階累積矩陣進行聯合近似對角化。

    參數：
    X - 觀察訊號矩陣，型別為 numpy matrix。
    m - 目標分離信號數量。
    Wprev - 上一次迭代矩陣，初始為 0。

    返回：
    Winv - 反轉矩陣估計。
    Zhat - 分離後純淨信號。

    核心步驟：
    1. 計算矩陣白化。
    2. 組成四階累積矩陣。
    3. 對該矩陣進行特徵值分解。
    4. 利用 Givens 旋轉進行聯合近似對角化。
    5. 返回估計分離矩陣與信號。

    註：
    JADE 是經典的 ICA 演算法之一，適用於非高斯信號分離，
    複雜數學細節詳見 Cardoso 等原始文獻。
    """

    n = X.shape[0]
    T = X.shape[1]
    nem = m
    seuil = 1 / math.sqrt(T) / 100

    # 計算白化矩陣和初步估計W
    if m < n:
        D, U = np.linalg.eig(np.matmul(X, np.mat(X).H) / T)
        Diag = D
        k = np.argsort(Diag)
        pu = Diag[k]
        ibl = np.sqrt(pu[n - m:n] - np.mean(pu[0:n - m]))
        bl = np.true_divide(np.ones(m, 1), ibl)
        W = np.matmul(np.diag(bl), np.transpose(U[0:n, k[n - m:n]]))
        IW = np.matmul(U[0:n, k[n - m:n]], np.diag(ibl))
    else:
        IW = linalg.sqrtm(np.matmul(X, X.H) / T)
        W = np.linalg.inv(IW)

    Y = np.mat(np.matmul(W, X))

    R = np.matmul(Y, Y.H) / T
    C = np.matmul(Y, Y.T) / T

    Q = np.zeros((m * m * m * m, 1))
    index = 0

    # 建立四階累積矩陣 Q，用於後續特徵分解
    for lx in range(m):
        Y1 = Y[lx, :]
        for kx in range(m):
            Yk1 = np.multiply(Y1, np.conj(Y[kx, :]))
            for jx in range(m):
                Yjk1 = np.multiply(Yk1, np.conj(Y[jx, :]))
                for ix in range(m):
                    Q[index] = np.matmul(Yjk1 / math.sqrt(T), Y[ix, :].T / math.sqrt(T)) \
                                - R[ix, jx] * R[lx, kx] - R[ix, kx] * R[lx, jx] - C[ix, lx] * np.conj(C[jx, kx])
                    index += 1

    # 矩陣變形與特徵值分解
    D, U = np.linalg.eig(Q.reshape(m * m, m * m))
    Diag = abs(D)
    K = np.argsort(Diag)
    la = Diag[K]

    M = np.zeros((m, nem * m), dtype=complex)
    Z = np.zeros(m)
    h = m * m - 1

    for u in range(0, nem * m, m):
        Z = U[:, K[h]].reshape((m, m))
        M[:, u:u + m] = la[h] * Z
        h = h - 1

    # Givens 旋轉矩陣，實現聯合近似對角化
    B = np.array([[1, 0, 0],
                  [0, 1, 1],
                  [0, 0 - 1j, 0 + 1j]])
    Bt = np.mat(B).H

    encore = 1

    if Wprev == 0:
        V = np.eye(m).astype(complex)
    else:
        V = np.linalg.inv(Wprev)

    # 主迴圈，不斷更新旋轉矩陣以接近對角化
    while encore:
        encore = 0
        for p in range(m - 1):
            for q in range(p + 1, m):
                Ip = np.arange(p, nem * m, m)
                Iq = np.arange(q, nem * m, m)
                g = np.mat([M[p, Ip] - M[q, Iq], M[p, Iq], M[q, Ip]])
                temp1 = np.matmul(g, g.H)
                temp2 = np.matmul(B, temp1)
                temp = np.matmul(temp2, Bt)
                D, vcp = np.linalg.eig(np.real(temp))
                K = np.argsort(D)
                la = D[K]
                angles = vcp[:, K[2]]

                if angles[0, 0] < 0:
                    angles = -angles

                c = np.sqrt(0.5 + angles[0, 0] / 2)
                s = 0.5 * (angles[1, 0] - 1j * angles[2, 0]) / c

                if abs(s) > seuil:
                    encore = 1
                    pair = [p, q]
                    G = np.mat([[c, -np.conj(s)],
                                [s, c]])  # Givens Rotation
                    V[:, pair] = np.matmul(V[:, pair], G)
                    M[pair, :] = np.matmul(G.H, M[pair, :])
                    temp1 = c * M[:, Ip] + s * M[:, Iq]
                    temp2 = -np.conj(s) * M[:, Ip] + c * M[:, Iq]
                    temp = np.concatenate((temp1, temp2), axis=1)
                    M[:, Ip] = temp1
                    M[:, Iq] = temp2

    # 完成旋轉矩陣更新後，計算分離矩陣及信號
    A = np.matmul(IW, V)
    S = np.matmul(np.mat(V).H, Y)

    return A, S
