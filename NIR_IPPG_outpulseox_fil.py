# quick_ppg_qc_zoom.py —— 只顯示，不存檔（新增放大檢視）
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, medfilt, find_peaks
import matplotlib.pyplot as plt

# ======== 使用者設定 ========
MAT_PATH = r"C:\Users\ag133\Desktop\subject19_garage_small_motion_940\PulseOX\pulseOx.mat"

# 放大檢視模式： "time"（手動指定） 或 "auto"（自動尋找最穩定片段）
ZOOM_MODE = "auto"         # "time" 或 "auto"
ZOOM_T0   = 10.0           # ZOOM_MODE="time" 時：起點秒數
ZOOM_DUR  = 8.0            # 放大區間長度（秒）

# 頻帶、峰值偵測等參數
BP_LO, BP_HI = 0.7, 4.0    # 心跳頻帶（約 42–240 bpm）
ORDER = 3
PEAK_MIN_DISTANCE_SEC = 0.35   # 峰與峰最小間距（秒）~ 170 bpm 上限
MEDFILT_SEC = 0.2              # median filter 視窗秒數（去除尖峰/離群）
WIN_HR_SEC = 8.0               # 視窗 HR 計算的視窗長度
WIN_STEP_SEC = 2.0             # 視窗 HR 計算的步進

# ======== 載入資料與基礎處理 ========
m = loadmat(MAT_PATH)
sig = np.squeeze(m["pulseOxRecord"]).astype(float)
t   = np.squeeze(m["pulseOxTime"]).astype(float)

# 取相對秒並估 fs
t = t - t[0]
dt = np.diff(t); dt = dt[(dt > 0) & np.isfinite(dt)]
fs = 1.0 / np.median(dt)
print(f"[INFO] len={sig.size}, fs≈{fs:.3f} Hz, range=({sig.min():.1f},{sig.max():.1f})")

# 1) 移除離群尖峰（median filter）
def _odd(n):  # 取最近的奇數
    n = int(round(n))
    return n if n % 2 == 1 else n + 1
med_kernel = max(3, _odd(MEDFILT_SEC * fs))
sig_med = medfilt(sig, kernel_size=med_kernel)

# 2) 去直流並帶通 0.7–4 Hz
def bandpass(x, fs, lo=0.7, hi=4.0, order=3):
    b, a = butter(order, [lo/(fs*0.5), hi/(fs*0.5)], btype="band")
    return filtfilt(b, a, x - np.mean(x))
sig_bp = bandpass(sig_med, fs, BP_LO, BP_HI, ORDER)

# 3) 偵測峰值並估 HR
min_dist = max(1, int(round(PEAK_MIN_DISTANCE_SEC * fs)))
peaks, _ = find_peaks(sig_bp, distance=min_dist)
ibi = np.diff(t[peaks])                   # RR/PP 間隔
hr_inst = 60.0 / ibi if ibi.size else np.array([])  # 每搏心率
hr_avg = float(np.nanmean(hr_inst)) if hr_inst.size else np.nan
print(f"[HR] avg ≈ {hr_avg:.1f} bpm (peak-based)")

# 4) 視窗 HR（WIN_HR_SEC 滑動）
win = int(WIN_HR_SEC * fs)
step = int(WIN_STEP_SEC * fs)
hr_win_t, hr_win_v = [], []
if len(sig_bp) >= win:
    for i in range(0, len(sig_bp) - win, step):
        seg_t = t[i:i + win]
        seg = sig_bp[i:i + win]
        pk, _ = find_peaks(seg, distance=min_dist)
        if len(pk) > 1:
            ibi_seg = np.diff(seg_t[pk])
            if np.all(np.isfinite(ibi_seg)) and np.median(ibi_seg) > 0:
                hr_win_t.append(seg_t[0] + (seg_t[-1] - seg_t[0]) / 2)
                hr_win_v.append(60.0 / np.median(ibi_seg))
hr_win_t = np.array(hr_win_t); hr_win_v = np.array(hr_win_v)

# ======== 自動尋找最穩定視窗（可用於 ZOOM_MODE="auto"）========
def pick_stable_window(t, x_bp, fs, dur_sec=8.0):
    """回傳 (t0, t1, idx0, idx1, hr_med, ibi_cv) 最穩定的一段。
       以 IBI 變異係數(CV)最小、且至少有3個峰為準。"""
    L = len(x_bp)
    w = int(dur_sec * fs)
    if L < w:
        return (t[0], t[-1], 0, L, np.nan, np.inf)
    best = None
    for i in range(0, L - w, int(0.5 * fs)):  # 0.5 秒滑動
        j = i + w
        seg_t = t[i:j]
        seg = x_bp[i:j]
        pk, _ = find_peaks(seg, distance=min_dist)
        if len(pk) >= 3:
            ibi = np.diff(seg_t[pk])
            if np.all(np.isfinite(ibi)) and np.median(ibi) > 0:
                hr_med = 60.0 / np.median(ibi)
                cv = np.std(ibi) / np.mean(ibi)
                score = cv  # 越小越穩定
                if (best is None) or (score < best[-1]):
                    best = (seg_t[0], seg_t[-1], i, j, hr_med, cv)
    if best is None:  # 找不到足夠峰值就退而求其次
        return (t[0], t[min(L, w)-1], 0, min(L, w), np.nan, np.inf)
    return best

# 依模式決定放大區間
if ZOOM_MODE == "time":
    t0, t1 = ZOOM_T0, ZOOM_T0 + ZOOM_DUR
    idx0 = int(np.searchsorted(t, t0))
    idx1 = int(np.searchsorted(t, t1))
    idx0 = max(0, min(idx0, len(t)-1))
    idx1 = max(idx0+1, min(idx1, len(t)))
    t0, t1 = t[idx0], t[idx1-1]
    # 視窗 HR 與 IBI CV
    seg_t = t[idx0:idx1]
    seg = sig_bp[idx0:idx1]
    pk_z, _ = find_peaks(seg, distance=min_dist)
    if len(pk_z) >= 3:
        ibi_z = np.diff(seg_t[pk_z])
        hr_med_z = 60.0 / np.median(ibi_z)
        ibi_cv_z = float(np.std(ibi_z) / np.mean(ibi_z))
    else:
        hr_med_z = np.nan
        ibi_cv_z = np.inf
elif ZOOM_MODE == "auto":
    t0, t1, idx0, idx1, hr_med_z, ibi_cv_z = pick_stable_window(t, sig_bp, fs, ZOOM_DUR)
else:
    raise ValueError("ZOOM_MODE 需為 'time' 或 'auto'")

# 取放大視窗資料
tz = t[idx0:idx1]
sig_raw_z = sig[idx0:idx1]
sig_bp_z = sig_bp[idx0:idx1]
pk_z, _ = find_peaks(sig_bp_z, distance=min_dist)

# ======== 繪圖 ========
plt.figure(figsize=(12, 4))
plt.plot(t, sig, lw=0.8)
plt.title("Raw Pulse Oximeter (with baseline & spikes)")
plt.xlabel("Time (s)"); plt.ylabel("PPG")

plt.figure(figsize=(12, 4))
plt.plot(t, sig_bp, lw=0.8, label="Band-passed")
plt.plot(t[peaks], sig_bp[peaks], "o", ms=3, label="Peaks")
if hr_win_t.size:
    ax2 = plt.gca().twinx()
    ax2.plot(hr_win_t, hr_win_v, ".", alpha=0.6, label="HR (8s window)")
    ax2.set_ylabel("HR (bpm)")
    plt.legend(loc="upper right")
plt.title(f"Cleaned PPG  |  avg HR ≈ {hr_avg:.1f} bpm")
plt.xlabel("Time (s)"); plt.ylabel("PPG (a.u.)")
plt.tight_layout()

# 放大視窗（原始 + 濾波 + 峰值 + 逐搏標示）
plt.figure(figsize=(12, 4.8))
plt.plot(tz, sig_raw_z, lw=0.7, alpha=0.6, label="Raw")
plt.plot(tz, sig_bp_z, lw=1.2, label="Band-passed")
plt.plot(tz[pk_z], sig_bp_z[pk_z], "o", ms=4, label="Peaks")

# 標示逐搏 IBI
if len(pk_z) >= 2:
    for a, b in zip(pk_z[:-1], pk_z[1:]):
        t_a, t_b = tz[a], tz[b]
        plt.plot([t_a, t_b], [sig_bp_z[a], sig_bp_z[b]], ":", lw=0.8)
        plt.text((t_a + t_b)/2, max(sig_bp_z[a], sig_bp_z[b]),
                 f"{(t_b - t_a)*1000:.0f} ms", fontsize=8, ha="center", va="bottom")

stable_txt = f"HR≈{hr_med_z:.1f} bpm, IBI-CV={ibi_cv_z:.2f}" if np.isfinite(ibi_cv_z) else "HR/IBI-CV 不足以估計"
plt.title(f"ZOOM [{t0:.2f}s ~ {t1:.2f}s]  |  {stable_txt}")
plt.xlabel("Time (s)"); plt.ylabel("PPG (a.u.)")
plt.legend(loc="upper right")
plt.tight_layout()

plt.show()
