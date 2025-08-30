# -*- coding: utf-8 -*-
"""
inspect_pulseox.py
用途：檢視 MR-NIRP-Car 的 PulseOX/pulseOx.mat 內容、嘗試抽取 PPG 與時間軸。
- 兼容 MATLAB v7.3 (HDF5) 與舊版 .mat
- 自動猜測鍵名：ppg/pleth/signal/pulseOxRecord 與 time/t/pulseOxTime
- 可選擇輸出 CSV、顯示波形圖

用法：
  python inspect_pulseox.py <path_to_pulseOx.mat> [--plot] [--save_csv out.csv]
  例：
  python inspect_pulseox.py r"D:\Data\Subject_01\driving_still_940\PulseOX\pulseOx.mat" --plot

依賴：
  pip install scipy h5py numpy matplotlib
"""

import argparse
import sys
import numpy as np
import os

# SciPy 舊版 .mat 讀取
try:
    from scipy.io import loadmat
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# v7.3/HDF5 讀取
try:
    import h5py
    H5PY_OK = True
except Exception:
    H5PY_OK = False


def is_mat_v73(path: str) -> bool:
    """粗略判斷是否為 v7.3（HDF5）格式。"""
    if not os.path.isfile(path):
        return False
    with open(path, 'rb') as f:
        head = f.read(8)
    # HDF5 檔案標頭常見為 \x89HDF\r\n\x1a\n
    return head.startswith(b"\x89HDF")


def _to_numpy(arr):
    """把各種型態（標量、1D/2D）轉成 1D numpy 陣列（必要時 squeeze）。"""
    if isinstance(arr, np.ndarray):
        return np.squeeze(arr).astype(np.float64)
    try:
        a = np.array(arr)
        return np.squeeze(a).astype(np.float64)
    except Exception:
        return None


def list_all_datasets_h5(h5obj, prefix=""):
    """遞迴列出 HDF5 內所有 dataset 的路徑與形狀。"""
    items = []
    def _recurse(g, p):
        for k, v in g.items():
            path = f"{p}/{k}" if p else k
            if isinstance(v, h5py.Dataset):
                items.append((path, v.shape, v.dtype))
            elif isinstance(v, h5py.Group):
                _recurse(v, path)
    _recurse(h5obj, prefix)
    return items


def load_mat_any(path: str):
    """
    回傳 (data, backend)
    - backend: 'scipy' 或 'h5py'
    """
    if is_mat_v73(path):
        if not H5PY_OK:
            raise RuntimeError("此檔案為 v7.3/HDF5，但未安裝 h5py。請先 pip install h5py")
        f = h5py.File(path, "r")
        return f, "h5py"
    else:
        if not SCIPY_OK:
            raise RuntimeError("讀取舊版 .mat 需要 scipy。請先 pip install scipy")
        mat = loadmat(path, squeeze_me=False, struct_as_record=False)
        data = {k: v for k, v in mat.items() if not k.startswith("__")}
        return data, "scipy"


def try_extract_signal_and_time_from_scipy_dict(data: dict, prefer_sig=None, prefer_time=None):
    """從 scipy 讀到的 dict 嘗試抓出 ppg 與 time/fs。"""
    # 可能的鍵名（都會做 .lower() 比對）
    signal_keys = ["ppg", "pleth", "signal", "sig", "y", "pulseoxrecord", "pulseox"]
    time_keys   = ["time", "t", "pulseoxtime"]
    fs_keys     = ["fs", "sampling_rate", "sr"]

    sig = t = fs = None
    sig_name = t_name = fs_name = None

    keys_lower = {k.lower(): k for k in data.keys()}

    # 先處理使用者指定
    if prefer_sig and prefer_sig.lower() in keys_lower:
        cand = _to_numpy(data[keys_lower[prefer_sig.lower()]])
        if cand is not None and cand.ndim == 1 and cand.size > 0:
            sig = cand
            sig_name = keys_lower[prefer_sig.lower()]
    if prefer_time and prefer_time.lower() in keys_lower:
        cand = _to_numpy(data[keys_lower[prefer_time.lower()]])
        if cand is not None and cand.ndim == 1 and cand.size > 1:
            t = cand
            t_name = keys_lower[prefer_time.lower()]

    # 找 signal
    if sig is None:
        for k in data.keys():
            kl = k.lower()
            if kl in signal_keys or any(s in kl for s in signal_keys):
                cand = _to_numpy(data[k])
                if cand is not None and cand.ndim == 1 and cand.size > 0:
                    sig = cand
                    sig_name = k
                    break

    # 找時間
    if t is None:
        for k in data.keys():
            kl = k.lower()
            if kl in time_keys or any(s in kl for s in time_keys):
                cand = _to_numpy(data[k])
                if cand is not None and cand.ndim == 1 and cand.size > 1:
                    t = cand
                    t_name = k
                    break

    # 找 fs
    for k in data.keys():
        kl = k.lower()
        if kl in fs_keys or any(s in kl for s in fs_keys):
            try:
                cand = float(np.squeeze(data[k]))
                if cand > 0:
                    fs = cand
                    fs_name = k
                    break
            except Exception:
                pass

    meta = {"signal_key": sig_name, "time_key": t_name, "fs_key": fs_name}
    return sig, t, fs, meta


def try_extract_signal_and_time_from_h5(h5f: "h5py.File", prefer_sig=None, prefer_time=None):
    """從 v7.3/HDF5 結構中猜測 ppg/time/fs 所在的 dataset。"""
    items = list_all_datasets_h5(h5f)
    sig = t = fs = None
    sig_name = t_name = fs_name = None

    def read_ds(name):
        arr = np.array(h5f[name])
        return _to_numpy(arr)

    # 使用者指定優先
    if prefer_sig and prefer_sig in h5f:
        cand = read_ds(prefer_sig)
        if cand is not None and cand.ndim == 1 and cand.size > 0:
            sig = cand
            sig_name = prefer_sig
    if prefer_time and prefer_time in h5f:
        cand = read_ds(prefer_time)
        if cand is not None and cand.ndim == 1 and cand.size > 1:
            t = cand
            t_name = prefer_time

    # 自動找 signal
    if sig is None:
        for name, shape, dt in items:
            ln = name.lower()
            if any(x in ln for x in ["ppg", "pleth", "signal", "sig", "y", "pulseoxrecord", "pulseox"]):
                cand = read_ds(name)
                if cand is not None and cand.ndim == 1 and cand.size > 0:
                    sig = cand
                    sig_name = name
                    break

    # 自動找 time
    if t is None:
        for name, shape, dt in items:
            ln = name.lower()
            leaf = ln.split("/")[-1]
            if (leaf in ["time", "t", "pulseoxtime"]) or any(x in ln for x in ["time", "pulseoxtime"]):
                cand = read_ds(name)
                if cand is not None and cand.ndim == 1 and cand.size > 1:
                    t = cand
                    t_name = name
                    break

    # 找 fs
    for name, shape, dt in items:
        ln = name.lower()
        if any(x in ln for x in ["fs", "sampling_rate", "sr"]):
            try:
                cand = float(np.squeeze(np.array(h5f[name])))
                if cand > 0:
                    fs = cand
                    fs_name = name
                    break
            except Exception:
                pass

    meta = {"signal_key": sig_name, "time_key": t_name, "fs_key": fs_name, "all_items": items}
    return sig, t, fs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat_path", type=str, help="pulseOx.mat 的路徑")
    ap.add_argument("--plot", action="store_true", help="顯示簡單波形圖（不儲存）")
    ap.add_argument("--save_csv", type=str, default=None, help="把偵測到的 (time, ppg) 存成 CSV 檔")
    ap.add_argument("--sig_key", type=str, default=None, help="明確指定 signal 的鍵名（如 pulseOxRecord）")
    ap.add_argument("--time_key", type=str, default=None, help="明確指定 time 的鍵名（如 pulseOxTime）")
    args = ap.parse_args()

    mat_path = args.mat_path
    print(f"[INFO] 讀取：{mat_path}")

    obj, backend = load_mat_any(mat_path)
    print(f"[INFO] 讀取方式：{backend}")

    sig = t = fs = None
    meta = {}

    # 列出內容
    if backend == "scipy":
        print("[INFO] 檔案變數清單：")
        for k, v in obj.items():
            shape = getattr(v, "shape", None)
            dtype = getattr(v, "dtype", None)
            print(f"  - {k:20s} shape={shape} dtype={dtype}")
        sig, t, fs, meta = try_extract_signal_and_time_from_scipy_dict(
            obj, prefer_sig=args.sig_key, prefer_time=args.time_key
        )

    else:  # h5py
        print("[INFO] HDF5 datasets：")
        items = list_all_datasets_h5(obj)
        for name, shape, dt in items:
            print(f"  - {name:40s} shape={shape} dtype={dt}")
        meta["all_items"] = items
        sig, t, fs, meta = try_extract_signal_and_time_from_h5(
            obj, prefer_sig=args.sig_key, prefer_time=args.time_key
        )

    # 顯示偵測結果
    print("\n[RESULT] 偵測鍵名：")
    print(f"  signal_key = {meta.get('signal_key')}")
    print(f"  time_key   = {meta.get('time_key')}")
    print(f"  fs_key     = {meta.get('fs_key')}")

    if sig is not None:
        print(f"[PPG] shape={sig.shape}, dtype={sig.dtype}, min={np.nanmin(sig):.4f}, max={np.nanmax(sig):.4f}")
    else:
        print("[WARN] 沒找到 PPG 波形（請用 --sig_key 指定鍵名）")

    # 組合時間軸與採樣率
    if t is not None:
        print(f"[TIME] shape={t.shape}, t[0..5]={t[:6]}")
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size > 0:
            fs_est = 1.0 / np.median(dt)
            print(f"[INFO] 由時間軸推估 fs ≈ {fs_est:.3f} Hz")
            if fs is None:
                fs = fs_est
    elif fs is not None and sig is not None:
        t = np.arange(sig.size) / float(fs)
        print(f"[TIME] 無時間軸，但偵測到 fs={fs:.3f} Hz，已生成 t[0..5]={t[:6]}")
    else:
        print("[WARN] 無時間軸與 fs，無法推估時間（僅顯示波形樣本）")

    if sig is not None:
        print(f"[HEAD] PPG 前 10 筆：{sig[:10]}")

    # 存 CSV（你若只想展示就不要帶 --save_csv 參數）
    if args.save_csv and sig is not None:
        import csv
        out = args.save_csv
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if t is not None:
                w.writerow(["time", "ppg"])
                for ti, yi in zip(t, sig):
                    w.writerow([f"{ti:.6f}", f"{yi:.9f}"])
            else:
                w.writerow(["idx", "ppg"])
                for i, yi in enumerate(sig):
                    w.writerow([i, f"{yi:.9f}"])
        print(f"[SAVE] 已輸出 CSV：{out}")

    # 畫圖（只展示不儲存）
    if args.plot and sig is not None:
        import matplotlib.pyplot as plt
        if t is not None:
            # 若為 Unix 秒（很大），轉相對秒
            t_plot = t - t[0] if np.nanmax(t) > 1e6 else t
            plt.figure()
            plt.plot(t_plot, sig)
            plt.xlabel("Time (s)")
            plt.ylabel("PPG")
            plt.title("Pulse Oximeter Signal")
            plt.tight_layout()
            plt.show()
        else:
            plt.figure()
            plt.plot(sig)
            plt.xlabel("Sample")
            plt.ylabel("PPG")
            plt.title("Pulse Oximeter Signal")
            plt.tight_layout()
            plt.show()

    if backend == "h5py":
        obj.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e)
        sys.exit(1)
