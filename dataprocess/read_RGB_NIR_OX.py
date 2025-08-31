# -*- coding: utf-8 -*-
"""
read_pgm_and_mat.py
從 IPPG_C3/dataprocess 執行，使用相對路徑讀入：
- ../raw_data/subject19/subject19_garage_small_motion_940/NIR/*.pgm
- ../raw_data/subject19/subject19_garage_small_motion_940/RGB/*.pgm
- ../raw_data/subject19/subject19_garage_small_motion_940/PulseOX/pulseOx.mat

功能：
1) 掃描並列出 NIR/RGB 影格統計（數量、解析度、檔名範圍）
2) 載入 pulseOx.mat，印出變數與（若有）取樣率
3) 簡單播放器：併排顯示 NIR|RGB，支援播放/暫停/逐幀與 offset 對齊

操作：
  Space  : 播放/暫停
  → / ←  : 下一幀 / 上一幀
  q      : 離開
"""

import re
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

# ============ 使用者可調參數（也可用 CLI 覆蓋） ============
DEFAULT_BASE = Path("..") / "raw_data" / "subject19" / "subject19_garage_small_motion_940"
DEFAULT_NIR_DIR = DEFAULT_BASE / "NIR"
DEFAULT_RGB_DIR = DEFAULT_BASE / "RGB"
DEFAULT_MAT     = DEFAULT_BASE / "PulseOX" / "pulseOx.mat"

# 預設 offset（幀對齊）：顧及你提供的檔名差一幀：NIR從 Frame00001、RGB 從 Frame00000
DEFAULT_NIR_OFFSET = -1
DEFAULT_RGB_OFFSET = 0

# 播放速度（毫秒/幀）；暫停時不會作用
DEFAULT_DELAY_MS = 30


def natural_key(p: Path):
    """用於自然排序 Frame00001 這種數字部分。"""
    s = p.name
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def scan_frames(dir_path: Path):
    files = sorted(dir_path.glob("*.pgm"), key=natural_key)
    if not files:
        raise FileNotFoundError(f"找不到 PGM 檔於：{dir_path}")
    # 試讀一張取得解析度
    sample = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
    if sample is None:
        raise RuntimeError(f"無法讀取影像：{files[0]}")
    h, w = sample.shape[:2]
    return files, (w, h)


def print_seq_info(tag: str, files, wh):
    w, h = wh
    print(f"[{tag}] 數量={len(files)}, 解析度={w}x{h}")
    print(f"[{tag}] 首幀={files[0].name}, 末幀={files[-1].name}")


def read_mat(mat_path: Path):
    if not mat_path.exists():
        print(f"[WARN] 找不到 MAT：{mat_path}")
        return None
    m = loadmat(mat_path)
    # 可能的鍵名（你前面提供過）
    rec_key_candidates  = ["pulseOxRecord", "ppg", "signal", "pulseox", "PulseOx", "Pulse"]
    time_key_candidates = ["pulseOxTime", "time", "t"]

    def pick_key(dd, cands):
        for k in cands:
            if k in dd:
                return k
        return None

    rec_key  = pick_key(m, rec_key_candidates)
    time_key = pick_key(m, time_key_candidates)

    if rec_key is None:
        print(f"[MAT] 變數鍵名：{list(m.keys())}")
        print("[WARN] 找不到 PPG 訊號欄位（例如 pulseOxRecord），僅回傳原始 dict。")
        return {"raw": m}

    sig = np.squeeze(m[rec_key]).astype(float)
    t = None
    fs = None
    if time_key and time_key in m:
        t = np.squeeze(m[time_key]).astype(float)
        # 估取樣率
        t = t - t[0]
        dt = np.diff(t)
        dt = dt[(dt > 0) & np.isfinite(dt)]
        if dt.size:
            fs = 1.0 / np.median(dt)

    print(f"[MAT] 檔案：{mat_path.name}")
    print(f"[MAT] 訊號長度={sig.size}, 值域=({sig.min():.1f},{sig.max():.1f})")
    if fs is not None:
        print(f"[MAT] 估計取樣率 fs≈{fs:.3f} Hz（依 {time_key}）")
    else:
        print("[MAT] 未找到時間軸（或無法估 fs）")

    return {"signal": sig, "time": t, "fs": fs, "keys": list(m.keys())}


def fetch_frame(files, idx):
    if idx < 0 or idx >= len(files):
        return None, None
    img = cv2.imread(str(files[idx]), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    # 轉成 BGR 方便併排畫面（灰階也轉 3 通道）
    if img.ndim == 2:
        img_viz = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_viz = img.copy()
    return img, img_viz


def make_merged(nir_viz, rgb_viz):
    if nir_viz is None and rgb_viz is None:
        return None
    if nir_viz is None:
        return rgb_viz
    if rgb_viz is None:
        return nir_viz
    # 高度對齊後水平拼接
    h = min(nir_viz.shape[0], rgb_viz.shape[0])
    def resize_h(im):
        return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
    nir_r = resize_h(nir_viz)
    rgb_r = resize_h(rgb_viz)
    merged = np.hstack([nir_r, rgb_r])
    # 標示來源
    cv2.putText(merged, "NIR", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(merged, "RGB", (nir_r.shape[1]+10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    return merged


def parse_frame_index(name: str):
    """從 Frame00012.pgm 取整數索引，取不到回傳 None。"""
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nir_dir", type=str, default=str(DEFAULT_NIR_DIR), help="NIR 影格資料夾（相對路徑）")
    ap.add_argument("--rgb_dir", type=str, default=str(DEFAULT_RGB_DIR), help="RGB 影格資料夾（相對路徑）")
    ap.add_argument("--mat",     type=str, default=str(DEFAULT_MAT),     help="PulseOx MAT 檔（相對路徑）")
    ap.add_argument("--nir_offset", type=int, default=DEFAULT_NIR_OFFSET, help="NIR 幀索引偏移（用於對齊）")
    ap.add_argument("--rgb_offset", type=int, default=DEFAULT_RGB_OFFSET, help="RGB 幀索引偏移（用於對齊）")
    ap.add_argument("--delay_ms", type=int, default=DEFAULT_DELAY_MS, help="播放延遲（ms/幀）")
    args = ap.parse_args()

    nir_dir = Path(args.nir_dir)
    rgb_dir = Path(args.rgb_dir)
    mat_path = Path(args.mat)

    # 掃描 NIR / RGB
    nir_files, nir_wh = scan_frames(nir_dir)
    rgb_files, rgb_wh = scan_frames(rgb_dir)

    print_seq_info("NIR", nir_files, nir_wh)
    print_seq_info("RGB", rgb_files, rgb_wh)

    # 顯示起始索引（從檔名解析）
    nir_first_idx = parse_frame_index(nir_files[0].name)
    rgb_first_idx = parse_frame_index(rgb_files[0].name)
    print(f"[NIR] 起始檔名索引 ≈ {nir_first_idx}, [RGB] 起始檔名索引 ≈ {rgb_first_idx}")

    # 讀 MAT
    mat_info = read_mat(mat_path)

    # 播放器
    win = "NIR | RGB preview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # 以「檔名數字索引」為基準做粗略對齊：
    # 給一個統一的 frame_id，分別加上 offset 去抓兩路幀
    # 先估可播放的 frame_id 範圍
    nir_indices = [parse_frame_index(p.name) for p in nir_files]
    rgb_indices = [parse_frame_index(p.name) for p in rgb_files]

    if None in (nir_indices[0], rgb_indices[0]):
        # 若檔名無數字，退回以陣列索引 0..n-1
        print("[WARN] 檔名無數字索引，改用陣列索引對齊。")
        frame_id_min = 0
        frame_id_max = min(len(nir_files) + args.nir_offset, len(rgb_files) + args.rgb_offset) - 1
        use_array_index = True
    else:
        # 使用數字索引
        frame_id_min = max(nir_indices[0] - args.nir_offset, rgb_indices[0] - args.rgb_offset)
        frame_id_max = min(nir_indices[-1] - args.nir_offset, rgb_indices[-1] - args.rgb_offset)
        use_array_index = False

    cur_id = frame_id_min
    playing = True

    def get_by_frame_id(files, indices, fid, offset):
        if use_array_index:
            # 以 0..n-1 的索引（fid 當作陣列索引）
            arr_idx = fid + offset
            if arr_idx < 0 or arr_idx >= len(files):
                return None, None
            return fetch_frame(files, arr_idx)
        else:
            # 用數字索引，把 fid+offset 轉回檔案位置
            target = fid + offset
            # 二分搜尋最近檔案
            lo, hi = 0, len(indices) - 1
            pos = None
            while lo <= hi:
                mid = (lo + hi) // 2
                if indices[mid] == target:
                    pos = mid; break
                elif indices[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            if pos is None:
                # 沒有 exact match，就不要回傳，避免錯頻
                return None, None
            return fetch_frame(files, pos)

    print(f"[INFO] 播放索引範圍：{frame_id_min} ~ {frame_id_max}（offset: NIR={args.nir_offset}, RGB={args.rgb_offset}）")
    print("[INFO] Space=播放/暫停，→/←=逐幀，q=離開")

    while True:
        if playing:
            nir_img, nir_viz = get_by_frame_id(nir_files, nir_indices, cur_id, args.nir_offset)
            rgb_img, rgb_viz = get_by_frame_id(rgb_files, rgb_indices, cur_id, args.rgb_offset)

            if nir_viz is None and rgb_viz is None:
                # 跳到下一幀
                cur_id += 1
                if cur_id > frame_id_max:
                    cur_id = frame_id_min
                continue

            merged = make_merged(nir_viz, rgb_viz)
            # 在畫面上顯示目前 frame_id
            if merged is not None:
                cv2.putText(merged, f"frame_id={cur_id}", (10, merged.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2, cv2.LINE_AA)
                cv2.imshow(win, merged)

            cur_id += 1
            if cur_id > frame_id_max:
                cur_id = frame_id_min

            key = cv2.waitKey(args.delay_ms) & 0xFF
        else:
            key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == 32:   # Space
            playing = not playing
        elif key == 83:   # Right arrow
            playing = False
            cur_id = min(cur_id + 1, frame_id_max)
        elif key == 81:   # Left arrow
            playing = False
            cur_id = max(cur_id - 1, frame_id_min)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
