# -*- coding: utf-8 -*-
"""
face_extractor.py
基礎臉部偵測與裁切（OpenCV Haar），提供可重用 API：
- FaceExtractor 類別：load_haar()、detect()、crop()、process_directory()

依賴：opencv-python, numpy
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import csv
import cv2
import numpy as np


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _to_bgr(img: np.ndarray) -> np.ndarray:
    """PGM/灰階轉 3 通道 BGR，其他情況原樣回傳。"""
    if img is None:
        return img
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def _expand_with_padding(x0, y0, x1, y1, w, h, padding: float) -> Tuple[int, int, int, int]:
    """依 padding 擴張 bbox，並限制於影像範圍。"""
    bw = x1 - x0
    bh = y1 - y0
    pad_w = int(round(bw * padding))
    pad_h = int(round(bh * padding))
    nx0 = max(0, x0 - pad_w)
    ny0 = max(0, y0 - pad_h)
    nx1 = min(w, x1 + pad_w)
    ny1 = min(h, y1 + pad_h)
    return nx0, ny0, nx1, ny1


class FaceExtractor:
    """可重用的臉部偵測/裁切器（Haar 基礎法）。"""

    def __init__(
        self,
        cascade_path: Optional[str] = None,
        min_size: Tuple[int, int] = (64, 64),
        scaleFactor: float = 1.1,
        minNeighbors: int = 5,
        padding: float = 0.15,
        target_size: Optional[int] = 256
    ):
        """
        Args:
            cascade_path: 自訂 Haar XML 路徑；None 則用內建 `haarcascade_frontalface_default.xml`
            min_size: 最小臉尺寸
            scaleFactor: 影像金字塔縮放係數（小一點更準但慢）
            minNeighbors: 偵測器鄰近門檻（大一點更穩定）
            padding: 在偵測框外擴比例（0~0.5 常見）
            target_size: 裁切輸出成正方形邊長；None/0 表示不縮放
        """
        self.cascade = self.load_haar(cascade_path)
        self.min_size = min_size
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.padding = padding
        self.target_size = target_size

    @staticmethod
    def load_haar(cascade_path: Optional[str] = None) -> cv2.CascadeClassifier:
        if cascade_path:
            cascade = cv2.CascadeClassifier(cascade_path)
        else:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if cascade.empty():
            raise RuntimeError("無法載入 Haar 分類器，請確認 OpenCV 安裝與路徑。")
        return cascade

    def detect(self, img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        偵測最大臉，回傳 (x, y, w, h)；找不到回傳 None。
        """
        if img is None:
            return None
        bgr = _to_bgr(img)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=self.min_size
        )
        if len(faces) == 0:
            return None
        # 選最大臉
        x, y, w, h = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
        return int(x), int(y), int(w), int(h)

    def crop(self, img: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        偵測並裁切人臉。
        Return:
            {
              "crop": 裁切影像 (BGR),
              "bbox_expanded": (x0,y0,x1,y1),  # 含 padding
              "bbox_raw": (x,y,w,h)            # Haar 原框
            } or None
        """
        if img is None:
            return None
        bgr = _to_bgr(img)
        h, w = bgr.shape[:2]

        det = self.detect(bgr)
        if det is None:
            return None
        x, y, fw, fh = det
        x0, y0, x1, y1 = _expand_with_padding(x, y, x + fw, y + fh, w, h, self.padding)
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None

        if self.target_size and self.target_size > 0:
            crop = cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

        return {
            "crop": crop,
            "bbox_expanded": (x0, y0, x1, y1),
            "bbox_raw": (x, y, fw, fh)
        }

    def process_directory(
        self,
        src_dir: Path,
        dst_dir: Path,
        pattern: str = "*.pgm",
        overwrite: bool = False,
        limit: Optional[int] = None,
        csv_name: str = "crops.csv"
    ) -> Dict[str, Any]:
        """
        批次處理資料夾，輸出 PNG 臉部影像與 CSV。
        """
        _ensure_dir(dst_dir)
        files = sorted(Path(src_dir).glob(pattern))
        if limit is not None:
            files = files[:limit]
        if not files:
            raise FileNotFoundError(f"在 {src_dir} 找不到 {pattern}")

        csv_path = dst_dir / csv_name
        with open(csv_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(["out_file", "src_file", "x0", "y0", "x1", "y1", "raw_x", "raw_y", "raw_w", "raw_h"])

            success, fail = 0, 0
            for i, p in enumerate(files, 1):
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                res = self.crop(img)
                if res is None:
                    fail += 1
                    continue

                out_name = f"{p.stem}_face.png"
                out_path = dst_dir / out_name
                if out_path.exists() and not overwrite:
                    success += 1  # 視為成功
                else:
                    if cv2.imwrite(str(out_path), res["crop"]):
                        success += 1
                    else:
                        fail += 1
                        continue

                x0, y0, x1, y1 = res["bbox_expanded"]
                rx, ry, rw, rh = res["bbox_raw"]
                writer.writerow([out_name, str(p), x0, y0, x1, y1, rx, ry, rw, rh])

                if i % 500 == 0:
                    print(f"[INFO] processed {i}/{len(files)}")

        summary = {"total": len(files), "success": success, "fail": fail, "csv": str(csv_path)}
        print(f"[DONE] {summary}")
        return summary
