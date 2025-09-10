"""
GREEN 方法參考：

Verkruysse, W. 等人提出的利用環境光遠端容積脈搏影像法。

主要思想：利用綠光對血液的吸收較強，通過提取綠色頻道的光度變化捕捉脈搏信號。

"""

import numpy as np
import math
from scipy import signal
from scipy import linalg
from unsupervised_methods import utils

def GREEN(frames):
    """
    GREEN 方法實作函式：輸入影像序列，產出脈搏信號（BVP）

    參數：
    - frames：多幀影像的 numpy 陣列，輸入視頻或影像序列。

    流程：
    1. 利用 utils.process_video() 函式，
       將多張影像轉成每幀 RGB 三色通道的時間序列數據。
    2. 從處理結果中取出綠色頻道資料 (索引1) ，
       因此 BVP 信號被定義為綠光通道感測到的血液容積變化。
    3. 將該頻道序列展平成一維波形數據作為輸出。

    返回：
    - BVP：一維 numpy array 脈搏血容量變化信號。

    備註：
    此方法為經典基準方案，簡潔且計算效率高。
    """
    precessed_data = utils.process_video(frames)  # 轉成 (幀數, 3, 時間等) 的時間序列
    BVP = precessed_data[:, 1, :]                  # 取出綠色頻道數據
    BVP = BVP.reshape(-1)                           # 展平為一維信號
    return BVP
