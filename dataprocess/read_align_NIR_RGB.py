import cv2
from pathlib import Path
import numpy as np

# 固定路徑定義，這裏修改成你想要的資料夾路徑即可
nir_dir_path = r"F:\subject1\subject1_garage_still_975\NIR"
rgb_dir_path = r"F:\subject1\subject1_garage_still_975\RGB"
mat_file_path = r"F:\subject1\subject1_garage_still_975\PulseOX\pulseOx.mat"

def load_and_align_images(nir_dir, rgb_dir):
    # 這裏取偶數幀示範，依需求調整
    nir_files = sorted(Path(nir_dir).glob("Frame*.pgm"))[::2]
    rgb_files = sorted(Path(rgb_dir).glob("Frame*.pgm"))
    n_frames = min(len(nir_files), len(rgb_files))

    for i in range(n_frames):
        nir_img = cv2.imread(str(nir_files[i]), cv2.IMREAD_GRAYSCALE)
        rgb_img = cv2.imread(str(rgb_files[i]))

        if nir_img is None or rgb_img is None:
            print(f"Skipped frame {i}")
            continue

        nir_color = cv2.cvtColor(nir_img, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((rgb_img, nir_color))

        cv2.imshow("RGB | NIR Side-by-Side", combined)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    load_and_align_images(nir_dir_path, rgb_dir_path)

    # 後續可自行讀取 mat_file_path 做同步處理或其他分析
