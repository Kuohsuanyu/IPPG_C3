import cv2
from pathlib import Path
import numpy as np

def load_and_align_images(nir_dir, rgb_dir):
    # 只取偶數幀（Index從0算起，即取0, 2, 4,...）
    nir_files = sorted(nir_dir.glob("Frame*.pgm"))[::2]
    rgb_files = sorted(rgb_dir.glob("Frame*.pgm"))
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
    base_path = Path(r"F:\subject1\subject1_driving_large_motion_975")
    nir_path = base_path / "NIR"
    rgb_path = base_path / "RGB"

    load_and_align_images(nir_path, rgb_path)
