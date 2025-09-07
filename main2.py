import numpy as np
from metric import mean_absolute_error, root_mean_squared_error, success_rate, mean_error_rate
from dataprocess.nir_extractor import extract_nir_channel 
from methods.nir_ec_dc_ir import ec_dc_ir_model          # 假設的NIR模型
from dataprocess.face_extractor import FaceExtractor

def estimate_hr(rppg_signal, fps=30):
    # 與之前相同的FFT估計心率方法
    rppg_signal = rppg_signal - np.mean(rppg_signal)
    freqs = np.fft.rfftfreq(len(rppg_signal), d=1/fps)
    fft_spec = np.abs(np.fft.rfft(rppg_signal))
    idx_peak = np.argmax(fft_spec)
    return freqs[idx_peak] * 60

def main():
    # 讀取 NIR 影片 & 擷取臉部ROI
    cap = cv2.VideoCapture("your_nir_video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    face_extractor = FaceExtractor()
    nir_channels = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_crop = face_extractor.crop(frame)
        if face_crop is None:
            continue
        nir_channel = extract_nir_channel(face_crop["crop"])  # 取得NIR三通道平均
        nir_channels.append(nir_channel)
    cap.release()

    nir_channels = np.array(nir_channels)
    
    # 產生rPPG訊號
    rppg_signal = ec_dc_ir_model(nir_channels)

    # 預測心率
    pred_hr = estimate_hr(rppg_signal, fps=fps)

    # 載入真實心率 ground truth，確保格式為 np.array
    ground_truth_hr = np.array([...])  # 輸入你的真實心率陣列

    # 計算評估指標
    print("MAE:", mean_absolute_error(ground_truth_hr, [pred_hr]))
    print("RMSE:", root_mean_squared_error(ground_truth_hr, [pred_hr]))
    print("SUCI:", success_rate(ground_truth_hr, [pred_hr]))
    print("MER:", mean_error_rate(ground_truth_hr, [pred_hr]))

if __name__ == "__main__":
    main()
