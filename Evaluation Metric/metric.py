import numpy as np

def mean_absolute_error(y_true, y_pred):
    """
    計算平均絕對誤差 (MAE)
    y_true: 真實心率陣列 (numpy array)
    y_pred: 預測心率陣列 (numpy array)
    回傳 MAE 數值
    """
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    """
    計算均方根誤差 (RMSE)
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def success_rate(y_true, y_pred, threshold=6):
    """
    成功率 SUC_I：預測誤差在 threshold bpm 以內的比例
    """
    success = np.abs(y_true - y_pred) <= threshold
    return np.mean(success) * 100

def mean_error_rate(y_true, y_pred):
    """
    平均誤差率 (MER)
    """
    return np.mean(np.abs(y_true - y_pred) / y_true) * 100

