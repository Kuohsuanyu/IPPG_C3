import metric

# 模擬的真實與預測心率
y_true = [70, 75, 80] #真實
y_pred = [68, 78, 79] #預測

print("MAE:", metric.mean_absolute_error(y_true, y_pred))
print("RMSE:", metric.root_mean_squared_error(y_true, y_pred))
print("SUC6:", metric.success_rate(y_true, y_pred, threshold=6), "%")
print("MER:", metric.mean_error_rate(y_true, y_pred), "%")
