import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import RobustScaler

# ✅ 1. โหลดผลลัพธ์จาก 4 โมเดล (LSTM, GRU, XGBoost, RandomForest)
predictions_lstm = pd.read_csv("../LSTM_model/all_predictions_per_day_multi_task.csv")
predictions_gru = pd.read_csv("../GRU_Model/all_predictions_per_day_multi_task.csv")

# ✅ 2. รวมผลลัพธ์จากทุกโมเดลเป็น Feature Set ใหม่
ensemble_features = pd.DataFrame({
    "Ticker": predictions_lstm["Ticker"],
    "Date": pd.to_datetime(predictions_lstm["Date"]),
    "Actual_Price": predictions_lstm["Actual_Price"],
    "Predicted_Price_LSTM": predictions_lstm["Predicted_Price"],
    "Predicted_Price_GRU": predictions_gru["Predicted_Price"],
    "Actual_Direction": predictions_lstm["Actual_Dir"],
    "Predicted_Dir_LSTM": predictions_lstm["Predicted_Dir"],
    "Predicted_Dir_GRU": predictions_gru["Predicted_Dir"]
})

# ✅ 3. แบ่งข้อมูล Train/Test
train_cutoff = pd.Timestamp("2024-12-01")
train_mask = ensemble_features["Date"] < train_cutoff

X_train = ensemble_features.loc[train_mask, ["Predicted_Price_LSTM", "Predicted_Price_GRU"]].values
y_train_price = ensemble_features.loc[train_mask, "Actual_Price"].values
y_train_dir = ensemble_features.loc[train_mask, "Actual_Direction"].values

X_test = ensemble_features.loc[~train_mask, ["Predicted_Price_LSTM", "Predicted_Price_GRU"]].values
y_test_price = ensemble_features.loc[~train_mask, "Actual_Price"].values
y_test_dir = ensemble_features.loc[~train_mask, "Actual_Direction"].values

# ✅ 4. โหลด Scaler เดิมที่เคยใช้
scaler_target = joblib.load("../LSTM_model/scaler_target.pkl")

# ✅ 5. ใช้ Scaler กับราคาปิด
y_train_price_scaled = scaler_target.transform(y_train_price.reshape(-1, 1))
y_test_price_scaled = scaler_target.transform(y_test_price.reshape(-1, 1))

# ✅ 6. Train Base Models (ราคา & ทิศทาง)
xgb_model = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.002, max_depth=8, subsample=0.85, colsample_bytree=0.85, gamma=0.2, objective="reg:squarederror", random_state=42)
rf_model = RandomForestRegressor(n_estimators=2000, max_depth=8, min_samples_split=4, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=42)
xgb_dir_model = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.002, max_depth=8, subsample=0.85, colsample_bytree=0.85, gamma=0.2, objective="binary:logistic", random_state=42)
rf_dir_model = RandomForestClassifier(n_estimators=2000, max_depth=8, min_samples_split=4, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=42)

print("\n🔍 กำลังฝึก Base Models (ราคาปิด & ทิศทาง)...")
xgb_model.fit(X_train, y_train_price_scaled.ravel())
rf_model.fit(X_train, y_train_price_scaled.ravel())
xgb_dir_model.fit(X_train, y_train_dir.ravel())
rf_dir_model.fit(X_train, y_train_dir.ravel())

# ✅ 7. ทำนายค่าจาก Base Models
y_pred_xgb = xgb_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
dir_pred_xgb = xgb_dir_model.predict(X_test)
dir_pred_rf = rf_dir_model.predict(X_test)

# ✅ 8. ย้อนกลับค่าทำนายราคาปิดเป็นสเกลจริง
y_pred_xgb_actual = scaler_target.inverse_transform(y_pred_xgb.reshape(-1, 1))
y_pred_rf_actual = scaler_target.inverse_transform(y_pred_rf.reshape(-1, 1))
y_test_actual = scaler_target.inverse_transform(y_test_price_scaled.reshape(-1, 1))

# ✅ 9. ใช้ Weighted Stacking (รวม 4 โมเดล)
w_lstm, w_gru, w_xgb, w_rf = 0.3, 0.3, 0.2, 0.2
ensemble_weighted_price = (w_lstm * ensemble_features.loc[~train_mask, "Predicted_Price_LSTM"].values.reshape(-1, 1)) + \
                          (w_gru * ensemble_features.loc[~train_mask, "Predicted_Price_GRU"].values.reshape(-1, 1)) + \
                          (w_xgb * y_pred_xgb_actual) + (w_rf * y_pred_rf_actual)

# ✅ 10. ใช้ Weighted Stacking สำหรับทิศทาง
ensemble_weighted_dir = (0.3 * ensemble_features.loc[~train_mask, "Predicted_Dir_LSTM"].values) + \
                        (0.3 * ensemble_features.loc[~train_mask, "Predicted_Dir_GRU"].values) + \
                        (0.2 * dir_pred_xgb) + (0.2 * dir_pred_rf)

ensemble_weighted_dir = (ensemble_weighted_dir > 0.5).astype(int)  # แปลงเป็น 0 หรือ 1

# ✅ 11. ฟังก์ชันประเมินผลลัพธ์
def evaluate_price(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n✅ **ผลลัพธ์การพยากรณ์ราคาปิด {model_name}**")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

def evaluate_direction(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ **ผลลัพธ์การพยากรณ์ทิศทาง {model_name}**")
    print(f"Accuracy: {acc:.4f}")

# ✅ 12. แสดงผลลัพธ์แยกแต่ละโมเดล
evaluate_price(y_test_actual, ensemble_features.loc[~train_mask, "Predicted_Price_LSTM"].values, "LSTM Model")
evaluate_direction(y_test_dir, ensemble_features.loc[~train_mask, "Predicted_Dir_LSTM"].values, "LSTM Model")

evaluate_price(y_test_actual, ensemble_features.loc[~train_mask, "Predicted_Price_GRU"].values, "GRU Model")
evaluate_direction(y_test_dir, ensemble_features.loc[~train_mask, "Predicted_Dir_GRU"].values, "GRU Model")

evaluate_price(y_test_actual, y_pred_xgb_actual, "XGBoost Model")
evaluate_direction(y_test_dir, dir_pred_xgb, "XGBoost Model")

evaluate_price(y_test_actual, y_pred_rf_actual, "RandomForest Model")
evaluate_direction(y_test_dir, dir_pred_rf, "RandomForest Model")

evaluate_price(y_test_actual, ensemble_weighted_price, "Weighted Stacking")
evaluate_direction(y_test_dir, ensemble_weighted_dir, "Weighted Stacking")

# ✅ 13. บันทึกโมเดล Weighted Stacking
joblib.dump(ensemble_weighted_price, "weighted_stacking_price.pkl")
joblib.dump(ensemble_weighted_dir, "weighted_stacking_direction.pkl")

# ✅ บันทึกโมเดล Base Models
joblib.dump(xgb_model, "xgb_model_price.pkl")
joblib.dump(rf_model, "rf_model_price.pkl")
joblib.dump(xgb_dir_model, "xgb_model_direction.pkl")
joblib.dump(rf_dir_model, "rf_model_direction.pkl")

# ✅ บันทึก Scaler เพื่อใช้ในอนาคต
joblib.dump(scaler_target, "scaler_target.pkl")

print("✅ บันทึกโมเดล Weighted Stacking และ Base Models สำเร็จ!")


print("✅ เสร็จสิ้น!")
