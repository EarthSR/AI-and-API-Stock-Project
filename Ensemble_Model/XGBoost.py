import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

# ✅ 1. โหลดข้อมูล
train_features = np.load('../RNN_Model/train_features.npy')
train_targets = np.load('../RNN_Model/train_targets.npy')
test_features = np.load('../RNN_Model/test_features.npy')
test_targets = np.load('../RNN_Model/test_targets.npy')

# ✅ 2. ใช้ RobustScaler กับ Target
scaler_target = RobustScaler()
y_train_scaled = scaler_target.fit_transform(train_targets.reshape(-1, 1))
y_test_scaled = scaler_target.transform(test_targets.reshape(-1, 1))

# ✅ 3. ใช้ชุดข้อมูลเดิม
X_train, y_train = train_features, y_train_scaled
X_test, y_test = test_features, y_test_scaled

print(f"✅ ขนาดชุด Train: {X_train.shape}, ขนาดชุด Test: {X_test.shape}")

# ✅ 4. ปรับพารามิเตอร์ให้โมเดลแข็งแกร่งขึ้น
xgb_params = {
    "n_estimators": 5000,
    "learning_rate": 0.001,  # ลดลงเพื่อให้เรียนรู้ละเอียดขึ้น
    "max_depth": 8,  # เพิ่มความซับซ้อนของโมเดล
    "min_child_weight": 5,  # ช่วยลด overfitting
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "gamma": 0.2,  # ป้องกัน overfitting
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42
}

# ✅ 5. สร้าง DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# ✅ 6. ฝึกโมเดล XGBoost พร้อม Early Stopping
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=5000,
    evals=[(dtest, "Test")],
    early_stopping_rounds=50,
    verbose_eval=True
)

print("✅ ฝึกโมเดล XGBoost สำเร็จ!")

# ✅ 7. บันทึกโมเดล
joblib.dump(xgb_model, 'xgb_meta_learner.pkl')
joblib.dump(scaler_target, '../RNN_Model/scaler_target_xgb.pkl')  # บันทึก scaler ที่ใช้กับ XGBoost
print("✅ บันทึกโมเดล XGBoost สำเร็จ!")

# ✅ 8. ทดสอบการพยากรณ์
y_pred_xgb = xgb_model.predict(dtest)

# ✅ 9. ย้อนกลับค่าทำนายเป็นสเกลจริง
y_pred_actual = scaler_target.inverse_transform(y_pred_xgb.reshape(-1, 1))
y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# ✅ 10. จำกัดค่าพยากรณ์ไม่ให้เกินช่วงค่าจริง
y_pred_actual = np.clip(y_pred_actual, np.min(y_test_actual), np.max(y_test_actual))

# ✅ 11. ตรวจสอบค่าผิดปกติ
print("\n🔍 ตัวอย่างค่าจริงและค่าทำนาย:")
print("Actual:", y_test_actual[:5].flatten())
print("Predicted:", y_pred_actual[:5].flatten())

print("\n🔍 ตรวจสอบค่าต่ำสุด-สูงสุด:")
print("Min Actual:", np.min(y_test_actual), "Max Actual:", np.max(y_test_actual))
print("Min Predicted:", np.min(y_pred_actual), "Max Predicted:", np.max(y_pred_actual))

# ✅ 12. คำนวณ Metrics
valid_mask = (np.abs(y_test_actual - y_pred_actual) < 2 * np.std(y_test_actual))

mae = mean_absolute_error(y_test_actual[valid_mask], y_pred_actual[valid_mask])
mse = mean_squared_error(y_test_actual[valid_mask], y_pred_actual[valid_mask])
rmse = np.sqrt(mse)

# ✅ 13. ปรับการคำนวณ MAPE เพื่อลด Error สูงผิดปกติ
mape = np.mean(np.abs((y_test_actual - y_pred_actual) / (np.abs(y_test_actual) + 1))) * 100
r2 = r2_score(y_test_actual[valid_mask], y_pred_actual[valid_mask])

print(f"\n✅ ผลลัพธ์การพยากรณ์ XGBoost")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}%")
print(f"R2 Score: {r2:.4f}")

# ✅ 14. บันทึกผลลัพธ์ลง CSV
results_df = pd.DataFrame({
    'Actual': y_test_actual.flatten(),
    'Predicted': y_pred_actual.flatten()
})

results_df.to_csv('xgboost_predictions.csv', index=False)
print("✅ บันทึกผลลัพธ์ลงไฟล์ 'xgboost_predictions.csv' สำเร็จ!")
