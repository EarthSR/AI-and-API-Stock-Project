import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
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

# ✅ 4. ตั้งค่า Base Models
xgb_model = xgb.XGBRegressor(
    n_estimators=2000, learning_rate=0.002, max_depth=8, subsample=0.85,
    colsample_bytree=0.85, gamma=0.2, objective="reg:squarederror", random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=2000, max_depth=8, min_samples_split=4, min_samples_leaf=2,
    max_features="sqrt", n_jobs=-1, random_state=42
)

# ✅ 5. ฝึก Base Models
print("\n🔍 กำลังฝึก Base Models...")
xgb_model.fit(X_train, y_train.ravel())
rf_model.fit(X_train, y_train.ravel())
print("✅ ฝึก Base Models สำเร็จ!")

# ✅ 6. ทำนายค่าจาก Base Models
y_pred_xgb = xgb_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# ✅ 7. ย้อนกลับค่าทำนายเป็นสเกลจริง
y_pred_xgb_actual = scaler_target.inverse_transform(y_pred_xgb.reshape(-1, 1))
y_pred_rf_actual = scaler_target.inverse_transform(y_pred_rf.reshape(-1, 1))
y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# ✅ 8. สร้าง Weighted Stacking (ใช้สัดส่วน 0.6 : 0.4)
ensemble_weighted = (0.6 * y_pred_xgb_actual) + (0.4 * y_pred_rf_actual)

# ✅ 9. สร้าง Meta Learner (ใช้ LightGBM)
meta_features_train = np.column_stack((y_pred_xgb, y_pred_rf))  # ใช้ค่าทำนายจาก Base Models
meta_learner = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.005, max_depth=5, random_state=42
)

# ✅ **ใช้ `y_test` เป็น Target ของ Meta Learner**
meta_learner.fit(meta_features_train, y_test.ravel())

y_pred_meta = meta_learner.predict(meta_features_train)
y_pred_meta_actual = scaler_target.inverse_transform(y_pred_meta.reshape(-1, 1))

# ✅ 10. คำนวณ Metrics
def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n✅ **ผลลัพธ์การพยากรณ์ {model_name}**")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    return mae, mse, rmse, r2

# ✅ ประเมินผลลัพธ์
results = {
    "XGBoost": evaluate(y_test_actual, y_pred_xgb_actual, "XGBoost"),
    "Random Forest": evaluate(y_test_actual, y_pred_rf_actual, "Random Forest"),
    "Weighted Stacking": evaluate(y_test_actual, ensemble_weighted, "Weighted Stacking"),
    "Meta Learner Stacking (LightGBM)": evaluate(y_test_actual, y_pred_meta_actual, "Meta Learner Stacking")
}

# ✅ 11. บันทึกผลลัพธ์ลง CSV
results_df = pd.DataFrame({
    'Actual': y_test_actual.flatten(),
    'Predicted_XGB': y_pred_xgb_actual.flatten(),
    'Predicted_RF': y_pred_rf_actual.flatten(),
    'Weighted_Stacking': ensemble_weighted.flatten(),
    'Meta_Learner': y_pred_meta_actual.flatten()
})

results_df.to_csv('stacking_ensemble_predictions.csv', index=False)

# ✅ บันทึกโมเดล
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(meta_learner, 'meta_learner.pkl')
joblib.dump(scaler_target, 'scaler_target.pkl')

print("✅ บันทึกโมเดลสำเร็จ!")
print("✅ บันทึกผลลัพธ์ลงไฟล์ 'stacking_ensemble_predictions.csv' สำเร็จ!")
