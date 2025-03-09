import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import RobustScaler

# ✅ 1. โหลดข้อมูล
train_features = np.load('../GRU_Model/train_features.npy')
train_targets = np.load('../GRU_Model/train_price.npy')
test_features = np.load('../GRU_Model/test_features.npy')
test_targets = np.load('../GRU_Model/test_price.npy')

# ✅ 2. ใช้ RobustScaler กับราคาปิด
scaler_price = RobustScaler()
y_train_scaled = scaler_price.fit_transform(train_targets.reshape(-1, 1))
y_test_scaled = scaler_price.transform(test_targets.reshape(-1, 1))

# ✅ 3. แปลงทิศทางเป็น 1 (ขึ้น) / 0 (ลง)
direction_train = (train_targets > np.median(train_targets)).astype(int)
direction_test = (test_targets > np.median(test_targets)).astype(int)

# ✅ 4. ใช้ชุดข้อมูลเดิม
X_train, y_train_price, y_train_dir = train_features, y_train_scaled, direction_train
X_test, y_test_price, y_test_dir = test_features, y_test_scaled, direction_test

print(f"✅ ขนาดชุด Train: {X_train.shape}, ขนาดชุด Test: {X_test.shape}")

# ✅ 5. ตั้งค่า Base Models (โมเดลเดียวทำนาย 2 ค่า)
xgb_model = xgb.XGBRegressor(
    n_estimators=2000, learning_rate=0.002, max_depth=8, subsample=0.85,
    colsample_bytree=0.85, gamma=0.2, objective="reg:squarederror", random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=2000, max_depth=8, min_samples_split=4, min_samples_leaf=2,
    max_features="sqrt", n_jobs=-1, random_state=42
)

xgb_dir_model = xgb.XGBClassifier(
    n_estimators=2000, learning_rate=0.002, max_depth=8, subsample=0.85,
    colsample_bytree=0.85, gamma=0.2, objective="binary:logistic", random_state=42
)

rf_dir_model = RandomForestClassifier(
    n_estimators=2000, max_depth=8, min_samples_split=4, min_samples_leaf=2,
    max_features="sqrt", n_jobs=-1, random_state=42
)

# ✅ 6. ฝึก Base Models
print("\n🔍 กำลังฝึก Base Models (ราคาปิด & ทิศทาง)...")
xgb_model.fit(X_train, y_train_price.ravel())
rf_model.fit(X_train, y_train_price.ravel())
xgb_dir_model.fit(X_train, y_train_dir.ravel())
rf_dir_model.fit(X_train, y_train_dir.ravel())

print("✅ ฝึก Base Models สำเร็จ!")

# ✅ 7. ทำนายค่าจาก Base Models
y_pred_xgb = xgb_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
dir_pred_xgb = xgb_dir_model.predict(X_test)
dir_pred_rf = rf_dir_model.predict(X_test)

# ✅ 8. ย้อนกลับค่าทำนายราคาปิดเป็นสเกลจริง
y_pred_xgb_actual = scaler_price.inverse_transform(y_pred_xgb.reshape(-1, 1))
y_pred_rf_actual = scaler_price.inverse_transform(y_pred_rf.reshape(-1, 1))
y_test_actual = scaler_price.inverse_transform(y_test_price.reshape(-1, 1))

# ✅ 9. ใช้ Weighted Stacking
ensemble_weighted_price = (0.6 * y_pred_xgb_actual) + (0.4 * y_pred_rf_actual)
ensemble_weighted_dir = (dir_pred_xgb + dir_pred_rf) / 2
ensemble_weighted_dir = (ensemble_weighted_dir > 0.5).astype(int)  # Convert to binary

# ✅ 10. สร้าง Meta Learner (ใช้ LightGBM)
meta_features_train = np.column_stack((y_pred_xgb, y_pred_rf))  
meta_dir_features_train = np.column_stack((dir_pred_xgb, dir_pred_rf))

meta_learner = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.005, max_depth=5, random_state=42
)

meta_dir_learner = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.005, max_depth=5, random_state=42
)

# ✅ 11. ฝึก Meta Learner
meta_learner.fit(meta_features_train, y_test_price.ravel())
meta_dir_learner.fit(meta_dir_features_train, y_test_dir.ravel())

# ✅ 12. ทำนายด้วย Meta Learner
y_pred_meta = meta_learner.predict(meta_features_train)
y_pred_meta_actual = scaler_price.inverse_transform(y_pred_meta.reshape(-1, 1))

dir_pred_meta = meta_dir_learner.predict(meta_dir_features_train)

# ✅ 13. คำนวณ Metrics
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
    return mae, mse, rmse, r2

def evaluate_direction(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ **ผลลัพธ์การพยากรณ์ทิศทาง {model_name}**")
    print(f"Accuracy: {acc:.4f}")
    return acc

# ✅ 14. ประเมินผลลัพธ์
evaluate_price(y_test_actual, y_pred_meta_actual, "Meta Learner Stacking")
evaluate_direction(y_test_dir, dir_pred_meta, "Meta Learner Stacking")

# ✅ 15. บันทึกผลลัพธ์ลง CSV
results_df = pd.DataFrame({
    'Actual_Price': y_test_actual.flatten(),
    'Predicted_XGB_Price': y_pred_xgb_actual.flatten(),
    'Predicted_RF_Price': y_pred_rf_actual.flatten(),
    'Weighted_Stacking_Price': ensemble_weighted_price.flatten(),
    'Meta_Learner_Price': y_pred_meta_actual.flatten(),
    'Actual_Direction': y_test_dir.flatten(),
    'Predicted_Direction_Meta': dir_pred_meta.flatten()
})

results_df.to_csv('stacking_ensemble_predictions.csv', index=False)

# ✅ 16. บันทึกโมเดล
joblib.dump(meta_learner, 'meta_learner_price.pkl')
joblib.dump(meta_dir_learner, 'meta_learner_direction.pkl')
joblib.dump(scaler_price, 'scaler_price.pkl')

print("✅ บันทึกโมเดลสำเร็จ!")
print("✅ บันทึกผลลัพธ์ลงไฟล์ 'stacking_ensemble_predictions.csv' สำเร็จ!")
