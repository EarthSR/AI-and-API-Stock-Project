import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

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

# ✅ 4. Feature Selection ด้วย PCA (ลดมิติข้อมูล)
pca = PCA(n_components=20)  # ลดเหลือ 20 Features
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# ✅ 5. กำหนด Base Models
xgb_params = {
    "n_estimators": 2000,
    "learning_rate": 0.002,
    "max_depth": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "gamma": 0.2,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42
}
xgb_model = xgb.XGBRegressor(**xgb_params)

rf_model = RandomForestRegressor(n_estimators=2000, max_depth=8, random_state=42, n_jobs=-1)
etr_model = ExtraTreesRegressor(n_estimators=2000, max_depth=8, random_state=42, n_jobs=-1)
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, random_state=42)

# ✅ 6. ฝึก Base Models
print("\n🔍 กำลังฝึก Base Models...")
xgb_model.fit(X_train_pca, y_train)
rf_model.fit(X_train_pca, y_train.ravel())
etr_model.fit(X_train_pca, y_train.ravel())
svr_model.fit(X_train_pca, y_train.ravel())
mlp_model.fit(X_train_pca, y_train.ravel())
print("✅ ฝึก Base Models สำเร็จ!")

# ✅ 7. สร้าง Features สำหรับ Meta Learner (Stacking)
train_meta_features = np.column_stack([
    xgb_model.predict(X_train_pca),
    rf_model.predict(X_train_pca),
    etr_model.predict(X_train_pca),
    svr_model.predict(X_train_pca),
    mlp_model.predict(X_train_pca)
])

test_meta_features = np.column_stack([
    xgb_model.predict(X_test_pca),
    rf_model.predict(X_test_pca),
    etr_model.predict(X_test_pca),
    svr_model.predict(X_test_pca),
    mlp_model.predict(X_test_pca)
])

# ✅ 8. สร้าง Weighted Stacking โดยให้ XGBoost น้ำหนักสูงสุด
ensemble_weights = [0.4, 0.2, 0.15, 0.15, 0.1]  # XGB: 40%, RF: 20%, ETR: 15%, SVR: 15%, MLP: 10%
y_pred_weighted = (
    ensemble_weights[0] * test_meta_features[:, 0] +
    ensemble_weights[1] * test_meta_features[:, 1] +
    ensemble_weights[2] * test_meta_features[:, 2] +
    ensemble_weights[3] * test_meta_features[:, 3] +
    ensemble_weights[4] * test_meta_features[:, 4]
)

# ✅ 9. ฝึก Meta Learner (ใช้ XGBoost แทน Ridge Regression)
meta_xgb = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.005, max_depth=5, random_state=42)
meta_xgb.fit(train_meta_features, y_train.ravel())
y_pred_meta = meta_xgb.predict(test_meta_features)

# ✅ 10. ย้อนกลับค่าทำนายเป็นสเกลจริง
y_pred_xgb_actual = scaler_target.inverse_transform(xgb_model.predict(X_test_pca).reshape(-1, 1))
y_pred_rf_actual = scaler_target.inverse_transform(rf_model.predict(X_test_pca).reshape(-1, 1))
y_pred_etr_actual = scaler_target.inverse_transform(etr_model.predict(X_test_pca).reshape(-1, 1))
y_pred_svr_actual = scaler_target.inverse_transform(svr_model.predict(X_test_pca).reshape(-1, 1))
y_pred_mlp_actual = scaler_target.inverse_transform(mlp_model.predict(X_test_pca).reshape(-1, 1))
y_pred_weighted_actual = scaler_target.inverse_transform(y_pred_weighted.reshape(-1, 1))
y_pred_meta_actual = scaler_target.inverse_transform(y_pred_meta.reshape(-1, 1))
y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# ✅ 11. คำนวณ Metrics
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n✅ **ผลลัพธ์การพยากรณ์ {name}**")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    return mae, mse, rmse, r2

results = {
    "XGBoost": evaluate_model("XGBoost", y_test_actual, y_pred_xgb_actual),
    "Random Forest": evaluate_model("Random Forest", y_test_actual, y_pred_rf_actual),
    "Extra Trees": evaluate_model("Extra Trees", y_test_actual, y_pred_etr_actual),
    "SVR": evaluate_model("SVR", y_test_actual, y_pred_svr_actual),
    "MLP": evaluate_model("MLP", y_test_actual, y_pred_mlp_actual),
    "Weighted Stacking": evaluate_model("Weighted Stacking", y_test_actual, y_pred_weighted_actual),
    "Meta Learner Stacking": evaluate_model("Meta Learner Stacking", y_test_actual, y_pred_meta_actual)
}

# ✅ 12. บันทึกโมเดล
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(etr_model, 'etr_model.pkl')
joblib.dump(svr_model, 'svr_model.pkl')
joblib.dump(mlp_model, 'mlp_model.pkl')
joblib.dump(meta_xgb, 'meta_xgb.pkl')
joblib.dump(scaler_target, '../RNN_Model/scaler_target_stacking.pkl')
print("✅ บันทึกโมเดลสำเร็จ!")

# ✅ 13. บันทึกผลลัพธ์ลง CSV
results_df = pd.DataFrame({
    'Actual': y_test_actual.flatten(),
    'Predicted_XGB': y_pred_xgb_actual.flatten(),
    'Predicted_RF': y_pred_rf_actual.flatten(),
    'Predicted_ETR': y_pred_etr_actual.flatten(),
    'Predicted_SVR': y_pred_svr_actual.flatten(),
    'Predicted_MLP': y_pred_mlp_actual.flatten(),
    'Weighted_Stacking': y_pred_weighted_actual.flatten(),
    'Meta_Stacking': y_pred_meta_actual.flatten()
})
results_df.to_csv('stacking_ensemble_predictions.csv', index=False)
print("✅ บันทึกผลลัพธ์ลงไฟล์ 'stacking_ensemble_predictions.csv' สำเร็จ!")
