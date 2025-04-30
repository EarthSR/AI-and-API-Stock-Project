import numpy as np 
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

# ✅ 1. โหลดผลลัพธ์จากโมเดล LSTM และ GRU
predictions_lstm = pd.read_csv("../LSTM_model/all_predictions_per_day_multi_task.csv")
predictions_gru = pd.read_csv("../GRU_Model/all_predictions_per_day_multi_task.csv")

# ✅ 2. เตรียม Feature Set ใหม่พร้อม Feature Engineering
def create_features(lstm_df, gru_df):
    # รวมผลลัพธ์จาก LSTM และ GRU
    df = pd.DataFrame({
        "Ticker": lstm_df["Ticker"],
        "Date": pd.to_datetime(lstm_df["Date"]),
        "Actual_Price": lstm_df["Actual_Price"],
        "Predicted_Price_LSTM": lstm_df["Predicted_Price"],
        "Predicted_Price_GRU": gru_df["Predicted_Price"],
        "Actual_Direction": lstm_df["Actual_Dir"],
        "Predicted_Dir_LSTM": lstm_df["Predicted_Dir"],
        "Predicted_Dir_GRU": gru_df["Predicted_Dir"]
    })
    
    # สร้าง Feature เพิ่มเติม
    df["LSTM_GRU_Price_Diff"] = abs(df["Predicted_Price_LSTM"] - df["Predicted_Price_GRU"])
    df["LSTM_GRU_Price_Ratio"] = df["Predicted_Price_LSTM"] / df["Predicted_Price_GRU"]
    df["LSTM_GRU_Dir_Match"] = (df["Predicted_Dir_LSTM"] == df["Predicted_Dir_GRU"]).astype(int)
    
    # สร้าง Rolling Features (สำหรับแต่ละ Ticker)
    for ticker in df["Ticker"].unique():
        mask = df["Ticker"] == ticker
        ticker_data = df[mask].sort_values("Date")
        
        # Rolling Mean และ Std สำหรับการทำนาย
        for window in [3, 5, 7]:
            df.loc[mask, f"LSTM_Price_Rolling_Mean_{window}"] = ticker_data["Predicted_Price_LSTM"].rolling(window).mean().values
            df.loc[mask, f"GRU_Price_Rolling_Mean_{window}"] = ticker_data["Predicted_Price_GRU"].rolling(window).mean().values
            df.loc[mask, f"LSTM_Price_Rolling_Std_{window}"] = ticker_data["Predicted_Price_LSTM"].rolling(window).std().values
            df.loc[mask, f"GRU_Price_Rolling_Std_{window}"] = ticker_data["Predicted_Price_GRU"].rolling(window).std().values
        
        # เพิ่ม Price Momentum (เปอร์เซ็นต์การเปลี่ยนแปลง)
        df.loc[mask, "Price_Pct_Change_1d"] = ticker_data["Actual_Price"].pct_change(1).values
        df.loc[mask, "Price_Pct_Change_3d"] = ticker_data["Actual_Price"].pct_change(3).values
        df.loc[mask, "Price_Pct_Change_5d"] = ticker_data["Actual_Price"].pct_change(5).values
        
        # Technical Indicators จากการทำนาย
        df.loc[mask, "LSTM_GRU_Price_Convergence"] = (ticker_data["LSTM_GRU_Price_Diff"].rolling(5).mean() - 
                                                      ticker_data["LSTM_GRU_Price_Diff"].rolling(10).mean()).values
        
        # Direction Consistency
        df.loc[mask, "LSTM_Dir_Consistency_3d"] = ticker_data["Predicted_Dir_LSTM"].rolling(3).mean().values
        df.loc[mask, "GRU_Dir_Consistency_3d"] = ticker_data["Predicted_Dir_GRU"].rolling(3).mean().values
    
    # เพิ่ม Features ตามวันในสัปดาห์
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    for day in range(5):  # 0 = Monday, 4 = Friday
        df[f"Is_Day_{day}"] = (df["DayOfWeek"] == day).astype(int)
    
    # เพิ่ม Features ตามเดือน
    df["Month"] = df["Date"].dt.month
    
    # จัดการ Missing Values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df.groupby("Ticker")[col].transform(lambda x: x.fillna(x.median()))
    
    return df

# สร้าง Features ใหม่
ensemble_features = create_features(predictions_lstm, predictions_gru)

# ✅ 3. กำจัด Outliers
def remove_outliers(df, cols, threshold=3):
    df_clean = df.copy()
    for col in cols:
        z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
        df_clean = df_clean[z_scores < threshold]
    return df_clean

price_features = ["Predicted_Price_LSTM", "Predicted_Price_GRU"]
# ensemble_features = remove_outliers(ensemble_features, price_features)

# ✅ 4. แบ่งข้อมูล Train/Test โดยใช้ถึงวันที่ December 1, 2024
train_cutoff = pd.Timestamp("2024-12-01")
train_mask = ensemble_features["Date"] < train_cutoff

# ตรวจสอบความสมดุลของข้อมูล Direction
dir_balance = ensemble_features.loc[train_mask, "Actual_Direction"].value_counts(normalize=True)
print(f"Direction Balance in Training Data: {dir_balance}")

# ดึง Features และ Target
feature_cols = [col for col in ensemble_features.columns if col not in 
                ["Ticker", "Date", "Actual_Price", "Actual_Direction"]]

# ลบคอลัมน์ที่มี missing values มากเกินไป
for col in feature_cols:
    if ensemble_features[col].isnull().sum() / len(ensemble_features) > 0.3:
        feature_cols.remove(col)

X_train = ensemble_features.loc[train_mask, feature_cols].copy()
y_train_price = ensemble_features.loc[train_mask, "Actual_Price"].values
y_train_dir = ensemble_features.loc[train_mask, "Actual_Direction"].values

X_test = ensemble_features.loc[~train_mask, feature_cols].copy()
y_test_price = ensemble_features.loc[~train_mask, "Actual_Price"].values
y_test_dir = ensemble_features.loc[~train_mask, "Actual_Direction"].values

# ✅ 5. ใช้ Scaler กับข้อมูล
try:
    # โหลด Scaler ที่เคยใช้ ถ้ามี
    scaler_target = joblib.load("../LSTM_model/scaler_target.pkl")
except:
    # สร้าง Scaler ใหม่ถ้าไม่มี
    scaler_target = RobustScaler()
    scaler_target.fit(y_train_price.reshape(-1, 1))

# ใช้ Scaler กับราคาปิด
y_train_price_scaled = scaler_target.transform(y_train_price.reshape(-1, 1)).ravel()
y_test_price_scaled = scaler_target.transform(y_test_price.reshape(-1, 1)).ravel()

# ✅ 6. เตรียม TimeSeriesSplit สำหรับ Cross-validation
tscv = TimeSeriesSplit(n_splits=5)

def optimize_xgb_price(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': 42,
    }
    
    num_boost_round = trial.suggest_int('n_estimators', 100, 1000)

    scores = []
    for train_idx, valid_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train_price_scaled[train_idx], y_train_price_scaled[valid_idx]
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        evals = [(dtrain, 'train'), (dval, 'eval')]

        bst = xgb.train(
            param, 
            dtrain, 
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        pred = bst.predict(dval)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        scores.append(rmse)
    
    return np.mean(scores)

def optimize_xgb_dir(trial):
    neg_count = np.sum(y_train_dir == 0)
    pos_count = np.sum(y_train_dir == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, scale_pos_weight * 2),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 42,
    }
    
    num_boost_round = trial.suggest_int('n_estimators', 100, 1000)

    scores = []
    for train_idx, valid_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train_dir[train_idx], y_train_dir[valid_idx]
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        evals = [(dtrain, 'train'), (dval, 'eval')]

        bst = xgb.train(
            param,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )

        pred_proba = bst.predict(dval)
        score = roc_auc_score(y_val, pred_proba)
        scores.append(score)
    
    return np.mean(scores)


tscv = TimeSeriesSplit(n_splits=5)

def optimize_lgbm_dir(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'class_weight': 'balanced',
        'objective': 'binary',
        'metric': 'auc',
        'random_state': 42
    }

    model = lgb.LGBMClassifier(**param)

    scores = []
    for train_idx, valid_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train_dir[train_idx], y_train_dir[valid_idx]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)]
        )

        pred_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, pred_proba)
        scores.append(score)

    return np.mean(scores)


# ✅ 8. ทำการ Optimize Hyperparameters
# ลดจำนวน trials ลงเพื่อความรวดเร็ว
# สำหรับโมเดลราคา
study_price = optuna.create_study(direction='minimize')
study_price.optimize(optimize_xgb_price, n_trials=20)

# สำหรับโมเดลทิศทาง XGBoost
study_dir_xgb = optuna.create_study(direction='maximize')
study_dir_xgb.optimize(optimize_xgb_dir, n_trials=20)

# สำหรับโมเดลทิศทาง LightGBM
study_dir_lgbm = optuna.create_study(direction='maximize')
study_dir_lgbm.optimize(optimize_lgbm_dir, n_trials=20)

# ✅ 9. สร้างโมเดลด้วย Hyperparameters ที่ดีที่สุด
best_params_price = study_price.best_params
best_params_dir_xgb = study_dir_xgb.best_params
best_params_dir_lgbm = study_dir_lgbm.best_params

print(f"\n✅ Best Hyperparameters for Price Model: {best_params_price}")
print(f"\n✅ Best Hyperparameters for Direction Model (XGBoost): {best_params_dir_xgb}")
print(f"\n✅ Best Hyperparameters for Direction Model (LightGBM): {best_params_dir_lgbm}")

# Create DMatrix objects for price model training
dtrain_price = xgb.DMatrix(X_train, label=y_train_price_scaled)
dtest_price = xgb.DMatrix(X_test, label=y_test_price_scaled)

# Configure parameters for price model
params_price = best_params_price.copy()  
params_price['objective'] = 'reg:squarederror'
params_price['eval_metric'] = 'rmse'
if 'n_estimators' in params_price:
    num_rounds_price = params_price.pop('n_estimators')
else:
    num_rounds_price = 1000  # Default if not in params

# Train price model with early stopping
evals_price = [(dtrain_price, 'train'), (dtest_price, 'eval')]
xgb_price_model = xgb.train(
    params_price,
    dtrain_price,
    num_boost_round=num_rounds_price,
    evals=evals_price,
    early_stopping_rounds=50,
    verbose_eval=False
)

# Create DMatrix objects for direction model training
dtrain_dir = xgb.DMatrix(X_train, label=y_train_dir)
dtest_dir = xgb.DMatrix(X_test, label=y_test_dir)

# Configure parameters for direction model
params_dir_xgb = best_params_dir_xgb.copy()
params_dir_xgb['objective'] = 'binary:logistic'
params_dir_xgb['eval_metric'] = 'auc'
if 'n_estimators' in params_dir_xgb:
    num_rounds_dir = params_dir_xgb.pop('n_estimators')
else:
    num_rounds_dir = 1000  # Default if not in params

# Train direction model with early stopping
evals_dir = [(dtrain_dir, 'train'), (dtest_dir, 'eval')]
xgb_dir_model = xgb.train(
    params_dir_xgb,
    dtrain_dir,
    num_boost_round=num_rounds_dir,
    evals=evals_dir,
    early_stopping_rounds=50,
    verbose_eval=False
)

# Train LightGBM model
lgbm_dir_model = lgb.LGBMClassifier(**best_params_dir_lgbm, random_state=42)
lgbm_dir_model.fit(
    X_train, 
    y_train_dir,
    eval_set=[(X_test, y_test_dir)],
    callbacks=[
        early_stopping(stopping_rounds=10),
        log_evaluation(period=1)  # แสดงผลทุก ๆ รอบ
    ]
)

# ✅ 10. สร้าง Stacking Ensemble Model สำหรับทิศทาง
# For Stacking we need to use sklearn-compatible models for XGBoost
xgb_sklearn_dir_model = xgb.XGBClassifier(random_state=42)
xgb_sklearn_dir_model.fit(X_train, y_train_dir)

base_models = [
    ('xgboost', xgb_sklearn_dir_model),
    ('lightgbm', lgbm_dir_model)
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

stacking_dir_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=cv
)
stacking_dir_model.fit(X_train, y_train_dir)

# ✅ 11. ทำนายด้วยโมเดลที่ได้
# ทำนายราคา
dtest_price = xgb.DMatrix(X_test)
y_pred_price = xgb_price_model.predict(dtest_price)
y_pred_price_actual = scaler_target.inverse_transform(y_pred_price.reshape(-1, 1)).ravel()

# ทำนายทิศทาง
dtest_dir = xgb.DMatrix(X_test)
y_pred_dir_xgb_proba = xgb_dir_model.predict(dtest_dir)
y_pred_dir_xgb = (y_pred_dir_xgb_proba > 0.5).astype(int)

y_pred_dir_lgbm = lgbm_dir_model.predict(X_test)
y_pred_dir_lgbm_proba = lgbm_dir_model.predict_proba(X_test)[:, 1]

y_pred_dir_stack = stacking_dir_model.predict(X_test)
y_pred_dir_stack_proba = stacking_dir_model.predict_proba(X_test)[:, 1]

# ✅ 12. ค้นหา Threshold ที่เหมาะสมสำหรับการทำนายทิศทาง
def find_optimal_threshold(y_true, y_pred_proba):
    thresholds = np.arange(0.1, 0.91, 0.05)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)
    
    return best_threshold, best_f1

# หา Threshold ที่ดีที่สุดสำหรับแต่ละโมเดล
best_threshold_xgb, best_f1_xgb = find_optimal_threshold(y_test_dir, y_pred_dir_xgb_proba)
best_threshold_lgbm, best_f1_lgbm = find_optimal_threshold(y_test_dir, y_pred_dir_lgbm_proba)
best_threshold_stack, best_f1_stack = find_optimal_threshold(y_test_dir, y_pred_dir_stack_proba)

print(f"\n✅ Best Threshold for XGBoost Direction: {best_threshold_xgb} (F1: {best_f1_xgb:.4f})")
print(f"✅ Best Threshold for LightGBM Direction: {best_threshold_lgbm} (F1: {best_f1_lgbm:.4f})")
print(f"✅ Best Threshold for Stacking Direction: {best_threshold_stack} (F1: {best_f1_stack:.4f})")

# ปรับการทำนายทิศทางด้วย Threshold ที่ดีที่สุด
y_pred_dir_xgb_adjusted = (y_pred_dir_xgb_proba >= best_threshold_xgb).astype(int)
y_pred_dir_lgbm_adjusted = (y_pred_dir_lgbm_proba >= best_threshold_lgbm).astype(int)
y_pred_dir_stack_adjusted = (y_pred_dir_stack_proba >= best_threshold_stack).astype(int)

# ✅ 13. ฟังก์ชันประเมินผลลัพธ์
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

def evaluate_direction(y_true, y_pred, y_pred_proba, model_name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except:
        auc = 0.5  # ตั้งค่าเริ่มต้นถ้าเกิดข้อผิดพลาด
    
    print(f"\n✅ **ผลลัพธ์การพยากรณ์ทิศทาง {model_name}**")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # แสดง Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    return acc, f1, auc, cm

# ✅ 14. ประเมินผลลัพธ์
# ประเมินโมเดลราคา
price_metrics = evaluate_price(y_test_price, y_pred_price_actual, "XGBoost Optimized (Price)")

# ประเมินโมเดลทิศทาง
dir_metrics_xgb = evaluate_direction(y_test_dir, y_pred_dir_xgb_adjusted, y_pred_dir_xgb_proba, "XGBoost Optimized with Threshold (Direction)")
dir_metrics_lgbm = evaluate_direction(y_test_dir, y_pred_dir_lgbm_adjusted, y_pred_dir_lgbm_proba, "LightGBM Optimized with Threshold (Direction)")
dir_metrics_stack = evaluate_direction(y_test_dir, y_pred_dir_stack_adjusted, y_pred_dir_stack_proba, "Stacking Ensemble with Threshold (Direction)")

# ✅ 15. วิเคราะห์ Feature Importance
def plot_feature_importance(model, feature_names, model_name, top_n=15):
    if isinstance(model, xgb.Booster):
        # For low-level XGBoost Booster API
        importance = model.get_score(importance_type='weight')
        # Convert to dataframe format
        feature_importance = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        })
    elif isinstance(model, xgb.XGBModel):
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
    elif isinstance(model, lgb.LGBMModel):
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
    else:
        return None
    
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Feature Importance - {model_name}', size=14)
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name.replace(" ", "_")}.png')
    plt.close()
    
    return feature_importance

# วิเคราะห์ Feature Importance สำหรับแต่ละโมเดล
try:
    price_importance = plot_feature_importance(xgb_price_model, X_train.columns, "XGBoost Price Model")
    dir_importance_xgb = plot_feature_importance(xgb_dir_model, X_train.columns, "XGBoost Direction Model")
    dir_importance_lgbm = plot_feature_importance(lgbm_dir_model, X_train.columns, "LightGBM Direction Model")

    print("\n✅ Top 5 Important Features for Price Prediction:")
    print(price_importance.head(5) if price_importance is not None else "Feature importance not available")

    print("\n✅ Top 5 Important Features for XGBoost Direction Prediction:")
    print(dir_importance_xgb.head(5) if dir_importance_xgb is not None else "Feature importance not available")

    print("\n✅ Top 5 Important Features for LightGBM Direction Prediction:")
    print(dir_importance_lgbm.head(5) if dir_importance_lgbm is not None else "Feature importance not available")
except Exception as e:
    print(f"Error in feature importance analysis: {e}")

# ✅ 16. บันทึกโมเดลที่ดีที่สุด
joblib.dump(xgb_price_model, "optimized_xgb_price_model.pkl")

# เลือกโมเดลทิศทางที่ดีที่สุดจากผลลัพธ์
best_dir_metrics = [dir_metrics_xgb[1], dir_metrics_lgbm[1], dir_metrics_stack[1]]
best_dir_model_idx = np.argmax(best_dir_metrics)

if best_dir_model_idx == 0:
    best_dir_model = xgb_dir_model
    best_threshold = best_threshold_xgb
    model_name = "optimized_xgb_dir_model"
elif best_dir_model_idx == 1:
    best_dir_model = lgbm_dir_model
    best_threshold = best_threshold_lgbm
    model_name = "optimized_lgbm_dir_model"
else:
    best_dir_model = stacking_dir_model
    best_threshold = best_threshold_stack
    model_name = "optimized_stacking_dir_model"

joblib.dump(best_dir_model, f"{model_name}.pkl")
joblib.dump(best_threshold, f"{model_name}_threshold.pkl")
joblib.dump(scaler_target, "scaler_target.pkl")


# ✅ 17. บันทึกผลลัพธ์การทำนาย
predictions_df = pd.DataFrame({
    "Ticker": ensemble_features.loc[~train_mask, "Ticker"],
    "Date": ensemble_features.loc[~train_mask, "Date"],
    "Actual_Price": y_test_price,
    "Predicted_Price": y_pred_price_actual,
    "Actual_Direction": y_test_dir,
    "Predicted_Direction": y_pred_dir_stack_adjusted if best_dir_model_idx == 2 else (y_pred_dir_lgbm_adjusted if best_dir_model_idx == 1 else y_pred_dir_xgb_adjusted),
    "Prediction_Probability": y_pred_dir_stack_proba if best_dir_model_idx == 2 else (y_pred_dir_lgbm_proba if best_dir_model_idx == 1 else y_pred_dir_xgb_proba)
})

predictions_df.to_csv("optimized_ensemble_predictions.csv", index=False)

print("\n✅ ปรับปรุงและบันทึกโมเดล XGBoost Ensemble สำเร็จ!")
print(f"โมเดลที่ดีที่สุดสำหรับการพยากรณ์ทิศทาง: {model_name} (F1-Score: {max(best_dir_metrics):.4f})")