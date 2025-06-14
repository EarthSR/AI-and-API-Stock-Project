import numpy as np 
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score,KFold
from sklearn.ensemble import StackingRegressor, StackingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso, Ridge
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from lightgbm import early_stopping, log_evaluation ,LGBMRegressor, LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from xgboost import XGBRegressor, XGBClassifier
from xgboost.callback import EarlyStopping


warnings.filterwarnings('ignore')

# ✅ 1. Load LSTM and GRU model results
predictions_lstm = pd.read_csv("../LSTM_model/all_predictions_per_day_multi_task.csv")
predictions_gru = pd.read_csv("../GRU_Model/all_predictions_per_day_multi_task.csv")

# ✅ 2. Prepare new Feature Set with Feature Engineering
def create_features(lstm_df, gru_df):
    # Combine results from LSTM and GRU
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

    # Basic comparison features
    df["LSTM_GRU_Price_Diff"] = abs(df["Predicted_Price_LSTM"] - df["Predicted_Price_GRU"])
    df["LSTM_GRU_Price_Ratio"] = df["Predicted_Price_LSTM"] / df["Predicted_Price_GRU"]
    df["LSTM_GRU_Dir_Match"] = (df["Predicted_Dir_LSTM"] == df["Predicted_Dir_GRU"]).astype(int)

    # Accuracy and momentum features
    df["LSTM_Accuracy_1d"] = 0.0
    df["GRU_Accuracy_1d"] = 0.0

    Ticker_groups = df.groupby("Ticker")
    for Ticker, group in Ticker_groups:
        Ticker_data = group.sort_values("Date")
        indices = Ticker_data.index

        # Accuracy comparison
        df.loc[indices, "LSTM_Accuracy_1d"] = (Ticker_data["Predicted_Dir_LSTM"].shift(1) == Ticker_data["Actual_Direction"].shift(1)).astype(float).values
        df.loc[indices, "GRU_Accuracy_1d"] = (Ticker_data["Predicted_Dir_GRU"].shift(1) == Ticker_data["Actual_Direction"].shift(1)).astype(float).values

        # Rolling directional accuracy
        df.loc[indices, "LSTM_Dir_Accuracy_3d"] = (
            (Ticker_data["Predicted_Dir_LSTM"].shift(1) == Ticker_data["Actual_Direction"].shift(1))
            .rolling(3).mean().values
        )
        df.loc[indices, "GRU_Dir_Accuracy_3d"] = (
            (Ticker_data["Predicted_Dir_GRU"].shift(1) == Ticker_data["Actual_Direction"].shift(1))
            .rolling(3).mean().values
        )

        # Price change features
        df.loc[indices, "Price_Pct_Change_1d"] = Ticker_data["Actual_Price"].pct_change(1).values
        df.loc[indices, "LSTM_Pred_Pct_Change_1d"] = Ticker_data["Predicted_Price_LSTM"].pct_change(1).values
        df.loc[indices, "GRU_Pred_Pct_Change_1d"] = Ticker_data["Predicted_Price_GRU"].pct_change(1).values

        # Technical indicator - Simple SMA diff
        df.loc[indices, "LSTM_Price_SMA_Diff_5_10"] = (
            Ticker_data["Predicted_Price_LSTM"].shift(1).rolling(5).mean() -
            Ticker_data["Predicted_Price_LSTM"].shift(1).rolling(10).mean()
        ).values
        df.loc[indices, "GRU_Price_SMA_Diff_5_10"] = (
            Ticker_data["Predicted_Price_GRU"].shift(1).rolling(5).mean() -
            Ticker_data["Predicted_Price_GRU"].shift(1).rolling(10).mean()
        ).values

        # Direction consistency
        df.loc[indices, "LSTM_Dir_Consistency_3d"] = Ticker_data["Predicted_Dir_LSTM"].shift(1).rolling(3).mean().values
        df.loc[indices, "GRU_Dir_Consistency_3d"] = Ticker_data["Predicted_Dir_GRU"].shift(1).rolling(3).mean().values

    # Time-based features
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Is_Day_0"] = (df["DayOfWeek"] == 0).astype(int)  # Monday
    df["Is_Day_4"] = (df["DayOfWeek"] == 4).astype(int)  # Friday

    df["DayOfMonth"] = df["Date"].dt.day
    df["IsFirstHalfOfMonth"] = (df["DayOfMonth"] <= 15).astype(int)
    df["IsSecondHalfOfMonth"] = (df["DayOfMonth"] > 15).astype(int)

    # Handle Missing Values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df.groupby("Ticker")[col].transform(lambda x: x.fillna(0))


    return df

# Create new features
ensemble_features = create_features(predictions_lstm, predictions_gru)

# ✅ 3. Remove Outliers using Winsorization
def handle_outliers(df, cols, lower_quantile=0.005, upper_quantile=0.995):
    df_clean = df.copy()
    for col in cols:
        lower_bound = df_clean[col].quantile(lower_quantile)
        upper_bound = df_clean[col].quantile(upper_quantile)
        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
    return df_clean

price_cols = ["Predicted_Price_LSTM", "Predicted_Price_GRU", "Actual_Price"]
ensemble_features = handle_outliers(ensemble_features, price_cols)

# ✅ 4. Split Train/Test data up to December 1, 2024
train_cutoff = pd.Timestamp("2024-12-01")
train_mask = ensemble_features["Date"] < train_cutoff

# Check Direction balance
dir_balance = ensemble_features.loc[train_mask, "Actual_Direction"].value_counts(normalize=True)
print(f"Direction Balance in Training Data: {dir_balance}")

# Get Features and Target
feature_cols = [col for col in ensemble_features.columns if col not in 
                ["Ticker", "Date", "Actual_Price", "Actual_Direction"]]

# Remove columns with too many missing values
cols_to_drop = []
for col in feature_cols:
    if ensemble_features[col].isnull().sum() / len(ensemble_features) > 0.3:
        cols_to_drop.append(col)

feature_cols = [col for col in feature_cols if col not in cols_to_drop]
print(f"Removed {len(cols_to_drop)} columns with too many missing values")

# Check for low variance columns
variance_threshold = 0.01
cols_low_variance = []
for col in feature_cols:
    if ensemble_features[col].var() < variance_threshold:
        cols_low_variance.append(col)

feature_cols = [col for col in feature_cols if col not in cols_low_variance]
print(f"Removed {len(cols_low_variance)} columns with too low variance")

X_train = ensemble_features.loc[train_mask, feature_cols].copy()
y_train_price = ensemble_features.loc[train_mask, "Actual_Price"].values
y_train_dir = ensemble_features.loc[train_mask, "Actual_Direction"].values

X_test = ensemble_features.loc[~train_mask, feature_cols].copy()
y_test_price = ensemble_features.loc[~train_mask, "Actual_Price"].values
y_test_dir = ensemble_features.loc[~train_mask, "Actual_Direction"].values

# ✅ 5. Apply Scaler to data
try:
    # Load existing scalers if available
    scaler_features = joblib.load("scaler_features.pkl")
    scaler_target = joblib.load("../LSTM_model/scaler_target.pkl")
except:
    # Create new scalers if not found
    scaler_features = StandardScaler()  # StandardScaler for features
    scaler_features.fit(X_train)

    scaler_target = RobustScaler()  # RobustScaler for target
    scaler_target.fit(y_train_price.reshape(-1, 1))  

# ถ้า y_train_price เป็น ndarray ให้สร้าง Series ที่มี index
y_train_price = pd.Series(y_train_price, index=X_train.index)
y_test_price = pd.Series(y_test_price, index=X_test.index)

# Scale features
X_train_scaled = pd.DataFrame(
    scaler_features.transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler_features.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# ✅ Scale closing prices อย่างถูกต้อง
y_train_price_scaled = pd.Series(
    scaler_target.transform(y_train_price.values.reshape(-1, 1)).ravel(),
    index=y_train_price.index
)

y_test_price_scaled = pd.Series(
    scaler_target.transform(y_test_price.values.reshape(-1, 1)).ravel(),
    index=y_test_price.index
)


# ✅ 6. Prepare TimeSeriesSplit for Cross-validation
tscv = TimeSeriesSplit(n_splits=5)
# Fix 2: Fix XGBoost Wrapper classes (updated comments and made them English)
class XGBoostWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **params):
        self.params = params
        self.model = None

    def fit(self, X, y):
        # Create XGBRegressor for training
        self.model = XGBRegressor(**self.params)
        self.model.fit(X, y)  # Use .fit() from XGBRegressor
        return self

    def predict(self, X):
        return self.model.predict(X)  # Use .predict() from XGBRegressor

# XGBoost Classifier Wrapper
class XGBoostClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **params):
        self.params = params
        self.model = None

    def fit(self, X, y):
        # Create XGBClassifier for training
        self.model = XGBClassifier(**self.params)
        self.model.fit(X, y)  # Use .fit() from XGBClassifier
        return self

    def predict_proba(self, X):
        preds = self.model.predict_proba(X)  # Use .predict_proba() from XGBClassifier
        return preds

    def predict(self, X):
        return self.model.predict(X)  # Use .predict() from XGBClassifier

# ✅ 7. Optimize Price Model with XGBoost
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
    for train_idx, valid_idx in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[valid_idx]
        y_tr, y_val = y_train_price_scaled.iloc[train_idx], y_train_price_scaled.iloc[valid_idx]
        
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

# ✅ 8. Optimize Price Model with LightGBM
def optimize_lgbm_price(trial):
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
        'objective': 'regression',
        'metric': 'rmse',
        'random_state': 42
    }

    model = lgb.LGBMRegressor(**param)

    scores = []
    for train_idx, valid_idx in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[valid_idx]
        y_tr, y_val = y_train_price_scaled.iloc[train_idx], y_train_price_scaled.iloc[valid_idx]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)]
        )

        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        scores.append(rmse)

    return np.mean(scores)

# ✅ 9. Optimize Direction Model with XGBoost
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
    for train_idx, valid_idx in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[valid_idx]
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

# ✅ 10. Optimize Direction Model with LightGBM
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
        # เพิ่มพารามิเตอร์ต่อไปนี้เพื่อจัดการกับคำเตือน "No further splits with positive gain"
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 1e-8, 1e-3, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1e-1, log=True),
        'class_weight': 'balanced',
        'objective': 'binary',
        'metric': 'auc',
        'random_state': 42,
        'verbose': -1  # ปิดการแสดงข้อความเตือน
    }

    model = lgb.LGBMClassifier(**param)

    scores = []
    for train_idx, valid_idx in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[valid_idx]
        y_tr, y_val = y_train_dir[train_idx], y_train_dir[valid_idx]


        # ตรวจสอบข้อมูลว่ามีความไม่สมดุลหรือไม่
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            continue  # ข้ามการแบ่งนี้ถ้าข้อมูลไม่สมดุล

        try:
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)]
            )

            pred_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, pred_proba)
            scores.append(score)
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการฝึกโมเดล: {e}")
            continue

    # ถ้าไม่มีคะแนนที่คำนวณได้ ให้ส่งคืนค่าต่ำ
    if len(scores) == 0:
        return 0.0

    return np.mean(scores)

# ✅ 11. Optimize Direction Model with Random Forest
def optimize_rf_dir(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': 'balanced',
        'random_state': 42
    }
    
    model = RandomForestClassifier(**param)
    
    scores = []
    for train_idx, valid_idx in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[valid_idx]
        y_tr, y_val = y_train_dir[train_idx], y_train_dir[valid_idx]
        
        model.fit(X_tr, y_tr)
        
        pred_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, pred_proba)
        scores.append(score)
    
    return np.mean(scores)

# ✅ 12. Optimize Hyperparameters
# Reduce number of trials for speed
n_trials = 20

# For Price Model (XGBoost)
study_price_xgb = optuna.create_study(direction='minimize')
study_price_xgb.optimize(optimize_xgb_price, n_trials=n_trials)

# For Price Model (LightGBM)
study_price_lgbm = optuna.create_study(direction='minimize')
study_price_lgbm.optimize(optimize_lgbm_price, n_trials=n_trials)

# For Direction Model (XGBoost)
study_dir_xgb = optuna.create_study(direction='maximize')
study_dir_xgb.optimize(optimize_xgb_dir, n_trials=n_trials)

# For Direction Model (LightGBM)
study_dir_lgbm = optuna.create_study(direction='maximize')
study_dir_lgbm.optimize(optimize_lgbm_dir, n_trials=n_trials)

# For Direction Model (Random Forest)
study_dir_rf = optuna.create_study(direction='maximize')
study_dir_rf.optimize(optimize_rf_dir, n_trials=n_trials)

# ✅ 13. Create models with best hyperparameters
best_params_price_xgb = study_price_xgb.best_params
best_params_price_lgbm = study_price_lgbm.best_params
best_params_dir_xgb = study_dir_xgb.best_params
best_params_dir_lgbm = study_dir_lgbm.best_params
best_params_dir_rf = study_dir_rf.best_params

print(f"\n✅ Best Hyperparameters for Price Model (XGBoost): {best_params_price_xgb}")
print(f"\n✅ Best Hyperparameters for Price Model (LightGBM): {best_params_price_lgbm}")
print(f"\n✅ Best Hyperparameters for Direction Model (XGBoost): {best_params_dir_xgb}")
print(f"\n✅ Best Hyperparameters for Direction Model (LightGBM): {best_params_dir_lgbm}")
print(f"\n✅ Best Hyperparameters for Direction Model (Random Forest): {best_params_dir_rf}")

# 14. XGBoost Price Model
params_price_xgb = best_params_price_xgb.copy()
params_price_xgb['objective'] = 'reg:squarederror'
params_price_xgb['eval_metric'] = 'rmse'
xgb_price_model = XGBRegressor(**params_price_xgb)
xgb_price_model.fit(X_train_scaled, y_train_price_scaled)  # เพิ่มบรรทัดนี้

# 15. LightGBM Price Model
params_price_lgbm = best_params_price_lgbm.copy()
params_price_lgbm['objective'] = 'regression'
params_price_lgbm['metric'] = 'rmse'
lgbm_price_model = LGBMRegressor(**params_price_lgbm)
lgbm_price_model.fit(X_train_scaled, y_train_price_scaled)  # เพิ่มบรรทัดนี้

# 16. XGBoost Direction Model
params_dir_xgb = best_params_dir_xgb.copy()
params_dir_xgb['objective'] = 'binary:logistic'
params_dir_xgb['eval_metric'] = 'auc'
xgb_dir_model = XGBClassifier(**params_dir_xgb)
xgb_dir_model.fit(X_train_scaled, y_train_dir)  # เพิ่มบรรทัดนี้

# 17. LightGBM Direction Model
params_dir_lgbm = best_params_dir_lgbm.copy()
params_dir_lgbm['objective'] = 'binary'
params_dir_lgbm['metric'] = 'auc'
lgbm_dir_model = LGBMClassifier(**params_dir_lgbm)
lgbm_dir_model.fit(X_train_scaled, y_train_dir)  # เพิ่มบรรทัดนี้

# 18. Random Forest Direction Model
params_dir_rf = best_params_dir_rf.copy()
params_dir_rf['class_weight'] = 'balanced'
rf_dir_model = RandomForestClassifier(**params_dir_rf)
rf_dir_model.fit(X_train_scaled, y_train_dir)  # เพิ่มบรรทัดนี้

# 19. Create Stacking Model for Price
price_estimators = [
    ('xgb', XGBRegressor(**params_price_xgb)),
    ('lgbm', LGBMRegressor(**params_price_lgbm))
]

stacking_price_model = StackingRegressor(
    estimators=price_estimators,
    final_estimator=Ridge(),
    cv=KFold(n_splits=3),  # เปลี่ยนมาใช้ KFold
    n_jobs=1
)

stacking_price_model.fit(
    X_train_scaled, y_train_price_scaled
)

# 20. Create Stacking Model for Direction
dir_estimators = [
    ('xgb', XGBClassifier(**params_dir_xgb)),
    ('lgbm', LGBMClassifier(**params_dir_lgbm)),
    ('rf', RandomForestClassifier(**params_dir_rf))
]

stacking_dir_model = StackingClassifier(
    estimators=dir_estimators,
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=KFold(n_splits=3),  # เปลี่ยนมาใช้ KFold เช่นกัน
    n_jobs=1
)

stacking_dir_model.fit(
    X_train_scaled, y_train_dir
)

# ✅ 21. Make predictions and evaluate models
def evaluate_models_by_Ticker():
    # Create DMatrix for XGBoost models
    dtest_price = xgb.DMatrix(X_test_scaled)
    dtest_dir = xgb.DMatrix(X_test_scaled)
    
    # Predict prices
    xgb_price_pred = xgb_price_model.predict(X_test_scaled)
    lgbm_price_pred = lgbm_price_model.predict(X_test_scaled)
    stacking_price_pred = stacking_price_model.predict(X_test_scaled)
    
    # Convert back to actual prices
    xgb_price_pred = scaler_target.inverse_transform(xgb_price_pred.reshape(-1, 1)).ravel()
    lgbm_price_pred = scaler_target.inverse_transform(lgbm_price_pred.reshape(-1, 1)).ravel()
    stacking_price_pred = scaler_target.inverse_transform(stacking_price_pred.reshape(-1, 1)).ravel()
    
    # Predict directions
    xgb_dir_proba = xgb_dir_model.predict_proba(X_test_scaled)[:, 1]
    xgb_dir_pred = (xgb_dir_proba >= 0.5).astype(int)
    
    lgbm_dir_proba = lgbm_dir_model.predict_proba(X_test_scaled)[:, 1]
    lgbm_dir_pred = (lgbm_dir_proba >= 0.5).astype(int)
    
    rf_dir_proba = rf_dir_model.predict_proba(X_test_scaled)[:, 1]
    rf_dir_pred = (rf_dir_proba >= 0.5).astype(int)
    
    stacking_dir_proba = stacking_dir_model.predict_proba(X_test_scaled)[:, 1]
    stacking_dir_pred = (stacking_dir_proba >= 0.5).astype(int)
    
    # Create DataFrame for results
    results_df = ensemble_features.loc[~train_mask].copy()
    results_df['XGBoost_Price_Pred'] = xgb_price_pred
    results_df['LightGBM_Price_Pred'] = lgbm_price_pred
    results_df['Stacking_Price_Pred'] = stacking_price_pred
    results_df['XGBoost_Dir_Pred'] = xgb_dir_pred
    results_df['LightGBM_Dir_Pred'] = lgbm_dir_pred
    results_df['RandomForest_Dir_Pred'] = rf_dir_pred
    results_df['Stacking_Dir_Pred'] = stacking_dir_pred
    results_df['Actual_Price'] = y_test_price
    results_df['Actual_Direction'] = y_test_dir
    
    # Get unique Ticker symbols
    Tickers = results_df['Ticker'].unique()
    
    print(f"📊 Evaluation Results by Ticker (Total: {len(Tickers)} Tickers)")
    
    # Store metrics for each Ticker
    Ticker_metrics = {}
    
    # Evaluate for each Ticker
    for Ticker in Tickers:
        Ticker_data = results_df[results_df['Ticker'] == Ticker]
        
        if len(Ticker_data) < 5:  # Skip Tickers with very few data points
            continue
            
        Ticker_metrics[Ticker] = {
            'price': {},
            'direction': {},
            'sample_size': len(Ticker_data)
        }
        
        print(f"\n\n🔍 Evaluating {Ticker} (Sample size: {len(Ticker_data)})")
        
        # Evaluate price models for this Ticker
        print("\n📈 Price Model Evaluation:")
        for name, pred_col in [('XGBoost', 'XGBoost_Price_Pred'), 
                           ('LightGBM', 'LightGBM_Price_Pred'), 
                           ('Stacking', 'Stacking_Price_Pred')]:
            
            mae = mean_absolute_error(Ticker_data['Actual_Price'], Ticker_data[pred_col])
            rmse = np.sqrt(mean_squared_error(Ticker_data['Actual_Price'], Ticker_data[pred_col]))
            r2 = r2_score(Ticker_data['Actual_Price'], Ticker_data[pred_col])
            
            Ticker_metrics[Ticker]['price'][name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
            
            print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        # Evaluate direction models for this Ticker
        print("\n📊 Direction Model Evaluation:")
        for name, pred_col, proba_col in [
            ('XGBoost', 'XGBoost_Dir_Pred', None),
            ('LightGBM', 'LightGBM_Dir_Pred', None),
            ('RandomForest', 'RandomForest_Dir_Pred', None),
            ('Stacking', 'Stacking_Dir_Pred', None)]:
            
            # Skip if Ticker has only one class in actual direction
            if len(Ticker_data['Actual_Direction'].unique()) < 2:
                print(f"{name} - Cannot evaluate (insufficient class diversity in test data)")
                continue
                
            acc = accuracy_score(Ticker_data['Actual_Direction'], Ticker_data[pred_col])
            
            # Check if we can calculate F1 score (requires both classes to be present)
            try:
                f1 = f1_score(Ticker_data['Actual_Direction'], Ticker_data[pred_col])
            except:
                f1 = float('nan')
            
            # We don't have probabilities stored in the dataframe, so we skip AUC
            
            Ticker_metrics[Ticker]['direction'][name] = {
                'Accuracy': acc,
                'F1': f1
            }
            
            print(f"{name} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
            
            # Show confusion matrix if there are enough samples
    
    # Find best performing models per Ticker
    print("\n\n🏆 Best Performing Models by Ticker:")
    
    for Ticker, metrics in Ticker_metrics.items():
        print(f"\n{Ticker} (Sample size: {metrics['sample_size']})")
        
        # Best price model (by RMSE)
        if metrics['price']:
            best_price_model = min(metrics['price'].items(), key=lambda x: x[1]['RMSE'])
            print(f"Best Price Model: {best_price_model[0]} - RMSE: {best_price_model[1]['RMSE']:.4f}")
        
        # Best direction model (by accuracy)
        if metrics['direction']:
            best_dir_model = max(metrics['direction'].items(), key=lambda x: x[1]['Accuracy'])
            print(f"Best Direction Model: {best_dir_model[0]} - Accuracy: {best_dir_model[1]['Accuracy']:.4f}")
    
    return results_df, Ticker_metrics

# Evaluate models
ensemble_results = evaluate_models_by_Ticker()

# ✅ 22. Save models and results
joblib.dump(xgb_price_model, 'xgb_price_model.pkl')
joblib.dump(lgbm_price_model, 'lgbm_price_model.pkl')
joblib.dump(stacking_price_model, 'stacking_price_model.pkl')

joblib.dump(xgb_dir_model, 'xgb_dir_model.pkl')
joblib.dump(lgbm_dir_model, 'lgbm_dir_model.pkl')
joblib.dump(rf_dir_model, 'rf_dir_model.pkl')
joblib.dump(stacking_dir_model, 'stacking_dir_model.pkl')

joblib.dump(scaler_features, 'scaler_features.pkl')

# Save results
ensemble_results.to_csv('ensemble_predictions.csv', index=False)

# ✅ 23. สร้าง Feature Importance Plot
def plot_feature_importance():
    plt.figure(figsize=(15, 20))
    
    # XGBoost Price Importance
    plt.subplot(3, 2, 1)
    xgb.plot_importance(xgb_price_model, max_num_features=20)
    plt.title('XGBoost Price Model Feature Importance')
    
    # LightGBM Price Importance
    plt.subplot(3, 2, 2)
    lgb.plot_importance(lgbm_price_model, max_num_features=20)
    plt.title('LightGBM Price Model Feature Importance')
    
    # XGBoost Direction Importance
    plt.subplot(3, 2, 3)
    xgb.plot_importance(xgb_dir_model, max_num_features=20)
    plt.title('XGBoost Direction Model Feature Importance')
    
    # LightGBM Direction Importance
    plt.subplot(3, 2, 4)
    lgb.plot_importance(lgbm_dir_model, max_num_features=20)
    plt.title('LightGBM Direction Model Feature Importance')
    
    # Random Forest Direction Importance
    plt.subplot(3, 2, 5)
    importances = rf_dir_model.feature_importances_
    indices = np.argsort(importances)[-20:]
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
    plt.title('Random Forest Direction Model Feature Importance')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

plot_feature_importance()

def plot_feature_importance(model, feature_names, title="Feature Importance", top_n=20):
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    importance = importance.sort_values(by='Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance, palette="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ✅ 24. สร้าง Actual vs Predicted Plot สำหรับราคา
def plot_actual_vs_predicted():
    plt.figure(figsize=(15, 6))
    
    # เลือกตัวอย่างบางส่วนเพื่อพล็อต (100 ตัวอย่างแรก)
    sample_size = min(100, len(y_test_price))
    idx = np.random.choice(len(y_test_price), sample_size, replace=False)
    
    plt.plot(y_test_price[idx], label='Actual Price', marker='o')
    plt.plot(ensemble_results['XGBoost_Price_Pred'].values[idx], label='XGBoost Pred', marker='x')
    plt.plot(ensemble_results['LightGBM_Price_Pred'].values[idx], label='LightGBM Pred', marker='^')
    plt.plot(ensemble_results['Stacking_Price_Pred'].values[idx], label='Stacking Pred', marker='s')
    
    plt.title('Actual vs Predicted Prices (Test Set)')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.savefig('actual_vs_predicted_prices.png')
    plt.close()

plot_actual_vs_predicted()

print("\n✅ Ensemble modeling completed successfully!")