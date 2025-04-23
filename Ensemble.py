import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import OneHotEncoder
import joblib

# ===============================
# 🔧 Load ข้อมูล
# ===============================
lstm_model = pd.read_csv("LSTM_Model/predictions_multi_task_walkforward_batch.csv")
gru_model  = pd.read_csv("GRU_Model/predictions_multi_task_walkforward_batch.csv")

# Targets
y_true_price = lstm_model["Actual_Price"].values
y_true_dir   = lstm_model["Actual_Dir"].values
tickers      = lstm_model["Ticker"]

# Meta Features
X_meta = pd.DataFrame({
    "lstm_price": lstm_model["Predicted_Price"],
    "gru_price":  gru_model["Predicted_Price"],
    "lstm_dir":   lstm_model["Predicted_Dir"],
    "gru_dir":    gru_model["Predicted_Dir"]
})

# One-hot encode ticker
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ticker_encoded = encoder.fit_transform(tickers.values.reshape(-1, 1))
ticker_df = pd.DataFrame(ticker_encoded, columns=encoder.get_feature_names_out(["Ticker"]))

# รวมเข้า X
X_full = pd.concat([X_meta.reset_index(drop=True), ticker_df.reset_index(drop=True)], axis=1)

# ===============================
# 🎯 K-Fold Meta-Model
# ===============================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

price_metrics = []
dir_metrics   = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_full)):
    print(f"\n🔁 Fold {fold + 1}")

    # Split
    X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
    y_train_price, y_val_price = y_true_price[train_idx], y_true_price[val_idx]
    y_train_dir, y_val_dir = y_true_dir[train_idx], y_true_dir[val_idx]

    # -------- Regression Model --------
    regressor = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    regressor.fit(X_train.drop(columns=["lstm_dir", "gru_dir"]), y_train_price)
    y_pred_price = regressor.predict(X_val.drop(columns=["lstm_dir", "gru_dir"]))

    # Metrics
    mae = mean_absolute_error(y_val_price, y_pred_price)
    rmse = np.sqrt(mean_squared_error(y_val_price, y_pred_price))
    r2 = r2_score(y_val_price, y_pred_price)
    price_metrics.append([mae, rmse, r2])

    # -------- Classification Model --------
    classifier = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3,
                               use_label_encoder=False, eval_metric='logloss', random_state=42)
    classifier.fit(X_train.drop(columns=["lstm_price", "gru_price"]), y_train_dir)
    y_pred_dir = classifier.predict(X_val.drop(columns=["lstm_price", "gru_price"]))

    acc = accuracy_score(y_val_dir, y_pred_dir)
    f1 = f1_score(y_val_dir, y_pred_dir)
    prec = precision_score(y_val_dir, y_pred_dir)
    rec = recall_score(y_val_dir, y_pred_dir)
    dir_metrics.append([acc, f1, prec, rec])

joblib.dump(regressor, "meta_price_model_xgb.pkl")
joblib.dump(classifier, "meta_direction_model_xgb.pkl")

# ===============================
# 🧾 สรุปผลเฉลี่ย
# ===============================
price_metrics = np.array(price_metrics)
dir_metrics   = np.array(dir_metrics)

print("\n📊 [K-Fold Average Results]")
print("📈 Price Regression:")
print(f"  MAE  : {price_metrics[:, 0].mean():.4f}")
print(f"  RMSE : {price_metrics[:, 1].mean():.4f}")
print(f"  R2   : {price_metrics[:, 2].mean():.4f}")

print("\n🧭 Direction Classification:")
print(f"  Acc  : {dir_metrics[:, 0].mean():.4f}")
print(f"  F1   : {dir_metrics[:, 1].mean():.4f}")
print(f"  Prec : {dir_metrics[:, 2].mean():.4f}")
print(f"  Rec  : {dir_metrics[:, 3].mean():.4f}")
