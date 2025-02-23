import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
import optuna
import shap
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 🚀 1️⃣ โหลดข้อมูล
stock_df = pd.read_csv("Thai_stock.csv")
news_df = pd.read_csv("Thai_News_WithSentiment_Cleaned_Filter.csv")

print("✅ โหลดข้อมูลสำเร็จ!")

# 🔥 2️⃣ Clean Data
stock_df["Date"] = pd.to_datetime(stock_df["Date"], errors="coerce")
news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")

stock_df.dropna(subset=["Date"], inplace=True)
news_df.dropna(subset=["date"], inplace=True)

stock_df["Market Cap"] = stock_df["Market Cap"].fillna(stock_df["Market Cap"].median())

news_df["sentiment"] = pd.to_numeric(news_df["sentiment"], errors="coerce").fillna(0)
news_df["confidence"] = pd.to_numeric(news_df["confidence"], errors="coerce").fillna(0)

# 🔥 3️⃣ รวมข้อมูล
merged_df = stock_df.merge(news_df, left_on="Date", right_on="date", how="left")
merged_df["Sentiment Weighted"] = merged_df["sentiment"] * merged_df["confidence"]

# 🔥 4️⃣ Feature Engineering: ใช้ Log Return แทน Price Change %
merged_df["Log Return"] = np.log(merged_df["Close"] / merged_df["Close"].shift(1))

# เพิ่ม Technical Indicators
merged_df["RSI"] = 100 - (100 / (1 + merged_df["Close"].pct_change().rolling(14).mean()))
merged_df["MACD"] = merged_df["Close"].ewm(span=12, adjust=False).mean() - merged_df["Close"].ewm(span=26, adjust=False).mean()

# 🔥 5️⃣ Feature Selection: ใช้ SHAP เพื่อตรวจสอบความสำคัญของฟีเจอร์
features = ["Close", "Volume", "Market Cap", "Sentiment Weighted", "RSI", "MACD"]
target = "Log Return"

merged_df.fillna(0, inplace=True)

scaler = StandardScaler()
merged_df[features] = scaler.fit_transform(merged_df[features])

# 🔥 6️⃣ เตรียมข้อมูลสำหรับ Training
sequence_length = 10
X, y = [], []

for i in range(sequence_length, len(merged_df)):
    X.append(merged_df[features].iloc[i-sequence_length:i].values)
    y.append(merged_df[target].iloc[i])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔥 7️⃣ สร้างโมเดล LSTM (เพิ่ม Attention Mechanism)
lstm_model = Sequential([
    Input(shape=(sequence_length, len(features))),
    Bidirectional(LSTM(64, return_sequences=True, activation='tanh')),
    Dropout(0.2),
    Bidirectional(LSTM(32, activation='tanh')),
    Dropout(0.2),
    Dense(1)
])

lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 🔥 8️⃣ ใช้ LSTM Extract Features
X_train_lstm = lstm_model.predict(X_train)
X_test_lstm = lstm_model.predict(X_test)

# 🔥 9️⃣ ใช้ Optuna หาค่าที่ดีที่สุดของ XGBoost
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'eval_metric': 'rmse'
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train_lstm, y_train, eval_set=[(X_test_lstm, y_test)], verbose=False)
    preds = model.predict(X_test_lstm)
    return np.sqrt(np.mean((y_test - preds) ** 2))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
best_params = study.best_params

# 🔥 🔟 ใช้ค่าที่ดีที่สุดของ XGBoost
xgb_model = xgb.XGBRegressor(**best_params)
xgb_model.fit(X_train_lstm, y_train, eval_set=[(X_test_lstm, y_test)], verbose=True)

# 🔥 11️⃣ ใช้ Stacking (รวม XGBoost + RandomForest)
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train_lstm, y_train)

y_pred_xgb = xgb_model.predict(X_test_lstm)
y_pred_rf = rf_model.predict(X_test_lstm)

# 🔥 12️⃣ Ensemble Learning
y_pred_final = (y_pred_xgb * 0.7) + (y_pred_rf * 0.3)

# 🔥 13️⃣ แสดงผลลัพธ์
plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual Log Return")
plt.plot(y_pred_final, label="Predicted Log Return", linestyle="dashed")
plt.legend()
plt.show()
