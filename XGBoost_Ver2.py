import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
import optuna
import shap
import h2o
from h2o.automl import H2OAutoML
from tpot import TPOTRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Attention
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging

# 🔥 1️⃣ ตั้งค่าระบบ Log
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

# 🚀 2️⃣ โหลดข้อมูล
logger.info("📥 กำลังโหลดข้อมูล...")
stock_df = pd.read_csv("Thai_stock.csv")
news_df = pd.read_csv("Thai_News_WithSentiment_Cleaned_Filter.csv")
logger.info("✅ โหลดข้อมูลสำเร็จ!")

# 🔥 3️⃣ Clean Data
stock_df["Date"] = pd.to_datetime(stock_df["Date"], errors="coerce")
news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")

stock_df.dropna(subset=["Date"], inplace=True)
news_df.dropna(subset=["date"], inplace=True)

stock_df["Market Cap"].fillna(stock_df["Market Cap"].median(), inplace=True)

news_df["sentiment"] = pd.to_numeric(news_df["sentiment"], errors="coerce").fillna(0)
news_df["confidence"] = pd.to_numeric(news_df["confidence"], errors="coerce").fillna(0)

# 🔥 4️⃣ รวมข้อมูล
merged_df = stock_df.merge(news_df, left_on="Date", right_on="date", how="left")
merged_df["Sentiment Weighted"] = merged_df["sentiment"] * merged_df["confidence"]

# 🔥 5️⃣ Feature Engineering
merged_df["Log Return"] = np.log(merged_df["Close"] / merged_df["Close"].shift(1))
merged_df["RSI"] = 100 - (100 / (1 + merged_df["Close"].pct_change().rolling(14).mean()))
merged_df["MACD"] = merged_df["Close"].ewm(span=12, adjust=False).mean() - merged_df["Close"].ewm(span=26, adjust=False).mean()

# 🔥 6️⃣ Feature Selection & Scaling
features = ["Close", "Volume", "Market Cap", "Sentiment Weighted", "RSI", "MACD"]
target = "Log Return"
merged_df.fillna(0, inplace=True)
scaler = StandardScaler()
merged_df[features] = scaler.fit_transform(merged_df[features])

# 🔥 7️⃣ เตรียมข้อมูลสำหรับ Training
sequence_length = 10
X, y = [], []
for i in range(sequence_length, len(merged_df)):
    X.append(merged_df[features].iloc[i-sequence_length:i].values)
    y.append(merged_df[target].iloc[i])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔥 8️⃣ โมเดล LSTM + Attention Mechanism
def build_lstm():
    inputs = Input(shape=(sequence_length, len(features)))
    x = Bidirectional(LSTM(64, return_sequences=True, activation='tanh'))(inputs)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(32, return_sequences=True, activation='tanh'))(x)
    attention = Attention()([x, x])
    x = tf.keras.layers.Flatten()(attention)
    x = Dense(1)(x)
    model = tf.keras.Model(inputs, x)
    model.compile(optimizer="adam", loss="mse")
    return model

lstm_model = build_lstm()
lstm_model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test), verbose=1)
X_train_lstm = lstm_model.predict(X_train)
X_test_lstm = lstm_model.predict(X_test)

# 🔥 9️⃣ ใช้ AutoML (H2O & TPOT)
logger.info("⚡ กำลังรัน H2O AutoML...")
h2o.init()
h2o_train = h2o.H2OFrame(pd.DataFrame(X_train_lstm))
h2o_train["y"] = h2o.H2OFrame(pd.DataFrame(y_train))  # ✅ แปลง y_train เป็น H2OFrame

aml = H2OAutoML(max_models=10, seed=42)
aml.train(y="y", training_frame=h2o_train)
best_h2o_model = aml.leader

logger.info("⚡ กำลังรัน TPOT AutoML...")
tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2, random_state=42)
tpot.fit(X_train_lstm, y_train)

# 🔥 11️⃣ Stacking Model (H2O + TPOT + XGBoost)
y_pred_h2o = best_h2o_model.predict(h2o.H2OFrame(pd.DataFrame(X_test_lstm))).as_data_frame().values.flatten()
y_pred_tpot = tpot.predict(X_test_lstm)

# 🔥 12️⃣ แสดงผลลัพธ์
plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual Log Return")
plt.plot(y_pred_h2o, label="Predicted Log Return (H2O)", linestyle="dashed")
plt.plot(y_pred_tpot, label="Predicted Log Return (TPOT)", linestyle="dotted")
plt.legend()
plt.show()

logger.info(f"🎯 Final RMSE (H2O): {np.sqrt(np.mean((y_test - y_pred_h2o) ** 2)):.5f}")
logger.info(f"🎯 Final RMSE (TPOT): {np.sqrt(np.mean((y_test - y_pred_tpot) ** 2)):.5f}")