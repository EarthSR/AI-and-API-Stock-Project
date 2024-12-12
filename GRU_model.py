import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
import ta
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, filename='training.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ฟังก์ชันสร้างลำดับข้อมูล (Time Series Sequences)
def create_sequences(features, targets, seq_length=10):
    X, y = [], []
    for i in range(len(features) - seq_length):
        if i + seq_length < len(targets):
            X.append(features[i:i + seq_length])
            y.append(targets[i + seq_length])
    return np.array(X), np.array(y)

# ฟังก์ชันสร้างโมเดล GRU
def build_GRU_model(input_shape):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ฟังก์ชันการพล็อตการฝึกฝน
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    # กราฟ Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # กราฟ MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE During Training')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

# ฟังก์ชันการพล็อตการทำนาย
def plot_predictions(y_true, y_pred, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='red', alpha=0.7)
    plt.title(f'True vs Predicted Prices for {ticker}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# ฟังก์ชันการพล็อต Residuals
def plot_residuals(y_true, y_pred, ticker):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(residuals)), residuals, alpha=0.5)
    plt.hlines(y=0, xmin=0, xmax=len(residuals), colors='red')
    plt.title(f'Residuals for {ticker}')
    plt.xlabel('Sample')
    plt.ylabel('Residual')
    plt.show()

# ตรวจสอบ GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.set_memory_growth(physical_devices[0], True)
    logging.info(f"Using GPU: {physical_devices[0]}")
    print("Using GPU:", physical_devices[0])
else:
    logging.info("GPU not found, using CPU")
    print("GPU not found, using CPU")

# อ่านข้อมูล
df_stock = pd.read_csv("NDQ_Stock_History_10Y.csv", parse_dates=["Date"]).sort_values(by=["Ticker", "Date"])
df_news = pd.read_csv("news_with_sentiment_gpu.csv")
df_news['Sentiment'] = df_news['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
df_news['Confidence'] = df_news['Confidence'] / 100
df = pd.merge(df_stock, df_news[['Date', 'Sentiment', 'Confidence']], on='Date', how='left')

# เติมค่าที่ขาดหายไป
df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)  # ในกรณีที่ยังมีค่า missing อยู่

# เพิ่มฟีเจอร์ใหม่
df['Change'] = df['Close'] - df['Open']
df['Change (%)'] = (df['Change'] / df['Open']) * 100

# ใช้ ta เพื่อคำนวณ RSI
df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
df['RSI'].fillna(method='ffill', inplace=True)
df['RSI'].fillna(0, inplace=True)

# เพิ่มฟีเจอร์เพิ่มเติม
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# เพิ่ม Bollinger Bands
bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
df['Bollinger_High'] = bollinger.bollinger_hband()
df['Bollinger_Low'] = bollinger.bollinger_lband()

# เติมค่า missing ที่เกิดจากการสร้างฟีเจอร์
df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment', 'Confidence',
                   'RSI', 'SMA_10', 'SMA_200', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low']

# แบ่งข้อมูลเป็น Train, Validation, Test
# ตัวอย่าง: 70% Train, 15% Validation, 15% Test
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

train_end = int(len(df)*train_ratio)
val_end = int(len(df)*(train_ratio+val_ratio))

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

# สร้าง target โดยเลื่อน 1 วันสำหรับแต่ละชุด
train_targets_price = train_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
train_df = train_df.iloc[:-1]

val_targets_price = val_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
val_df = val_df.iloc[:-1]

test_targets_price = test_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
test_df = test_df.iloc[:-1]

# สร้าง scaler จากข้อมูลเทรนเท่านั้น
scaler_features = MinMaxScaler()
train_features = scaler_features.fit_transform(train_df[feature_columns])
val_features = scaler_features.transform(val_df[feature_columns])
test_features = scaler_features.transform(test_df[feature_columns])

scaler_target = MinMaxScaler()
train_targets_scaled = scaler_target.fit_transform(train_targets_price)
val_targets_scaled = scaler_target.transform(val_targets_price)
test_targets_scaled = scaler_target.transform(test_targets_price)

joblib.dump(scaler_features, 'scaler_features.pkl')  # บันทึก scaler ฟีเจอร์
joblib.dump(scaler_target, 'scaler_target.pkl')     # บันทึก scaler เป้าหมาย

seq_length = 10
X_price_train, y_price_train = create_sequences(train_features, train_targets_scaled, seq_length)
X_price_val, y_price_val = create_sequences(val_features, val_targets_scaled, seq_length)
X_price_test, y_price_test = create_sequences(test_features, test_targets_scaled, seq_length)

# สร้างและฝึกโมเดล
price_model = build_GRU_model((X_price_train.shape[1], X_price_train.shape[2]))

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_price_model.keras', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

logging.info("เริ่มฝึกโมเดลสำหรับราคาหุ้นรวม")

history = price_model.fit(
    X_price_train, y_price_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_price_val, y_price_val),
    verbose=1,
    shuffle=False,          # เนื่องจากเป็น Time Series
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# บันทึกโมเดล
price_model.save('price_prediction_GRU_model.h5')
logging.info("บันทึกโมเดลราคาหุ้นรวมเรียบร้อยแล้ว")

# พล็อตผลการฝึก
plot_training_history(history)

# ประเมินโมเดลด้วยชุดทดสอบ (Test Set)
y_pred_scaled = price_model.predict(X_price_test)
y_pred = scaler_target.inverse_transform(y_pred_scaled)
y_true = scaler_target.inverse_transform(y_price_test)

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print("Evaluation on Test Set:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")
print(f"MAPE: {mape}")

# พล็อตการทำนายเทียบกับค่าจริง
plot_predictions(y_true, y_pred, "Test Set")

# พล็อต Residuals
plot_residuals(y_true, y_pred, "Test Set")
