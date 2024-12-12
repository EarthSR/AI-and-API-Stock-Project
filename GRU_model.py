import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping  # เพิ่ม EarlyStopping เพื่อหยุดการฝึกหากไม่มีการปรับปรุง
import matplotlib.pyplot as plt

# ตรวจสอบ GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.set_memory_growth(physical_devices[0], True)  # ใช้ set_memory_growth
    print("Using GPU:", physical_devices[0])
else:
    print("GPU not found, using CPU")

# โหลดข้อมูลหุ้น
df_stock = pd.read_csv("NDQ_Stock_History_10Y.csv", parse_dates=["Date"]).sort_values(by="Date")

# โหลดข้อมูลข่าว
df_news = pd.read_csv("news_with_sentiment_gpu.csv")
df_news['Sentiment'] = df_news['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
df_news['Confidence'] = df_news['Confidence'] / 100

# รวมข้อมูล
df = pd.merge(df_stock, df_news[['Date', 'Sentiment', 'Confidence']], on='Date', how='left')

# เพิ่มฟีเจอร์ใหม่
df['Change'] = df['Close'] - df['Open']
df['Change (%)'] = (df['Change'] / df['Open']) * 100
df['RSI'] = df['Close'].diff().apply(lambda x: max(x, 0)).rolling(window=14).mean() / \
            df['Close'].diff().apply(lambda x: -min(x, 0)).rolling(window=14).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Normalize features
scaler = MinMaxScaler()

# ใช้ scaler.fit_transform กับข้อมูลทั้งหมด
features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment', 'Confidence',
               'RSI', 'SMA_50', 'SMA_200', 'MACD', 'MACD_Signal']].fillna(0).values
features_scaled = scaler.fit_transform(features)  # ใช้ fit_transform กับข้อมูลทั้งหมดเพื่อลดเวลาในการคำนวณ

# สร้างลำดับข้อมูล
def create_sequences(features, targets, seq_length=10):
    X, y = [], []
    for i in range(len(features) - seq_length):
        if i + seq_length < len(targets):  
            X.append(features[i:i + seq_length])
            y.append(targets[i + seq_length])
    return np.array(X), np.array(y)

# ข้อมูลสำหรับโมเดลราคาหุ้น
targets_price = df['Close'].shift(-1).dropna().values
X_price, y_price = create_sequences(features_scaled, targets_price)

# Train-test split สำหรับโมเดลราคาหุ้น
X_price_train, X_price_test, y_price_train, y_price_test = train_test_split(X_price, y_price, test_size=0.2, random_state=42)

# โมเดลราคาหุ้นรวม
price_model = Sequential([
    GRU(64, activation='relu', return_sequences=True, input_shape=(X_price.shape[1], X_price.shape[2])),
    Dropout(0.2),
    GRU(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
price_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ใช้ EarlyStopping เพื่อหยุดการฝึกหากไม่มีการปรับปรุง
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ฝึกโมเดล
price_model.fit(X_price_train, y_price_train, epochs=50, batch_size=32, validation_data=(X_price_test, y_price_test), verbose=1, callbacks=[early_stopping])

# บันทึกโมเดล
price_model.save('price_prediction_gru_model.h5')


# โมเดลแยกสำหรับหุ้น Top 5
top_5_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB']
for ticker in top_5_tickers:
    ticker_data = df[df['Ticker'] == ticker]

    # ตรวจสอบว่าไม่มีข้อมูลหรือไม่
    if ticker_data.empty:
        print(f"Warning: No data found for ticker {ticker}. Skipping this ticker.")
        continue

    # เตรียมฟีเจอร์สำหรับการ normalization
    ticker_features = ticker_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)',
                                   'Sentiment', 'Confidence', 'RSI', 'SMA_50', 'SMA_200',
                                   'MACD', 'MACD_Signal']].fillna(0).values
    
    # ตรวจสอบว่ามีข้อมูลหลังจากการกรองหรือไม่
    if len(ticker_features) == 0:
        print(f"Warning: No features available for ticker {ticker}. Skipping this ticker.")
        continue
    
    # ใช้ scaler.transform เพื่อแปลงข้อมูลของแต่ละหุ้นโดยใช้ scaler เดิม
    features_scaled = scaler.transform(ticker_features)  # ใช้ transform แทน fit_transform เพื่อใช้ scaler เดิม

    # สร้างข้อมูลเป้าหมาย (targets)
    ticker_targets_price = ticker_data['Close'].shift(-1).dropna().values
    
    # ตรวจสอบว่ามีเป้าหมายหรือไม่
    if len(ticker_targets_price) == 0:
        print(f"Warning: No target price available for ticker {ticker}. Skipping this ticker.")
        continue
    
    # สร้างลำดับข้อมูล
    X_ticker_price, y_ticker_price = create_sequences(features_scaled, ticker_targets_price)
    
    # ตรวจสอบขนาดข้อมูลที่เตรียมไว้
    if X_ticker_price.shape[0] == 0:
        print(f"Warning: No sequences available for ticker {ticker}. Skipping this ticker.")
        continue
    
    X_price_train, X_price_test, y_price_train, y_price_test = train_test_split(X_ticker_price, y_ticker_price, test_size=0.2, random_state=42)
    
    # ฝึกฝนโมเดลราคาหุ้น
    price_model = Sequential([
        GRU(64, activation='relu', return_sequences=True, input_shape=(X_ticker_price.shape[1], X_ticker_price.shape[2])),
        Dropout(0.2),
        GRU(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    price_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # ใช้ EarlyStopping เพื่อหยุดการฝึกหากไม่มีการปรับปรุง
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = price_model.fit(X_price_train, y_price_train, epochs=50, batch_size=32, validation_data=(X_price_test, y_price_test), verbose=1, callbacks=[early_stopping])
    
    # ตรวจสอบ training history
    print(f"History for {ticker} training:")
    print(history.history)
    
    # บันทึกโมเดล
    price_model.save(f'price_model_{ticker}.h5')
    
    # ทำนายราคาหุ้น
    ticker_price_predictions = price_model.predict(X_price_test)
    
    # คำนวณความแตกต่างระหว่างราคาทำนายและราคาปิดเมื่อวาน
    price_diff = ticker_price_predictions.flatten() - y_price_test

    # หากราคาทำนายสูงกว่าราคาปิดเมื่อวาน ถือว่าเป็น "ขึ้น" (1) มิฉะนั้น "ลง" (0)
    trend_prediction = (price_diff > 0).astype(int)

    # ประเมินผลราคาหุ้น
    price_mae = mean_absolute_error(y_price_test, ticker_price_predictions)
    price_mse = mean_squared_error(y_price_test, ticker_price_predictions)
    price_rmse = np.sqrt(price_mse)
    price_r2 = r2_score(y_price_test, ticker_price_predictions)
    
    # คำนวณ Accuracy แบบช่วง ±5% ของค่าจริง
    price_tolerance = 0.05  # กำหนดช่วงที่ยอมรับได้ ±5%
    price_within_tolerance = np.abs((ticker_price_predictions.flatten() - y_price_test) / y_price_test) <= price_tolerance
    price_accuracy = np.mean(price_within_tolerance) * 100  # เปลี่ยนเป็นเปอร์เซ็นต์
    
    # ประเมินแนวโน้ม (Trend) Accuracy
    trend_accuracy = accuracy_score(y_price_test, trend_prediction)

    print(f"Stock: {ticker}")
    print(f"Price Accuracy: {price_accuracy:.2f}%")
    print(f"Price MAE: {price_mae:.2f}, MSE: {price_mse:.2f}, RMSE: {price_rmse:.2f}, R2: {price_r2:.2f}")
    print(f"Trend Accuracy: {trend_accuracy:.2f}")
    print("="*50)

import matplotlib.pyplot as plt

# หลังจากการฝึกโมเดลแล้ว ให้สร้างกราฟการฝึกฝน
def plot_training_history(history):
    # กราฟ Loss
    plt.figure(figsize=(12, 6))
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

# เรียกใช้ฟังก์ชันนี้หลังจากการฝึกฝนโมเดล
plot_training_history(history)

# เพิ่มกราฟทำนายกับค่าจริง
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='red', alpha=0.7)
    plt.title('True vs Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# ทำนายราคาหุ้นจากข้อมูลทดสอบ
ticker_price_predictions = price_model.predict(X_price_test)

# แสดงกราฟการทำนายกับค่าจริง
plot_predictions(y_price_test, ticker_price_predictions.flatten())
