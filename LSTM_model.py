import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf

# ตรวจสอบ GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
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
df['Trend'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Normalize features
scaler = MinMaxScaler()
features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment', 'Confidence',
               'RSI', 'SMA_50', 'SMA_200', 'MACD', 'MACD_Signal']].fillna(0).values
features_scaled = scaler.fit_transform(features)

# สร้างลำดับข้อมูล
def create_sequences(features, targets, seq_length=10):
    X, y = [], []
    for i in range(len(features) - seq_length):
        if i + seq_length < len(targets):  
            X.append(features[i:i + seq_length])
            y.append(targets[i + seq_length])
    return np.array(X), np.array(y)


# ข้อมูลสำหรับโมเดลรวม
targets_price = df['Close'].shift(-1).dropna().values
targets_trend = df['Trend'].dropna().values
X_price, y_price = create_sequences(features_scaled, targets_price)
X_trend, y_trend = create_sequences(features_scaled, targets_trend)

# Train-test split สำหรับโมเดลรวม
X_price_train, X_price_test, y_price_train, y_price_test = train_test_split(X_price, y_price, test_size=0.2, random_state=42)
X_trend_train, X_trend_test, y_trend_train, y_trend_test = train_test_split(X_trend, y_trend, test_size=0.2, random_state=42)

# โมเดลราคาหุ้นรวม
price_model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(X_price.shape[1], X_price.shape[2])),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
price_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
price_model.fit(X_price_train, y_price_train, epochs=50, batch_size=32, validation_data=(X_price_test, y_price_test), verbose=1)
price_model.save('price_prediction_lstm_model.h5')

# โมเดลแนวโน้มรวม
trend_model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(X_trend.shape[1], X_trend.shape[2])),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
trend_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
trend_model.fit(X_trend_train, y_trend_train, epochs=50, batch_size=32, validation_data=(X_trend_test, y_trend_test), verbose=1)
trend_model.save('trend_prediction_lstm_model.h5')

# วัดความแม่นยำของโมเดลรวม
print("\nEvaluating Combined Models:")
price_predictions = price_model.predict(X_price_test)
trend_predictions = (trend_model.predict(X_trend_test) > 0.5).astype(int)

# วัดผลสำหรับราคาหุ้น
price_mae = mean_absolute_error(y_price_test, price_predictions)
price_mse = mean_squared_error(y_price_test, price_predictions)
price_rmse = np.sqrt(price_mse)
price_r2 = r2_score(y_price_test, price_predictions)

# คำนวณ Accuracy แบบช่วง ±5% ของค่าจริง
price_tolerance = 0.05  # กำหนดช่วงที่ยอมรับได้ ±5%
price_within_tolerance = np.abs((price_predictions.flatten() - y_price_test) / y_price_test) <= price_tolerance
price_accuracy = np.mean(price_within_tolerance) * 100  # เปลี่ยนเป็นเปอร์เซ็นต์

print(f"Price Model - MAE: {price_mae:.4f}, MSE: {price_mse:.4f}, RMSE: {price_rmse:.4f}, R²: {price_r2:.4f}, Accuracy: {price_accuracy:.2f}%")

# สร้างโมเดลแยกสำหรับหุ้น Top 5
top_5_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB']
for ticker in top_5_tickers:
    ticker_data = df[df['Ticker'] == ticker]
    ticker_features = scaler.fit_transform(ticker_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)',
                                                         'Sentiment', 'Confidence', 'RSI', 'SMA_50', 'SMA_200',
                                                         'MACD', 'MACD_Signal']].fillna(0).values)
    ticker_targets_price = ticker_data['Close'].shift(-1).dropna().values
    ticker_targets_trend = ticker_data['Trend'].dropna().values
    
    X_ticker_price, y_ticker_price = create_sequences(ticker_features, ticker_targets_price)
    X_ticker_trend, y_ticker_trend = create_sequences(ticker_features, ticker_targets_trend)
    
    X_price_train, X_price_test, y_price_train, y_price_test = train_test_split(X_ticker_price, y_ticker_price, test_size=0.2, random_state=42)
    X_trend_train, X_trend_test, y_trend_train, y_trend_test = train_test_split(X_ticker_trend, y_ticker_trend, test_size=0.2, random_state=42)
    
    # ราคาหุ้น
    price_model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(X_ticker_price.shape[1], X_ticker_price.shape[2])),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    price_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    price_model.fit(X_price_train, y_price_train, epochs=50, batch_size=32, validation_data=(X_price_test, y_price_test), verbose=1)
    price_model.save(f'price_model_{ticker}.h5')
    
    # แนวโน้ม
    trend_model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(X_ticker_trend.shape[1], X_ticker_trend.shape[2])),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    trend_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    trend_model.fit(X_trend_train, y_trend_train, epochs=50, batch_size=32, validation_data=(X_trend_test, y_trend_test), verbose=1)
    trend_model.save(f'trend_model_{ticker}.h5')
    
    # ประเมินผลราคาหุ้น
    ticker_price_predictions = price_model.predict(X_price_test)
    price_mae = mean_absolute_error(y_price_test, ticker_price_predictions)
    price_mse = mean_squared_error(y_price_test, ticker_price_predictions)
    price_rmse = np.sqrt(price_mse)
    price_r2 = r2_score(y_price_test, ticker_price_predictions)
    
    # คำนวณ Accuracy แบบช่วง ±5% ของค่าจริง
    price_within_tolerance = np.abs((ticker_price_predictions.flatten() - y_price_test) / y_price_test) <= price_tolerance
    price_accuracy = np.mean(price_within_tolerance) * 100  # เปลี่ยนเป็นเปอร์เซ็นต์
    
    print(f"  Price Model - MAE: {price_mae:.4f}, MSE: {price_mse:.4f}, RMSE: {price_rmse:.4f}, R²: {price_r2:.4f}, Accuracy: {price_accuracy:.2f}%")
