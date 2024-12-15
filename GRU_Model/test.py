import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import ta
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, filename='testing.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ฟังก์ชันที่ใช้สำหรับการแสดงผลลัพธ์
def plot_predictions(y_true, y_pred, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='red', alpha=0.7)
    plt.title(f'True vs Predicted Prices for {ticker}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_residuals(y_true, y_pred, ticker):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(residuals)), residuals, alpha=0.5)
    plt.hlines(y=0, xmin=0, xmax=len(residuals), colors='red')
    plt.title(f'Residuals for {ticker}')
    plt.xlabel('Sample')
    plt.ylabel('Residual')
    plt.show()

# ฟังก์ชันที่ใช้แปลงข้อมูลให้เป็นลำดับ (sequence)
def create_sequences(data, time_steps=10):
    sequences = []
    labels = []
    
    for i in range(len(data) - time_steps):
        sequences.append(data[i:i + time_steps])  # Add time_steps number of samples
        labels.append(data[i + time_steps])  # The label is the next value after time_steps
        
    return np.array(sequences), np.array(labels)

# โหลดข้อมูล
df_stock = pd.read_csv("cleaned_data.csv", parse_dates=["Date"]).sort_values(by=["Ticker", "Date"])
df_news = pd.read_csv("news_with_sentiment_gpu.csv")
df_news['Sentiment'] = df_news['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
df_news['Confidence'] = df_news['Confidence'] / 100
df = pd.merge(df_stock, df_news[['Date', 'Sentiment', 'Confidence']], on='Date', how='left')

# เติมค่าที่ขาดหายไป
df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

# เพิ่มฟีเจอร์
df['Change'] = df['Close'] - df['Open']
df['Change (%)'] = (df['Change'] / df['Open']) * 100

df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
df['RSI'].fillna(method='ffill', inplace=True)
df['RSI'].fillna(0, inplace=True)

df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
df['Bollinger_High'] = bollinger.bollinger_hband()
df['Bollinger_Low'] = bollinger.bollinger_lband()

# เติมค่า NaN
df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment', 'Confidence',
                   'RSI', 'SMA_10', 'SMA_200', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low']

# Label Encode Ticker
ticker_encoder = LabelEncoder()
df['Ticker_ID'] = ticker_encoder.fit_transform(df['Ticker'])
num_tickers = len(ticker_encoder.classes_)

# แปลง Date ให้เป็น tz-naive ก่อน
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# ลบ TimeZone ถ้าเป็น tz-aware
df['Date'] = df['Date'].dt.tz_localize(None)

# สร้าง Timestamp แบบ tz-naive สำหรับการเปรียบเทียบ
test_date = pd.Timestamp('2023-01-01')

# การเปรียบเทียบกับ `Date` ที่เป็น tz-naive
test_df = df[df['Date'] > test_date].copy()

# โหลด scaler ที่บันทึกไว้
scaler_features = joblib.load('scaler_features_full.pkl')
scaler_target = joblib.load('scaler_target_full.pkl')

# แปลงข้อมูลการทดสอบ
X_test = test_df[feature_columns].values
X_scaled = scaler_features.transform(X_test)

y_true = test_df['Close'].values
y_scaled = scaler_target.transform(y_true.reshape(-1, 1))  # แปลงเป้าหมาย

# สร้าง sequences สำหรับ X_scaled
X_sequences, y_sequences = create_sequences(X_scaled, time_steps=10)

# แปลงข้อมูล target (y) ด้วย scaler
y_scaled = scaler_target.transform(y_sequences.reshape(-1, 1))

# ตรวจสอบขนาดของ X_sequences
print(X_sequences.shape)  # ควรเป็น (จำนวนลำดับ, time_steps, features)

# ตรวจสอบขนาดของ y_sequences
print(y_sequences.shape)  # ควรเป็น (จำนวนลำดับ, 1)

# โหลดโมเดล
model = load_model('best_price_model_full.keras')

# ทำการทำนาย
y_pred_scaled = model.predict(X_sequences)

# แปลงผลลัพธ์กลับเป็นค่าเดิม
y_pred = scaler_target.inverse_transform(y_pred_scaled)

# ประเมินผล
mae = mean_absolute_error(y_sequences, y_pred)
mse = mean_squared_error(y_sequences, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_sequences, y_pred)
mape = mean_absolute_percentage_error(y_sequences, y_pred)

# แสดงผลลัพธ์
logging.info(f'MAE: {mae}')
logging.info(f'MSE: {mse}')
logging.info(f'RMSE: {rmse}')
logging.info(f'R2: {r2}')
logging.info(f'MAPE: {mape}')
