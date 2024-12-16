import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import ta
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, filename='testing.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model, ensuring custom metric MeanSquaredError is passed
model = load_model('./best_price_model_full.keras', custom_objects={'MeanSquaredError': MeanSquaredError()})

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
def create_sequences(data, sentiment, time_steps=10):
    sequences = []
    sentiment_sequences = []
    labels = []

    # Ensure data and sentiment have the same length
    min_len = min(len(data), len(sentiment))  # Ensure the lengths are the same
    data = data[:min_len]  # Trim data if necessary
    sentiment = sentiment[:min_len]  # Trim sentiment if necessary

    for i in range(len(data) - time_steps):
        sequences.append(data[i:i + time_steps])  # Sequence of features
        sentiment_sequences.append(sentiment[i:i + time_steps])  # Sequence of sentiment data
        labels.append(data[i + time_steps, 0])  # The label is the next value after time_steps (ensure it is scalar)
        
    return np.array(sequences), np.array(sentiment_sequences), np.array(labels)



# โหลดข้อมูล
df_stock = pd.read_csv("cleaned_data.csv", parse_dates=["Date"]).sort_values(by=["Ticker", "Date"])
df_news = pd.read_csv("news_with_sentiment_gpu.csv")
df_news['Sentiment'] = df_news['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
df_news['Confidence'] = df_news['Confidence'] / 100
df = pd.merge(df_stock, df_news[['Date', 'Sentiment', 'Confidence']], on='Date', how='left')

if df.shape[0] == 0:
    print("No data after merge. Check your Date columns in both dataframes.")
    exit()

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

# โหลด scaler ที่บันทึกไว้
scaler_features = joblib.load('scaler_features_full.pkl')
scaler_target = joblib.load('scaler_target_full.pkl')

# Select features for scaling (15 features as per your case)
features_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment', 'Confidence',
                     'RSI', 'SMA_10', 'SMA_200', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low']

print(f"Shape of df[features_to_scale]: {df[features_to_scale].shape}")
print(f"Columns in df: {df.columns}")

# ตรวจสอบว่า dataframe ไม่เป็น 0 แถว
if df[features_to_scale].shape[0] > 0:
    # Normalize features
    scaled_features = scaler_features.transform(df[features_to_scale])

    # Normalize target
    scaled_target = scaler_target.transform(df[['Close']].values.reshape(-1, 1))
else:
    print("No valid data to scale.")
    # ทำการเตรียมข้อมูลเพิ่มเติมหากไม่มีข้อมูลที่ถูกต้อง
    # หรืออาจทำการตรวจสอบข้อมูลที่ขาดหายไป
    exit()
# Normalize features
scaled_features = scaler_features.transform(df[features_to_scale])

# Normalize target
scaled_target = scaler_target.transform(df[['Close']].values.reshape(-1, 1))

# Label Encode Ticker
ticker_encoder = LabelEncoder()
df['Ticker_ID'] = ticker_encoder.fit_transform(df['Ticker'])
num_tickers = len(ticker_encoder.classes_)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Date'] = df['Date'].dt.tz_localize(None)

# Split data into train/test sets
test_date = pd.Timestamp('2024-01-01')


test_data = df[df['Date'] >= test_date]

X_test = scaled_features[test_data.index]
y_test = scaled_target[test_data.index]

# สร้างข้อมูลลำดับ (sequence) สำหรับการเทรน
# สร้าง sequences สำหรับ features และ sentiment
sequences, sentiment_sequences, labels = create_sequences(scaled_features, df['Sentiment'].values)
print(f"Sequences shape: {sequences.shape}")
print(f"Sentiment Sequences shape: {sentiment_sequences.shape}")
# Predict using the trained model
model_input = [sequences, sentiment_sequences]
y_pred = model.predict(model_input)

# ใช้ inverse_transform หลังจาก reshaped ข้อมูลแล้ว
# Rescale the predictions and true values
y_true_rescaled = scaler_target.inverse_transform(y_test.reshape(-1, 1))  # reshape ให้เป็น 2D
y_pred_rescaled = scaler_target.inverse_transform(y_pred.reshape(-1, 1))  # reshape ให้เป็น 2D

# Calculate the evaluation metrics
mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
mse = mean_squared_error(y_true_rescaled, y_pred_rescaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_true_rescaled, y_pred_rescaled)
mape = mean_absolute_percentage_error(y_true_rescaled, y_pred_rescaled)

# Log the results
logging.info(f"MAE: {mae}")
logging.info(f"MSE: {mse}")
logging.info(f"RMSE: {rmse}")
logging.info(f"R2: {r2}")
logging.info(f"MAPE: {mape}")

# Plot the predictions
plot_predictions(y_true_rescaled, y_pred_rescaled, ticker="AAPL")

# Plot the residuals
plot_residuals(y_true_rescaled, y_pred_rescaled, ticker="AAPL")
