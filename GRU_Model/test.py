import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import joblib
import ta

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

df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment', 'Confidence',
                   'RSI', 'SMA_10', 'SMA_200', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low']

# Label Encode Ticker
ticker_encoder = LabelEncoder()
df['Ticker_ID'] = ticker_encoder.fit_transform(df['Ticker'])
num_tickers = len(ticker_encoder.classes_)

# แบ่งข้อมูล Train/Val/Test ตามเวลา
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

sorted_dates = df['Date'].sort_values().unique()
train_cutoff = sorted_dates[int(len(sorted_dates) * train_ratio)]
val_cutoff = sorted_dates[int(len(sorted_dates) * (train_ratio + val_ratio))]

train_df = df[df['Date'] <= train_cutoff].copy()
val_df = df[(df['Date'] > train_cutoff) & (df['Date'] <= val_cutoff)].copy()
test_df = df[df['Date'] > val_cutoff].copy()

# สร้าง target โดย shift(-1)
test_targets_price = test_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
test_df = test_df.iloc[:-1]

test_features = test_df[feature_columns].values
test_ticker_id = test_df['Ticker_ID'].values

# โหลด Scaler ที่บันทึกไว้
scaler_features = joblib.load('scaler_features_full.pkl')
scaler_target = joblib.load('scaler_target_full.pkl')

test_features_scaled = scaler_features.transform(test_features)
test_targets_scaled = scaler_target.transform(test_targets_price)

# สร้าง sequences สำหรับ AAPB
def create_sequences_for_ticker(features, ticker_ids, targets, seq_length=10):
    X_features, X_tickers, Y = [], [], []
    for i in range(len(features) - seq_length):
        X_features.append(features[i:i+seq_length])
        X_tickers.append(ticker_ids[i:i+seq_length])  # sequence ของ ticker_id
        Y.append(targets[i+seq_length])
    return np.array(X_features), np.array(X_tickers), np.array(Y)

seq_length = 10

X_test_list, X_test_ticker_list, y_test_list = [], [], []
for t_id in range(num_tickers):
    df_test_ticker = test_df[test_df['Ticker_ID'] == t_id]
    if len(df_test_ticker) > seq_length:
        indices = df_test_ticker.index
        mask_test = np.isin(test_df.index, indices)
        f_s = test_features_scaled[mask_test]
        t_s = test_ticker_id[mask_test]
        target_s = test_targets_scaled[mask_test]
        X_s, X_si, y_s = create_sequences_for_ticker(f_s, t_s, target_s, seq_length)
        X_test_list.append(X_s)
        X_test_ticker_list.append(X_si)
        y_test_list.append(y_s)

if len(X_test_list) > 0:
    X_price_test = np.concatenate(X_test_list, axis=0)
    X_ticker_test = np.concatenate(X_test_ticker_list, axis=0)
    y_price_test = np.concatenate(y_test_list, axis=0)
else:
    X_price_test, X_ticker_test, y_price_test = np.array([]), np.array([]), np.array([])

# โหลดโมเดลที่บันทึกไว้
model = load_model('price_prediction_GRU_model_embedding_full.h5', 
                   custom_objects={'mse': MeanSquaredError()})


# ทดสอบโมเดล
y_pred_scaled = model.predict([X_price_test, X_ticker_test])
y_pred = scaler_target.inverse_transform(y_pred_scaled)
y_true = scaler_target.inverse_transform(y_price_test)

# ประเมินผลการทำนาย
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print("Evaluation on AAPB:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")
print(f"MAPE: {mape}")

# การแสดงผลกราฟ True vs Predicted
def plot_predictions(y_true, y_pred, ticker="AAPB"):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='red', alpha=0.7)
    plt.title(f'True vs Predicted Prices for {ticker}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# การแสดงผล Residuals
def plot_residuals(y_true, y_pred, ticker="AAPB"):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(residuals)), residuals, alpha=0.5)
    plt.hlines(y=0, xmin=0, xmax=len(residuals), colors='red')
    plt.title(f'Residuals for {ticker}')
    plt.xlabel('Sample')
    plt.ylabel('Residual')
    plt.show()
    


plot_predictions(y_true[:200], y_pred[:200], "AAPB")
plot_residuals(y_true, y_pred, "AAPB")
