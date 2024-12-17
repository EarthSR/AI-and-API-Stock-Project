import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, concatenate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
import ta
import logging
from tensorflow.keras.losses import MeanSquaredError


# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, filename='training.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def create_sequences_for_ticker(features, ticker_ids, targets, seq_length=10):
    X_features, X_tickers, Y = [], [], []
    for i in range(len(features) - seq_length):
        X_features.append(features[i:i+seq_length])
        X_tickers.append(ticker_ids[i:i+seq_length])  # sequence ของ ticker_id
        Y.append(targets[i+seq_length])
    return np.array(X_features), np.array(X_tickers), np.array(Y)

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

# ตรวจสอบ GPU
# ตรวจสอบ GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    # Set visible devices to the first GPU (or any other specific one)
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    # Enable memory growth for the first GPU
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logging.info(f"Memory growth enabled for GPU: {physical_devices[0]}")
        print("Memory growth enabled for GPU:", physical_devices[0])
    except Exception as e:
        logging.error(f"Failed to set memory growth: {e}")
        print(f"Error setting memory growth: {e}")
else:
    logging.info("GPU not found, using CPU")
    print("GPU not found, using CPU")

# โหลดข้อมูล
df_stock = pd.read_csv("cleaned_data.csv", parse_dates=["Date"]).sort_values(by=["Ticker", "Date"])
df_news = pd.read_csv("news_with_sentiment_gpu.csv")
df_news['Date'] = pd.to_datetime(df_news['Date'], errors='coerce')
df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')
print(df_news['Date'].dtype)
print(df_stock['Date'].dtype)
df_news['Sentiment'] = df_news['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
df_news['Confidence'] = df_news['Confidence']
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

df['SMA_5'] = df['Close'].rolling(window=5).mean()  # SMA 50 วัน
df['SMA_10'] = df['Close'].rolling(window=10).mean()  # SMA 200 วัน
# คำนวณ MACD ด้วย EMA 12 และ EMA 26
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
# คำนวณ MACD = EMA(12) - EMA(26)
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].rolling(window=9).mean()  
bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
df['Bollinger_High'] = bollinger.bollinger_hband()
df['Bollinger_Low'] = bollinger.bollinger_lband()

df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment',
                    'RSI', 'SMA_10', 'SMA_5', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low']
# feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment', 'Confidence']

# Label Encode Ticker
ticker_encoder = LabelEncoder()
df['Ticker_ID'] = ticker_encoder.fit_transform(df['Ticker'])
num_tickers = len(ticker_encoder.classes_)

# แบ่งข้อมูล Train/Val/Test ตามเวลา
# สมมติเราแบ่งตาม quantile ของวันที่ หรือกำหนดโดยตรง
sorted_dates = df['Date'].unique()
train_cutoff = sorted_dates[int(len(sorted_dates) * 6 / 7)]  # ขอบเขตที่ 6 ปี

# ข้อมูล train, test
train_df = df[df['Date'] <= train_cutoff].copy()
test_df = df[df['Date'] > train_cutoff].copy()

# สร้าง target โดย shift(-1)
train_targets_price = train_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
train_df = train_df.iloc[:-1]

test_targets_price = test_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
test_df = test_df.iloc[:-1]

train_features = train_df[feature_columns].values
test_features = test_df[feature_columns].values

train_ticker_id = train_df['Ticker_ID'].values
test_ticker_id = test_df['Ticker_ID'].values

# สเกลข้อมูลจากเทรนเท่านั้น
scaler_features = MinMaxScaler()
train_features_scaled = scaler_features.fit_transform(train_features)
test_features_scaled = scaler_features.transform(test_features)

scaler_target = MinMaxScaler()
train_targets_scaled = scaler_target.fit_transform(train_targets_price)
test_targets_scaled = scaler_target.transform(test_targets_price)

joblib.dump(scaler_features, 'scaler_features.pkl')  # บันทึก scaler ฟีเจอร์
joblib.dump(scaler_target, 'scaler_target.pkl')     # บันทึก scaler เป้าหมาย

seq_length = 10

# สร้าง sequences แยกตาม Ticker
X_train_list, X_train_ticker_list, y_train_list = [], [], []
X_val_list, X_val_ticker_list, y_val_list = [], [], []
X_test_list, X_test_ticker_list, y_test_list = [], [], []

for t_id in range(num_tickers):
    # Train
    df_train_ticker = train_df[train_df['Ticker_ID'] == t_id]
    if len(df_train_ticker) > seq_length:
        indices = df_train_ticker.index
        mask_train = np.isin(train_df.index, indices)
        f_t = train_features_scaled[mask_train]
        t_t = train_ticker_id[mask_train]
        target_t = train_targets_scaled[mask_train]
        X_t, X_ti, y_t = create_sequences_for_ticker(f_t, t_t, target_t, seq_length)
        X_train_list.append(X_t)
        X_train_ticker_list.append(X_ti)
        y_train_list.append(y_t)
        
    # Test
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

if len(X_train_list) > 0:
    X_price_train = np.concatenate(X_train_list, axis=0)
    X_ticker_train = np.concatenate(X_train_ticker_list, axis=0)
    y_price_train = np.concatenate(y_train_list, axis=0)
else:
    X_price_train, X_ticker_train, y_price_train = np.array([]), np.array([]), np.array([])

if len(X_val_list) > 0:
    X_price_val = np.concatenate(X_val_list, axis=0)
    X_ticker_val = np.concatenate(X_val_ticker_list, axis=0)
    y_price_val = np.concatenate(y_val_list, axis=0)
else:
    X_price_val, X_ticker_val, y_price_val = np.array([]), np.array([]), np.array([])

if len(X_test_list) > 0:
    X_price_test = np.concatenate(X_test_list, axis=0)
    X_ticker_test = np.concatenate(X_test_ticker_list, axis=0)
    y_price_test = np.concatenate(y_test_list, axis=0)
else:
    X_price_test, X_ticker_test, y_price_test = np.array([]), np.array([]), np.array([])

num_feature = train_features_scaled.shape[1]  # จำนวน features ทางเทคนิค

# สร้างโมเดล LSTM + Embedding
features_input = Input(shape=(seq_length, num_feature), name='features_input')
ticker_input = Input(shape=(seq_length,), name='ticker_input')

print(f"Shape of X_price_train: {X_price_train.shape}")
print(f"Shape of X_ticker_train: {X_ticker_train.shape}")


embedding_dim = 32
ticker_embedding = Embedding(input_dim=num_tickers, output_dim=embedding_dim, name='ticker_embedding')(ticker_input)

merged = concatenate([features_input, ticker_embedding], axis=-1)

x = LSTM(64, return_sequences=True)(merged)
x = Dropout(0.2)(x)
x = LSTM(32)(x)
x = Dropout(0.2)(x)
output = Dense(1)(x)

model = Model(inputs=[features_input, ticker_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_price_model.keras', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

logging.info("เริ่มฝึกโมเดลสำหรับราคาหุ้นรวม (ใช้ Embedding สำหรับ Ticker)")

history = model.fit(
    [X_price_train, X_ticker_train], y_price_train,
    epochs=50,
    batch_size=32,
    verbose=1,
    shuffle=False,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)


model.save('price_prediction_LSTM_model_embedding.h5')
logging.info("บันทึกโมเดลราคาหุ้นรวมเรียบร้อยแล้ว")

# Load pre-existing model and scalers
model = load_model('price_prediction_LSTM_model_embedding.h5', custom_objects={'mse': MeanSquaredError()})
scaler_features = joblib.load('scaler_features.pkl')
scaler_target = joblib.load('scaler_target.pkl')

# Initializations
predictions = []
true_prices = []
dates = []

# Define sequence length
seq_length = 10

# Train the model incrementally (daily updates)
for i in range(seq_length, len(test_df)):
    # Get data for this day
    current_data = test_df.iloc[i-seq_length:i]  # Data for the last `seq_length` days
    
    # Prepare features and target
    features = current_data[feature_columns].values
    ticker_id = current_data['Ticker_ID'].values
    target = current_data['Close'].shift(-1).dropna().values.reshape(-1, 1)
    
    # Scale the features and target
    features_scaled = scaler_features.transform(features)
    target_scaled = scaler_target.transform(target)
    
    # Create sequences for this new data point
    X_features, X_tickers, y_target = create_sequences_for_ticker(features_scaled, ticker_id, target_scaled, seq_length)
    
    # Predict the next day's price
    y_pred_scaled = model.predict([X_features, X_tickers])
    y_pred = scaler_target.inverse_transform(y_pred_scaled)
    y_true = scaler_target.inverse_transform(y_target)
    
    predictions.append(y_pred[-1, 0])  # Predict the last point in the sequence
    true_prices.append(y_true[-1, 0])  # True price for that day
    dates.append(test_df['Date'].iloc[i])

    # Retrain the model with the new data point
    # Since we are only predicting one data point per day, we will add it back to the training set and retrain the model
    X_train = np.concatenate([X_train, X_features], axis=0)
    X_ticker_train = np.concatenate([X_ticker_train, X_tickers], axis=0)
    y_train = np.concatenate([y_train, y_target], axis=0)

    # Retrain the model incrementally
    model.fit(
        [X_train, X_ticker_train], y_train,
        epochs=1,
        batch_size=32,
        verbose=0,
        shuffle=False
    )

# Plot predictions vs true values
plt.figure(figsize=(12, 6))
plt.plot(dates, true_prices, label='True Prices', color='blue')
plt.plot(dates, predictions, label='Predicted Prices', color='red', alpha=0.7)
plt.title('True vs Predicted Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Evaluate the model performance at the end
mae = mean_absolute_error(true_prices, predictions)
mse = mean_squared_error(true_prices, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(true_prices, predictions)
mape = mean_absolute_percentage_error(true_prices, predictions)

print("Evaluation on Test Set:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")
print(f"MAPE: {mape}")