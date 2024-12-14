import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, concatenate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
import ta
from tensorflow.keras.regularizers import l2
import logging

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
# feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment', 'Confidence']

# Label Encode Ticker
ticker_encoder = LabelEncoder()
df['Ticker_ID'] = ticker_encoder.fit_transform(df['Ticker'])
num_tickers = len(ticker_encoder.classes_)

# แบ่งข้อมูล Train/Val/Test ตามเวลา
# สมมติเราแบ่งตาม quantile ของวันที่ หรือกำหนดโดยตรง
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

sorted_dates = df['Date'].sort_values().unique()
train_cutoff = sorted_dates[int(len(sorted_dates)*train_ratio)]
val_cutoff = sorted_dates[int(len(sorted_dates)*(train_ratio+val_ratio))]

train_df = df[df['Date'] <= train_cutoff].copy()
val_df = df[(df['Date'] > train_cutoff) & (df['Date'] <= val_cutoff)].copy()
test_df = df[df['Date'] > val_cutoff].copy()

# สร้าง target โดย shift(-1)
train_targets_price = train_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
train_df = train_df.iloc[:-1]

val_targets_price = val_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
val_df = val_df.iloc[:-1]

test_targets_price = test_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
test_df = test_df.iloc[:-1]

train_features = train_df[feature_columns].values
val_features = val_df[feature_columns].values
test_features = test_df[feature_columns].values

train_ticker_id = train_df['Ticker_ID'].values
val_ticker_id = val_df['Ticker_ID'].values
test_ticker_id = test_df['Ticker_ID'].values

# สเกลข้อมูลจากเทรนเท่านั้น
scaler_features = MinMaxScaler()
train_features_scaled = scaler_features.fit_transform(train_features)
val_features_scaled = scaler_features.transform(val_features)
test_features_scaled = scaler_features.transform(test_features)

scaler_target = MinMaxScaler()
train_targets_scaled = scaler_target.fit_transform(train_targets_price)
val_targets_scaled = scaler_target.transform(val_targets_price)
test_targets_scaled = scaler_target.transform(test_targets_price)

joblib.dump(scaler_features, 'scaler_features_full.pkl')  # บันทึก scaler ฟีเจอร์
joblib.dump(scaler_target, 'scaler_target_full.pkl')     # บันทึก scaler เป้าหมาย

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

    # Val
    df_val_ticker = val_df[val_df['Ticker_ID'] == t_id]
    if len(df_val_ticker) > seq_length:
        indices = df_val_ticker.index
        mask_val = np.isin(val_df.index, indices)
        f_v = val_features_scaled[mask_val]
        t_v = val_ticker_id[mask_val]
        target_v = val_targets_scaled[mask_val]
        X_v, X_vi, y_v = create_sequences_for_ticker(f_v, t_v, target_v, seq_length)
        X_val_list.append(X_v)
        X_val_ticker_list.append(X_vi)
        y_val_list.append(y_v)

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

embedding_dim = 32
ticker_embedding = Embedding(input_dim=num_tickers, output_dim=embedding_dim, name='ticker_embedding')(ticker_input)

merged = concatenate([features_input, ticker_embedding], axis=-1)

x = LSTM(64, return_sequences=True)(merged)
x = Dropout(0.3)(x)  # เพิ่ม Dropout Regularization
x = LSTM(32)(x)  
x = Dropout(0.3)(x)  # เพิ่ม Dropout Regularization
output = Dense(1)(x)

model = Model(inputs=[features_input, ticker_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# เพิ่ม EarlyStopping และ ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
checkpoint = ModelCheckpoint('best_price_model_full.keras', monitor='val_loss', save_best_only=True, mode='min')

logging.info("เริ่มฝึกโมเดลสำหรับราคาหุ้นรวม (ใช้ Embedding สำหรับ Ticker)")

history = model.fit(
    [X_price_train, X_ticker_train], y_price_train,
    epochs=50,
    batch_size=32,
    validation_data=([X_price_val, X_ticker_val], y_price_val),
    verbose=1,
    shuffle=False,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

model.save('price_prediction_LSTM_model_embedding_full.h5')
logging.info("บันทึกโมเดลราคาหุ้นรวมเรียบร้อยแล้ว")

plot_training_history(history)

y_pred_scaled = model.predict([X_price_test, X_ticker_test])
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

plot_predictions(y_true[:200], y_pred[:200], "Test Set")
plot_residuals(y_true, y_pred, "Test Set")