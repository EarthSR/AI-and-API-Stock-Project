import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, GRU, BatchNormalization, Embedding,
    concatenate, Bidirectional, Layer, Masking, Conv1D, Flatten,
    Attention, MultiHeadAttention, Add, LayerNormalization, Multiply,
    GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber, MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import CosineDecay, ExponentialDecay
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score
)
import ta
import matplotlib.pyplot as plt
import joblib
import logging
from tensorflow.keras.losses import Loss
from tensorflow.keras.saving import register_keras_serializable

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
print("TensorFlow running on:", tf.config.list_physical_devices())

# ------------------------------------------------------------------------------------
# ฟังก์ชันคำนวณ Error / Custom Loss
# ------------------------------------------------------------------------------------
def custom_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE) แบบไม่ให้เกิด Infinity ถ้า y_true = 0
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_mask = y_true != 0
    if not np.any(nonzero_mask):
        return np.nan
    return np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100

def focal_loss_fixed(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    loss = -alpha * (y_true * tf.math.pow(1 - y_pred, gamma) * tf.math.log(y_pred)) \
           - (1 - alpha) * ((1 - y_true) * tf.math.pow(y_pred, gamma) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(loss)

def softmax_axis1(x):
    return tf.keras.activations.softmax(x, axis=1)

def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    nonzero_mask = denominator != 0
    if not np.any(nonzero_mask):
        return np.nan
    return np.mean(diff[nonzero_mask] / denominator[nonzero_mask]) * 100

def focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-8
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt))
        return loss
    return focal_loss_fixed

def quantile_loss(y_true, y_pred, quantile=0.5):
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))

def cosine_similarity_loss(y_true, y_pred):
    y_true = tf.keras.backend.l2_normalize(y_true, axis=-1)
    y_pred = tf.keras.backend.l2_normalize(y_pred, axis=-1)
    return -tf.reduce_mean(y_true * y_pred)

# เปลี่ยน Optimizer เป็น AdamW
from tensorflow.keras.optimizers import AdamW
@register_keras_serializable()
def focal_loss_fixed(y_true, y_pred, alpha=0.25, gamma=2.0):
    epsilon = 1e-8
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    loss = -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt))
    return loss

# ตั้งค่า Logging
logging.basicConfig(level=logging.INFO, filename='training.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------------------------------
# ฟังก์ชันสร้าง Sequence สำหรับ Multi-Task
# ------------------------------------------------------------------------------------
def create_sequences_for_ticker(features, ticker_ids, market_ids, targets_price, targets_dir, seq_length=10):
    """
    คืนค่า 5 รายการ:
      X_features, X_tickers, X_markets, Y_price, Y_dir
    สำหรับแต่ละ Sequence ยาว seq_length
    """
    X_features, X_tickers, X_markets = [], [], []
    Y_price, Y_dir = [], []

    for i in range(len(features) - seq_length):
        X_features.append(features[i : i + seq_length])
        X_tickers.append(ticker_ids[i : i + seq_length])
        X_markets.append(market_ids[i : i + seq_length])
        Y_price.append(targets_price[i + seq_length])
        Y_dir.append(targets_dir[i + seq_length])

    return (
        np.array(X_features),
        np.array(X_tickers),
        np.array(X_markets),
        np.array(Y_price),
        np.array(Y_dir),
    )

def plot_training_history(history):
    keys = list(history.history.keys())
    print("Keys in history:", keys)

    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Total Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Total Loss')
    plt.plot(history.history['val_loss'], label='Val Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # บันทึกเฉพาะ subplot แรก
    plt.tight_layout()
    plt.savefig('training_total_loss.png')
    plt.close()

def plot_predictions(y_true, y_pred, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='red', alpha=0.7)
    plt.title(f'True vs Predicted Prices for {ticker}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'predictions_{ticker}.png')
    plt.close()

def plot_residuals(y_true, y_pred, ticker):
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(residuals)), residuals, alpha=0.5)
    plt.hlines(y=0, xmin=0, xmax=len(residuals), colors='red')
    plt.title(f'Residuals for {ticker}')
    plt.xlabel('Sample')
    plt.ylabel('Residual')
    plt.tight_layout()
    plt.savefig(f'residuals_{ticker}.png')
    plt.close()

def cosine_similarity_loss(y_true, y_pred):
    y_true = K.cast(y_true, dtype=tf.float32)
    y_pred = K.cast(y_pred, dtype=tf.float32)
    y_true = K.l2_normalize(y_true + K.epsilon(), axis=-1)
    y_pred = K.l2_normalize(y_pred + K.epsilon(), axis=-1)
    return -K.mean(y_true * y_pred)

tf.keras.utils.get_custom_objects()["cosine_similarity_loss"] = cosine_similarity_loss

# ตรวจสอบ GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
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

# ------------------------------------------------------------------------------------
# 1) โหลดและเตรียมข้อมูล
# ------------------------------------------------------------------------------------
df = pd.read_csv('../Preproces/merged_stock_sentiment_financial.csv')

df['Sentiment'] = df['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
df['Change'] = df['Close'] - df['Open']
df['Change (%)'] = df['Close'].pct_change() * 100
upper_bound = df["Change (%)"].quantile(0.99)
lower_bound = df["Change (%)"].quantile(0.01)
df["Change (%)"] = np.clip(df["Change (%)"], lower_bound, upper_bound)

import ta
df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
df['RSI'].fillna(method='ffill', inplace=True)
df['RSI'].fillna(0, inplace=True)
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].rolling(window=9).mean()

bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
df['Bollinger_High'] = bollinger.bollinger_hband()
df['Bollinger_Low'] = bollinger.bollinger_lband()

atr = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
df['ATR'] = atr.average_true_range()

keltner = ta.volatility.KeltnerChannel(
    high=df['High'], low=df['Low'], close=df['Close'],
    window=20, window_atr=10
)
df['Keltner_High'] = keltner.keltner_channel_hband()
df['Keltner_Low'] = keltner.keltner_channel_lband()
df['Keltner_Middle'] = keltner.keltner_channel_mband()

window_cv = 10
df['High_Low_Diff'] = df['High'] - df['Low']
df['High_Low_EMA'] = df['High_Low_Diff'].ewm(span=window_cv, adjust=False).mean()
df['Chaikin_Vol'] = df['High_Low_EMA'].pct_change(periods=window_cv) * 100

window_dc = 20
df['Donchian_High'] = df['High'].rolling(window=window_dc).max()
df['Donchian_Low'] = df['Low'].rolling(window=window_dc).min()

psar = ta.trend.PSARIndicator(
    high=df['High'], low=df['Low'], close=df['Close'],
    step=0.02, max_step=0.2
)
df['PSAR'] = psar.psar()

financial_columns = [
    'Total Revenue', 'QoQ Growth (%)', 'Earnings Per Share (EPS)', 'ROE (%)',
    'Net Profit Margin (%)', 'Debt to Equity', 'P/E Ratio',
    'P/BV Ratio', 'Dividend Yield (%)'
]
df_financial = df[['Date', 'Ticker'] + financial_columns].drop_duplicates()
df_financial[financial_columns] = df_financial[financial_columns].where(df_financial[financial_columns].ne(0)).bfill()

stock_columns = [
    'RSI','EMA_12','EMA_26','MACD','MACD_Signal','Bollinger_High',
    'Bollinger_Low','ATR','Keltner_High','Keltner_Low','Keltner_Middle',
    'Chaikin_Vol','Donchian_High','Donchian_Low','PSAR','SMA_50','SMA_200'
]
df[stock_columns] = df[stock_columns].fillna(method='ffill')
df.fillna(0, inplace=True)

feature_columns = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment','total_news',
    'Total Revenue', 'QoQ Growth (%)','Earnings Per Share (EPS)','ROE (%)',
    'ATR','Keltner_High','Keltner_Low','Keltner_Middle','Chaikin_Vol','Donchian_High','Donchian_Low','PSAR',
    'Net Profit Margin (%)', 'Debt to Equity', 'P/E Ratio',
    'P/BV Ratio', 'Dividend Yield (%)','RSI', 'EMA_10', 'EMA_20', 'MACD', 'MACD_Signal',
    'Bollinger_High', 'Bollinger_Low','SMA_50', 'SMA_200'
]
joblib.dump(feature_columns, 'feature_columns.pkl')

df['Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df['TargetPrice'] = df['Close'].shift(-1)

df.dropna(subset=['Direction', 'TargetPrice'], inplace=True)

market_encoder = LabelEncoder()
df['Market_ID'] = market_encoder.fit_transform(df['Market_ID'])
num_markets = len(market_encoder.classes_)
joblib.dump(market_encoder, 'market_encoder.pkl')

ticker_encoder = LabelEncoder()
df['Ticker_ID'] = ticker_encoder.fit_transform(df['Ticker'])
num_tickers = len(ticker_encoder.classes_)
joblib.dump(ticker_encoder, 'ticker_encoder.pkl')

sorted_dates = df['Date'].unique()
train_cutoff = sorted_dates[int(len(sorted_dates) * 6 / 7)]
train_df = df[df['Date'] <= train_cutoff].copy()
test_df  = df[df['Date'] > train_cutoff].copy()

train_df.to_csv('train_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)
print("Train cutoff:", train_cutoff)
print("First date in train set:", train_df['Date'].min())
print("Last date in train set:", train_df['Date'].max())

# ------------------------------------------------------------------------------------
# 2) เตรียม Target (Price, Direction) + Per-Ticker Scaling
# ------------------------------------------------------------------------------------
train_features = train_df[feature_columns].values
test_features  = test_df[feature_columns].values

train_features[train_features == np.inf] = np.max(train_features[train_features != np.inf])
train_features[train_features == -np.inf] = np.min(train_features[train_features != -np.inf])

train_ticker_id = train_df['Ticker_ID'].values
train_market_id = train_df['Market_ID'].values
test_ticker_id  = test_df['Ticker_ID'].values
test_market_id  = test_df['Market_ID'].values

train_price = train_df['TargetPrice'].values.reshape(-1, 1)
test_price  = test_df['TargetPrice'].values.reshape(-1, 1)

train_dir = train_df['Direction'].values
test_dir  = test_df['Direction'].values

train_features_scaled = np.zeros_like(train_features)
train_price_scaled    = np.zeros_like(train_price)
test_features_scaled  = np.zeros_like(test_features)
test_price_scaled     = np.zeros_like(test_price)

ticker_scalers = {}

unique_tickers_train = train_df['Ticker_ID'].unique()
for t_id in unique_tickers_train:
    mask_train = (train_ticker_id == t_id)
    X_part = train_features[mask_train]
    y_part = train_price[mask_train]

    scaler_f = RobustScaler()
    scaler_p = RobustScaler()

    X_scaled = scaler_f.fit_transform(X_part)
    y_scaled = scaler_p.fit_transform(y_part)

    train_features_scaled[mask_train] = X_scaled
    train_price_scaled[mask_train]    = y_scaled

    ticker_scalers[t_id] = {
        'feature_scaler': scaler_f,
        'price_scaler': scaler_p
    }

unique_tickers_test = test_df['Ticker_ID'].unique()
for t_id in unique_tickers_test:
    if t_id not in ticker_scalers:
        print(f"Ticker {t_id} not found in training scalers. Skipping.")
        continue
    mask_test = (test_ticker_id == t_id)
    X_part = test_features[mask_test]
    y_part = test_price[mask_test]

    scaler_f = ticker_scalers[t_id]['feature_scaler']
    scaler_p = ticker_scalers[t_id]['price_scaler']

    X_scaled = scaler_f.transform(X_part)
    y_scaled = scaler_p.transform(y_part)

    test_features_scaled[mask_test] = X_scaled
    test_price_scaled[mask_test]    = y_scaled

np.save('test_features.npy', test_features_scaled)
np.save('test_price.npy',   test_price_scaled)
np.save('train_features.npy', train_features_scaled)
np.save('train_price.npy',   train_price_scaled)
print("✅ บันทึก test_features.npy และ test_price.npy สำเร็จ!")

seq_length = 10

# ------------------------------------------------------------------------------------
# 3) สร้าง Sequence (ต่อ Ticker) สำหรับ Multi-Task (Price + Direction)
# ------------------------------------------------------------------------------------
X_train_list, X_train_ticker_list, X_train_market_list = [], [], []
y_price_train_list, y_dir_train_list = [], []

X_test_list, X_test_ticker_list, X_test_market_list = [], [], []
y_price_test_list, y_dir_test_list = [], []

for t_id in range(num_tickers):
    df_train_ticker = train_df[train_df['Ticker_ID'] == t_id]
    if len(df_train_ticker) > seq_length:
        idx_train = df_train_ticker.index
        mask_train = np.isin(train_df.index, idx_train)

        f_t = train_features_scaled[mask_train]
        t_t = train_ticker_id[mask_train]
        m_t = train_market_id[mask_train]
        p_t = train_price_scaled[mask_train]
        d_t = train_dir[mask_train]

        (Xf, Xt, Xm, Yp, Yd) = create_sequences_for_ticker(
            f_t, t_t, m_t, p_t, d_t, seq_length
        )
        X_train_list.append(Xf)
        X_train_ticker_list.append(Xt)
        X_train_market_list.append(Xm)
        y_price_train_list.append(Yp)
        y_dir_train_list.append(Yd)

    df_test_ticker = test_df[test_df['Ticker_ID'] == t_id]
    if len(df_test_ticker) > seq_length:
        idx_test = df_test_ticker.index
        mask_test = np.isin(test_df.index, idx_test)

        f_s = test_features_scaled[mask_test]
        t_s = test_ticker_id[mask_test]
        m_s = test_market_id[mask_test]
        p_s = test_price_scaled[mask_test]
        d_s = test_dir[mask_test]

        (Xs, Xts, Xms, Yps, Yds) = create_sequences_for_ticker(
            f_s, t_s, m_s, p_s, d_s, seq_length
        )
        X_test_list.append(Xs)
        X_test_ticker_list.append(Xts)
        X_test_market_list.append(Xms)
        y_price_test_list.append(Yps)
        y_dir_test_list.append(Yds)

if len(X_train_list) > 0:
    X_price_train = np.concatenate(X_train_list, axis=0)
    X_ticker_train = np.concatenate(X_train_ticker_list, axis=0)
    X_market_train = np.concatenate(X_train_market_list, axis=0)
    y_price_train = np.concatenate(y_price_train_list, axis=0)
    y_dir_train   = np.concatenate(y_dir_train_list, axis=0)
else:
    X_price_train, X_ticker_train, X_market_train, y_price_train, y_dir_train = (np.array([]),)*5

if len(X_test_list) > 0:
    X_price_test = np.concatenate(X_test_list, axis=0)
    X_ticker_test = np.concatenate(X_test_ticker_list, axis=0)
    X_market_test = np.concatenate(X_test_market_list, axis=0)
    y_price_test  = np.concatenate(y_price_test_list, axis=0)
    y_dir_test    = np.concatenate(y_dir_test_list, axis=0)
else:
    X_price_test, X_ticker_test, X_market_test, y_price_test, y_dir_test = (np.array([]),)*5

print("X_price_train shape :", X_price_train.shape)
print("X_ticker_train shape:", X_ticker_train.shape)
print("y_price_train shape :", y_price_train.shape)
print("y_dir_train shape   :", y_dir_train.shape)

num_feature = train_features_scaled.shape[1]

@register_keras_serializable()
def quantile_loss(y_true, y_pred, quantile=0.5):
    error = y_true - y_pred
    return K.mean(K.maximum(quantile * error, (quantile - 1) * error))

# -------------------------- สร้างโมเดล Multi-Task --------------------------
from tensorflow.keras.layers import (
    MultiHeadAttention, LayerNormalization, Dense, Dropout, Add,
    Input, LSTM, GlobalAveragePooling1D, Conv1D, Bidirectional, Embedding
)

features_input = Input(shape=(seq_length, num_feature), name='features_input')
ticker_input   = Input(shape=(seq_length,), name='ticker_input')
market_input   = Input(shape=(seq_length,), name='market_input')

embedding_dim = 16
ticker_embedding = Embedding(
    input_dim=num_tickers,
    output_dim=embedding_dim,
    name="ticker_embedding"
)(ticker_input)
ticker_embedding = Dense(8, activation="relu")(ticker_embedding)

embedding_dim_market = 4
market_embedding = Embedding(
    input_dim=num_markets,
    output_dim=embedding_dim_market,
    name="market_embedding"
)(market_input)
market_embedding = Dense(4, activation="relu")(market_embedding)

merged = concatenate([features_input, ticker_embedding, market_embedding], axis=-1)

x = Bidirectional(LSTM(64, return_sequences=True))(merged)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(32, return_sequences=False))(x)
x = Dropout(0.2)(x)
shared_repr = Dense(64, activation="relu")(x)

price_head = Dense(32, activation="relu")(shared_repr)
price_head = Dropout(0.2)(price_head)
price_output = Dense(1, name="price_output")(price_head)

dir_head = Dense(32, activation="relu")(shared_repr)
dir_head = Dropout(0.2)(dir_head)
direction_output = Dense(1, activation="sigmoid", name="direction_output")(dir_head)

model = Model(
    inputs=[features_input, ticker_input, market_input],
    outputs=[price_output, direction_output]
)

optimizer = AdamW(learning_rate=3e-4, weight_decay=1e-6)

model.compile(
    optimizer=optimizer,
    loss={
        "price_output": tf.keras.losses.Huber(delta=1.0),
        "direction_output": "binary_crossentropy"
    },
    loss_weights={"price_output": 0.8, "direction_output": 0.2},
    metrics={
        "price_output": ["mae"],
        "direction_output": ["accuracy"]
    }
)

model.summary()

early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
checkpoint = ModelCheckpoint("best_multi_task_model.keras", monitor="val_loss", save_best_only=True, mode="min")

callbacks = [early_stopping, lr_scheduler, checkpoint]

history = model.fit(
    [X_price_train, X_ticker_train, X_market_train],
    {"price_output": y_price_train, "direction_output": y_dir_train},
    epochs=200,
    batch_size=16,
    verbose=1,
    shuffle=False,
    validation_split=0.1,
    callbacks=callbacks
)

model.save("multi_task_model.keras")
print("✅ โมเดลที่ดีที่สุดถูกบันทึกแล้ว!")

# ------------------------------------------------------------------------------------
# 5) ฟังก์ชัน Walk-Forward Validation (ใช้ Per-Ticker Scaler)
# ------------------------------------------------------------------------------------
def walk_forward_validation_multi_task_batch(
    model,
    df,
    feature_columns,
    ticker_scalers,   # Dict ของ Scaler per Ticker
    ticker_encoder,
    market_encoder,
    seq_length=10,
    retrain_frequency=5
):
    """
    ทำ Walk-Forward Validation แบบ Multi-Task (Price + Direction)
    และ Online Learning เป็น batch
    โดยใช้ Per-Ticker Scaling (ticker_scalers) ตรงกับที่เทรน
    """

    all_predictions = []
    tickers = df['Ticker'].unique()

    for ticker in tickers:
        print(f"\nProcessing Ticker: {ticker}")
        df_ticker = df[df['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)

        if len(df_ticker) < seq_length + 1:
            print(f"Not enough data for ticker {ticker}, skipping...")
            continue

        batch_features = []
        batch_tickers = []
        batch_market  = []
        batch_price   = []
        batch_dir     = []

        for i in range(len(df_ticker) - seq_length):
            historical_data = df_ticker.iloc[i : i + seq_length]
            target_data     = df_ticker.iloc[i + seq_length]

            t_id = historical_data['Ticker_ID'].iloc[-1]
            if t_id not in ticker_scalers:
                print(f"Ticker {ticker} (Ticker_ID={t_id}) not found in ticker_scalers, skipping this portion.")
                continue

            scaler_f = ticker_scalers[t_id]['feature_scaler']
            scaler_p = ticker_scalers[t_id]['price_scaler']

            features = historical_data[feature_columns].values
            ticker_ids = historical_data['Ticker_ID'].values
            market_ids = historical_data['Market_ID'].values

            features_scaled = scaler_f.transform(features)

            X_features = features_scaled.reshape(1, seq_length, len(feature_columns))
            X_ticker   = ticker_ids.reshape(1, seq_length)
            X_market   = market_ids.reshape(1, seq_length)

            pred_price_scaled, pred_dir_prob = model.predict([X_features, X_ticker, X_market], verbose=0)

            predicted_price = scaler_p.inverse_transform(pred_price_scaled)[0][0]
            predicted_dir = 1 if pred_dir_prob[0][0] >= 0.5 else 0

            actual_price = target_data['Close']
            future_date  = target_data['Date']
            last_close   = historical_data.iloc[-1]['Close']
            actual_dir   = 1 if (target_data['Close'] > last_close) else 0

            all_predictions.append({
                'Ticker': ticker,
                'Date': future_date,
                'Predicted_Price': predicted_price,
                'Actual_Price': actual_price,
                'Predicted_Dir': predicted_dir,
                'Actual_Dir': actual_dir
            })

            batch_features.append(X_features)
            batch_tickers.append(X_ticker)
            batch_market.append(X_market)

            y_price_true_scaled = scaler_p.transform(np.array([[actual_price]], dtype=float))
            batch_price.append(y_price_true_scaled)

            y_dir_true = np.array([actual_dir], dtype=float)
            batch_dir.append(y_dir_true)

            if (i+1) % retrain_frequency == 0 or (i == (len(df_ticker) - seq_length - 1)):
                bf = np.concatenate(batch_features, axis=0)
                bt = np.concatenate(batch_tickers, axis=0)
                bm = np.concatenate(batch_market, axis=0)
                bp = np.concatenate(batch_price, axis=0)
                bd = np.concatenate(batch_dir, axis=0)

                model.fit(
                    [bf, bt, bm],
                    {
                        'price_output': bp,
                        'direction_output': bd
                    },
                    epochs=1,
                    batch_size=len(bf),
                    verbose=0,
                    shuffle=False
                )
                batch_features = []
                batch_tickers  = []
                batch_market   = []
                batch_price    = []
                batch_dir      = []

    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv('predictions_multi_task_walkforward_batch.csv', index=False)
    print("\n✅ Saved predictions to 'predictions_multi_task_walkforward_batch.csv'")

    # คำนวณ Metrics
    metrics_dict = {}
    for ticker, group in predictions_df.groupby('Ticker'):
        actual_prices = group['Actual_Price'].values
        pred_prices   = group['Predicted_Price'].values
        actual_dirs   = group['Actual_Dir'].values
        pred_dirs     = group['Predicted_Dir'].values

        mae_val  = mean_absolute_error(actual_prices, pred_prices)
        mse_val  = mean_squared_error(actual_prices, pred_prices)
        rmse_val = np.sqrt(mse_val)
        mape_val = custom_mape(actual_prices, pred_prices)
        smape_val= smape(actual_prices, pred_prices)
        r2_val   = r2_score(actual_prices, pred_prices)

        dir_acc  = accuracy_score(actual_dirs, pred_dirs)
        dir_f1   = f1_score(actual_dirs, pred_dirs)
        dir_precision = precision_score(actual_dirs, pred_dirs)
        dir_recall = recall_score(actual_dirs, pred_dirs)

        metrics_dict[ticker] = {
            'MAE': mae_val,
            'MSE': mse_val,
            'RMSE': rmse_val,
            'MAPE': mape_val,
            'SMAPE': smape_val,
            'R2 Score': r2_val,
            'Direction Accuracy': dir_acc,
            'Direction F1 Score': dir_f1,
            'Direction Precision': dir_precision,
            'Direction Recall': dir_recall
        }

    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    metrics_df.to_csv('metrics_per_ticker_multi_task_batch.csv')
    print("✅ Saved metrics to 'metrics_per_ticker_multi_task_batch.csv'")

    return predictions_df, metrics_dict

# ------------------------------------------------------------------------------------
# 6) เรียกใช้งาน Walk-Forward Validation สำหรับ Multi-Task
# ------------------------------------------------------------------------------------
best_multi_model = load_model(
    "best_multi_task_model.keras",
    custom_objects={
        "quantile_loss": quantile_loss,
        "focal_loss_fixed": focal_loss_fixed,
        "softmax_axis1": softmax_axis1
    },
    safe_mode=False
)

predictions_df, results_per_ticker = walk_forward_validation_multi_task_batch(
    model = best_multi_model,
    df = test_df,
    feature_columns = feature_columns,
    ticker_scalers = ticker_scalers,  # ใช้ dict Per-Ticker Scaler แทน scaler_features/scaler_target
    ticker_encoder = ticker_encoder,
    market_encoder = market_encoder,
    seq_length = 10,
    retrain_frequency=1
)

for ticker, metrics in results_per_ticker.items():
    print(f"\nMetrics for {ticker}:")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  MSE:  {metrics['MSE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.4f}")
    print(f"  SMAPE:{metrics['SMAPE']:.4f}")
    print(f"  R2 Score: {metrics['R2 Score']:.4f}")
    print(f"  Direction Accuracy: {metrics['Direction Accuracy']:.4f}")
    print(f"  Direction F1 Score: {metrics['Direction F1 Score']:.4f}")
    print(f"  Direction Precision: {metrics['Direction Precision']:.4f}")
    print(f"  Direction Recall: {metrics['Direction Recall']:.4f}")

metrics_df = pd.DataFrame.from_dict(results_per_ticker, orient='index')
metrics_df.to_csv('metrics_per_ticker_multi_task.csv', index=True)
print("\nSaved metrics per ticker to 'metrics_per_ticker_multi_task.csv'")

all_data = []

for ticker, group in predictions_df.groupby('Ticker'):
    for idx, row in group.iterrows():
        date_val = row['Date']
        actual_p = row['Actual_Price']
        pred_p   = row['Predicted_Price']
        actual_d = row['Actual_Dir']
        pred_d   = row['Predicted_Dir']
        
        all_data.append([ticker, date_val, actual_p, pred_p, actual_d, pred_d])

# จากนั้นสร้าง DataFrame ใหม่ได้ตามต้องการ
prediction_df = pd.DataFrame(all_data, columns=[
    'Ticker','Date','Actual_Price','Predicted_Price','Actual_Dir','Predicted_Dir'
])
prediction_df.to_csv('all_predictions_per_day_multi_task.csv', index=False)
print("Saved actual and predicted (price & direction) to 'all_predictions_per_day_multi_task.csv'")
