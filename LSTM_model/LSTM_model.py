# -*- coding: utf-8 -*-
"""
Production-ready Multi-Task (Price + Direction) for Stock AI Prediction
- โครงสร้างเดิม 2-head (price + direction)
- ปรับปรุงเพื่อ production:
  • Per-ticker calibration + per-ticker threshold (MCC)
  • Online learning gating (ความเชื่อมั่น & ความไม่แน่นอน)
  • ลดความถี่ mini-retrain และต้องมี sample >= 5
  • ฝึกหัวราคาเป็น log-return แล้วแปลงกลับเป็นราคาปิดระหว่าง inference
  • พิมพ์สรุปผลท้ายรันแบบ CSV-ready + บันทึกไฟล์
  • (อัปเดต) เปิดอัปเดตหัวราคาแบบเบา ๆ เมื่อผ่าน gate, เพิ่ม MC samples,
    ผ่อน gate เล็กน้อย และ clip ค่าก่อน inverse transform
"""

# =============================================================================
# 0) Imports & Global Config
# =============================================================================
import os, random, json, joblib, logging, warnings, sys, time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # ใช้ CPU เพื่อความคงที่ (ปรับได้ตามเครื่อง)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEED = 42
random.seed(SEED); np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, Embedding,
                                     Bidirectional, concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from tensorflow.keras.optimizers.schedules import CosineDecay
try:
    from tensorflow.keras.optimizers import AdamW  # TF >= 2.11
except Exception:
    from tensorflow_addons.optimizers import AdamW

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, f1_score, precision_score, recall_score,
                             matthews_corrcoef)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.isotonic import IsotonicRegression

print("TensorFlow devices:", tf.config.list_physical_devices())

# =============================================================================
# 1) Best Params + Switches
# =============================================================================
BEST_PARAMS = {
    'chunk_size': 100,
    'embedding_dim': 24,
    'LSTM_units_1': 48,
    'LSTM_units_2': 24,
    'dropout_rate': 0.15,
    'dense_units': 66,
    'learning_rate': 1.7e-4,     # 0.00017
    'retrain_frequency': 9,      # เดิม 3 → 9 เพื่อลด drift
    'seq_length': 10
}
MC_DIR_SAMPLES = 8               # (อัปเดต) เพิ่มรอบ MC dropout

# --- Online learning gates ---
CONF_GATE = True
UNC_MAX   = 0.15                 # (อัปเดต) ผ่อนความไม่แน่นอนนิดหน่อย
MARGIN    = 0.08                 # (อัปเดต) ลด margin เพื่อให้ได้อัปเดตมากขึ้น

# (อัปเดต) เปิดการอัปเดตหัวราคาแบบน้ำหนักเบา เมื่อผ่าน gate
ALLOW_PRICE_ONLINE = True

# โหมด target ของหัวราคา: 'logret' (แนะนำ) หรือ 'price'
PRICE_TARGET_MODE = 'logret'

# =============================================================================
# 2) Losses / Utils
# =============================================================================
@tf.keras.utils.register_keras_serializable()
def quantile_loss(y_true, y_pred, quantile=0.5):
    e = y_true - y_pred
    return tf.reduce_mean(tf.maximum(quantile*e, (quantile-1)*e))

def focal_weighted_binary_crossentropy(class_weights, gamma=1.95, alpha_pos=0.7):
    w0, w1 = tf.cast(class_weights[0], tf.float32), tf.cast(class_weights[1], tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = tf.constant(1e-7, tf.float32)
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        w = tf.where(tf.equal(y_true, 1.0), w1, w0)
        alpha = tf.where(tf.equal(y_true, 1.0), alpha_pos, 1.0 - alpha_pos)
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        focal = tf.pow(1.0 - pt, gamma)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return tf.reduce_mean(bce * w * alpha * focal)
    return loss

def sanitize(arr):
    arr = np.asarray(arr, dtype=float)
    mask = np.isfinite(arr)
    if not mask.all():
        median = np.nanmedian(arr[mask])
        arr[~mask] = median
    arr[np.isnan(arr)] = np.nanmedian(arr[np.isfinite(arr)])
    return arr

# =============================================================================
# 3) Load Data
# =============================================================================
DATA_PATH = '../Preproces/data/Stock/merged_stock_sentiment_financial.csv'
df = pd.read_csv(DATA_PATH)

# =============================================================================
# 4) Clean & Feature Engineering (no leakage)
# =============================================================================
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# Sentiment mapping
df['Sentiment'] = df['Sentiment'].map({'Positive':1, 'Negative':-1, 'Neutral':0}).fillna(0).astype(int)

# Basic changes
df['Change'] = df['Close'] - df['Open']
df['Change (%)'] = df.groupby('Ticker')['Close'].pct_change() * 100
upper = df['Change (%)'].quantile(0.99)
lower = df['Change (%)'].quantile(0.01)
df['Change (%)'] = df['Change (%)'].clip(lower, upper)

# ----- TA per Ticker -----
import ta
def add_ta(g):
    g = g.copy()
    g['EMA_12'] = g['Close'].ewm(span=12, adjust=False).mean()
    g['EMA_26'] = g['Close'].ewm(span=26, adjust=False).mean()
    g['EMA_10'] = g['Close'].ewm(span=10, adjust=False).mean()
    g['EMA_20'] = g['Close'].ewm(span=20, adjust=False).mean()
    g['SMA_50']  = g['Close'].rolling(50, min_periods=1).mean()
    g['SMA_200'] = g['Close'].rolling(200, min_periods=1).mean()

    rsi = ta.momentum.RSIIndicator(close=g['Close'], window=14)
    g['RSI'] = rsi.rsi()
    g['RSI'] = g['RSI'].fillna(g['RSI'].rolling(window=5, min_periods=1).mean())

    g['MACD'] = g['EMA_12'] - g['EMA_26']
    g['MACD_Signal'] = g['MACD'].rolling(9, min_periods=1).mean()

    bb = ta.volatility.BollingerBands(close=g['Close'], window=20, window_dev=2)
    g['Bollinger_High'] = bb.bollinger_hband()
    g['Bollinger_Low']  = bb.bollinger_lband()

    atr = ta.volatility.AverageTrueRange(high=g['High'], low=g['Low'], close=g['Close'], window=14)
    g['ATR'] = atr.average_true_range()

    kc = ta.volatility.KeltnerChannel(high=g['High'], low=g['Low'], close=g['Close'], window=20, window_atr=10)
    g['Keltner_High']   = kc.keltner_channel_hband()
    g['Keltner_Low']    = kc.keltner_channel_lband()
    g['Keltner_Middle'] = kc.keltner_channel_mband()

    g['High_Low_Diff'] = g['High'] - g['Low']
    g['High_Low_EMA']  = g['High_Low_Diff'].ewm(span=10, adjust=False).mean()
    g['Chaikin_Vol']   = g['High_Low_EMA'].pct_change(10) * 100
    g['Donchian_High'] = g['High'].rolling(20, min_periods=1).max()
    g['Donchian_Low']  = g['Low'].rolling(20, min_periods=1).min()
    g['PSAR'] = ta.trend.PSARIndicator(high=g['High'], low=g['Low'], close=g['Close'],
                                       step=0.02, max_step=0.2).psar()
    return g

df = df.groupby('Ticker', group_keys=False).apply(add_ta)

# Market tagging
us_stock = ['AAPL','NVDA','MSFT','AMZN','GOOGL','META','TSLA','AVGO','TSM','AMD']
thai_stock = ['ADVANC','TRUE','DITTO','DIF','INSET','JMART','INET','JAS','HUMAN']
df['Market_ID'] = np.where(df['Ticker'].isin(us_stock), 'US',
                   np.where(df['Ticker'].isin(thai_stock), 'TH', 'OTHER'))
df['Market_ID'] = df['Market_ID'].fillna('OTHER')

# Financials (ffill within ticker; no bfill)
financial_columns = [
    'Total Revenue','QoQ Growth (%)','Earnings Per Share (EPS)','ROE (%)',
    'Net Profit Margin (%)','Debt to Equity','P/E Ratio','P/BV Ratio','Dividend Yield (%)'
]
for c in financial_columns:
    if c not in df.columns:
        df[c] = np.nan
df[financial_columns] = df[financial_columns].replace(0, np.nan)
df[financial_columns] = df.groupby('Ticker')[financial_columns].ffill()

# Feature list
feature_columns = [
    'Open','High','Low','Close','Volume','Change (%)','Sentiment',
    'positive_news','negative_news','neutral_news',
    'Total Revenue','QoQ Growth (%)','Earnings Per Share (EPS)','ROE (%)',
    'ATR','Keltner_High','Keltner_Low','Keltner_Middle','Chaikin_Vol',
    'Donchian_High','Donchian_Low','PSAR',
    'Net Profit Margin (%)','Debt to Equity','P/E Ratio','P/BV Ratio','Dividend Yield (%)',
    'RSI','EMA_10','EMA_20','MACD','MACD_Signal',
    'Bollinger_High','Bollinger_Low','SMA_50','SMA_200'
]
for c in feature_columns:
    if c not in df.columns: df[c] = 0.0

# fill ต่อหุ้น
df[feature_columns] = (df.groupby('Ticker')[feature_columns]
                         .apply(lambda g: g.fillna(method='ffill'))
                         .reset_index(level=0, drop=True))
df[feature_columns] = df[feature_columns].fillna(0.0)

# Targets per-ticker
# - price_next = Close(t+1)
# - direction = 1 if Close(t+1) > Close(t)
df['TargetPrice'] = df.groupby('Ticker')['Close'].shift(-1)
df['Direction']   = (df.groupby('Ticker')['Close'].shift(-1) > df['Close']).astype(int)
df = df.dropna(subset=['TargetPrice','Direction']).reset_index(drop=True)

# =============================================================================
# 5) Encoders & Splits
# =============================================================================
market_encoder = LabelEncoder()
df['Market_ID_enc'] = market_encoder.fit_transform(df['Market_ID'])
num_markets = len(market_encoder.classes_)
joblib.dump(market_encoder, 'market_encoder.pkl')

ticker_encoder = LabelEncoder()
df['Ticker_ID'] = ticker_encoder.fit_transform(df['Ticker'])
num_tickers = len(ticker_encoder.classes_)
joblib.dump(ticker_encoder, 'ticker_encoder.pkl')

sorted_dates = df['Date'].unique()
train_cutoff = sorted_dates[int(len(sorted_dates) * 6 / 7)]
train_df = df[df['Date'] <= train_cutoff].copy()
test_df  = df[df['Date']  > train_cutoff].copy()
train_df.to_csv('train_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)
print("Train cutoff:", train_cutoff)

# =============================================================================
# 6) Per-ticker Robust Scaler (features + price-target or return-target)
# =============================================================================
SEQ_LEN = int(BEST_PARAMS['seq_length'])

# เตรียม target ของหัวราคา ตามโหมด
if PRICE_TARGET_MODE == 'logret':
    # y_price = log(Close(t+1)/Close(t))
    train_df['PriceTargetRaw'] = np.log(train_df['TargetPrice'] / train_df['Close'])
    test_df['PriceTargetRaw']  = np.log(test_df['TargetPrice']  / test_df['Close'])
else:  # 'price'
    train_df['PriceTargetRaw'] = train_df['TargetPrice']
    test_df['PriceTargetRaw']  = test_df['TargetPrice']

train_features = train_df[feature_columns].values.astype(float)
test_features  = test_df[feature_columns].values.astype(float)

train_price_t  = train_df['PriceTargetRaw'].values.reshape(-1,1).astype(float)
test_price_t   = test_df['PriceTargetRaw'].values.reshape(-1,1).astype(float)

train_ticker_id = train_df['Ticker_ID'].values
test_ticker_id  = test_df['Ticker_ID'].values

train_features = sanitize(train_features)
test_features  = sanitize(test_features)
train_price_t  = sanitize(train_price_t)
test_price_t   = sanitize(test_price_t)

train_features_scaled = np.zeros_like(train_features)
test_features_scaled  = np.zeros_like(test_features)
train_price_scaled    = np.zeros_like(train_price_t)
test_price_scaled     = np.zeros_like(test_price_t)

ticker_scalers = {}
id2ticker = {}
for t_id in np.unique(train_ticker_id):
    gmask = (train_ticker_id == t_id)
    X_part = train_features[gmask]
    y_part = train_price_t[gmask]
    fs = RobustScaler(); ps = RobustScaler()
    Xs = fs.fit_transform(X_part)
    ys = ps.fit_transform(y_part)
    train_features_scaled[gmask] = Xs
    train_price_scaled[gmask]    = ys
    ticker_name = train_df.loc[gmask, 'Ticker'].iloc[0]
    id2ticker[t_id] = ticker_name
    ticker_scalers[t_id] = {
        'feature_scaler': fs,
        'price_scaler': ps,
        'ticker': ticker_name,
        'price_target_mode': PRICE_TARGET_MODE
    }

for t_id in np.unique(test_ticker_id):
    if t_id not in ticker_scalers:  # unseen ticker
        continue
    gmask = (test_ticker_id == t_id)
    fs = ticker_scalers[t_id]['feature_scaler']
    ps = ticker_scalers[t_id]['price_scaler']
    test_features_scaled[gmask] = fs.transform(test_features[gmask])
    test_price_scaled[gmask]    = ps.transform(test_price_t[gmask])

joblib.dump(ticker_scalers, 'ticker_scalers.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')

# =============================================================================
# 7) Build sequences per ticker
# =============================================================================
def create_sequences_for_ticker(features, ticker_ids, market_ids, targets_price, targets_dir, seq_length=SEQ_LEN):
    X_features, X_tickers, X_markets, Y_price, Y_dir = [], [], [], [], []
    for i in range(len(features) - seq_length):
        X_features.append(features[i:i+seq_length])
        X_tickers.append(ticker_ids[i:i+seq_length])
        X_markets.append(market_ids[i:i+seq_length])
        Y_price.append(targets_price[i+seq_length])
        Y_dir.append(targets_dir[i+seq_length])
    return (np.array(X_features), np.array(X_tickers), np.array(X_markets),
            np.array(Y_price), np.array(Y_dir))

def build_dataset_sequences(base_df, features_scaled, price_scaled, seq_length=SEQ_LEN):
    Xf_list, Xt_list, Xm_list, Yp_list, Yd_list = [], [], [], [], []
    for t_id in range(num_tickers):
        idx = base_df.index[base_df['Ticker_ID']==t_id].tolist()
        if len(idx) <= seq_length:
            continue
        mask = np.isin(base_df.index, idx)
        f = features_scaled[mask]
        p = price_scaled[mask]
        d = base_df.loc[mask, 'Direction'].values
        t = base_df.loc[mask, 'Ticker_ID'].values
        m = base_df.loc[mask, 'Market_ID_enc'].values
        Xf, Xt, Xm, Yp, Yd = create_sequences_for_ticker(f, t, m, p, d, seq_length)
        if len(Xf):
            Xf_list.append(Xf); Xt_list.append(Xt); Xm_list.append(Xm)
            Yp_list.append(Yp); Yd_list.append(Yd)
    if len(Xf_list)==0:
        return (np.zeros((0,seq_length,len(feature_columns))),)*5
    return (np.concatenate(Xf_list, axis=0),
            np.concatenate(Xt_list, axis=0),
            np.concatenate(Xm_list, axis=0),
            np.concatenate(Yp_list, axis=0),
            np.concatenate(Yd_list, axis=0))

X_price_train, X_ticker_train, X_market_train, y_price_train, y_dir_train = build_dataset_sequences(
    train_df, train_features_scaled, train_price_scaled, SEQ_LEN
)
X_price_test, X_ticker_test, X_market_test, y_price_test, y_dir_test = build_dataset_sequences(
    test_df, test_features_scaled, test_price_scaled, SEQ_LEN
)

print("Train shapes:",
      X_price_train.shape, X_ticker_train.shape, X_market_train.shape,
      y_price_train.shape, y_dir_train.shape)

num_feature = len(feature_columns)

# =============================================================================
# 8) Model (ใช้ best params; ไม่มี recurrent_dropout เพื่อความเสถียร/เร็ว)
# =============================================================================
features_input = Input(shape=(SEQ_LEN, num_feature), name='features_input')
ticker_input   = Input(shape=(SEQ_LEN,), name='ticker_input')
market_input   = Input(shape=(SEQ_LEN,), name='market_input')

embedding_dim_ticker = int(BEST_PARAMS['embedding_dim'])
embedding_dim_market = 8

tick_emb = Embedding(input_dim=num_tickers, output_dim=embedding_dim_ticker,
                     embeddings_regularizer=tf.keras.regularizers.l2(1e-7),
                     name="ticker_embedding")(ticker_input)
tick_emb = Dense(16, activation="relu")(tick_emb)

mkt_emb = Embedding(input_dim=num_markets, output_dim=embedding_dim_market,
                    embeddings_regularizer=tf.keras.regularizers.l2(1e-7),
                    name="market_embedding")(market_input)
mkt_emb = Dense(8, activation="relu")(mkt_emb)

merged = concatenate([features_input, tick_emb, mkt_emb], axis=-1)

x = Bidirectional(LSTM(int(BEST_PARAMS['LSTM_units_1']), return_sequences=True,
                       dropout=float(BEST_PARAMS['dropout_rate'])))(merged)
x = Dropout(float(BEST_PARAMS['dropout_rate']))(x)
x = Bidirectional(LSTM(int(BEST_PARAMS['LSTM_units_2']), return_sequences=False,
                       dropout=float(BEST_PARAMS['dropout_rate'])))(x)
x = Dropout(float(BEST_PARAMS['dropout_rate']))(x)

shared = Dense(int(BEST_PARAMS['dense_units']), activation="relu",
               kernel_regularizer=tf.keras.regularizers.l2(1e-6))(x)

price_head = Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-6))(shared)
price_head = Dropout(0.22)(price_head)
price_output = Dense(1, name="price_output")(price_head)

dir_head = Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-6))(shared)
dir_head = Dropout(0.22)(dir_head)
direction_output = Dense(1, activation="sigmoid", name="direction_output")(dir_head)

model = Model(inputs=[features_input, ticker_input, market_input],
              outputs=[price_output, direction_output])

# Class weights for direction
direction_classes = np.unique(y_dir_train.flatten().astype(int))
class_weights_arr = compute_class_weight('balanced', classes=direction_classes, y=y_dir_train.flatten().astype(int))
class_weight_dict = {int(k):float(v) for k,v in zip(direction_classes, class_weights_arr)}
print("Class weights:", class_weight_dict)

# Optimizer & compile (CosineDecay)
BATCH_SIZE = 33
VAL_SPLIT  = 0.12
EPOCHS     = 105
steps_per_epoch = max(1, int(len(X_price_train)*(1-VAL_SPLIT)) // BATCH_SIZE)
decay_steps = max(1, int(steps_per_epoch * EPOCHS * 1.4))

lr_schedule = CosineDecay(initial_learning_rate=float(BEST_PARAMS['learning_rate']),
                          decay_steps=decay_steps, alpha=9e-6)
optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1.4e-5, clipnorm=0.95)

model.compile(
    optimizer=optimizer,
    loss={
        "price_output": tf.keras.losses.Huber(delta=0.75),
        "direction_output": focal_weighted_binary_crossentropy(class_weight_dict, gamma=1.95)
    },
    loss_weights={"price_output": 0.39, "direction_output": 0.61},
    metrics={"price_output":[tf.keras.metrics.MeanAbsoluteError()],
             "direction_output":[tf.keras.metrics.BinaryAccuracy()]}
)

print("\nModel Summary:")
model.summary()
print("Total params:", model.count_params())

# =============================================================================
# 9) Callbacks & Train
# =============================================================================
class StabilityCallback(Callback):
    def __init__(self): super().__init__(); self.best=float('inf')
    def on_epoch_end(self, epoch, logs=None):
        val_mae = logs.get('val_price_output_mean_absolute_error', 0)
        val_acc = logs.get('val_direction_output_binary_accuracy', 0)
        score = val_mae*2 + (1-val_acc)*1.5
        if score < self.best:
            self.best = score
            print(f"  🎯 New best combined score: {score:.4f}")

early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True,
                               verbose=1, min_delta=1.2e-4, start_from_epoch=12)
checkpoint = ModelCheckpoint("best_v6_plus_minimal_tuning_v2_final_model.keras",
                             monitor="val_loss", save_best_only=True, mode="min", verbose=1)
csv_logger = CSVLogger('v6_plus_minimal_tuning_v2_final_training_log.csv')
callbacks = [early_stopping, checkpoint, csv_logger, StabilityCallback()]

history = model.fit(
    [X_price_train, X_ticker_train, X_market_train],
    {"price_output": y_price_train, "direction_output": y_dir_train},
    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, shuffle=False,
    validation_split=VAL_SPLIT, callbacks=callbacks
)

pd.DataFrame(history.history).to_csv('v6_plus_minimal_tuning_v2_final_training_history.csv', index=False)

# Load best model
try:
    best_model = tf.keras.models.load_model(
        "best_v6_plus_minimal_tuning_v2_final_model.keras",
        custom_objects={"quantile_loss":quantile_loss,
                        "focal_weighted_binary_crossentropy":focal_weighted_binary_crossentropy},
        safe_mode=False
    )
    print("✅ Loaded best model.")
except Exception as e:
    print("⚠️ Could not load best model:", e)
    best_model = model

# =============================================================================
# 10) Calibrate direction probability per *ticker* + thresholds (MCC)
# =============================================================================
n_total = len(X_price_train)
n_val   = int(np.ceil(n_total * VAL_SPLIT))
val_slice = slice(n_total - n_val, n_total)

Xf_val = X_price_train[val_slice]
Xt_val = X_ticker_train[val_slice]
Xm_val = X_market_train[val_slice]
y_val_dir = y_dir_train[val_slice].astype(int).ravel()

p_raw_val = best_model.predict([Xf_val, Xt_val, Xm_val], verbose=0)[1].ravel()
tkr_val_last = Xt_val[:, -1]

def best_threshold_by_mcc(y_true, p_prob):
    ths = np.linspace(0.1, 0.9, 81)
    best_th, best_mcc = 0.5, -1.0
    for th in ths:
        yhat = (p_prob >= th).astype(int)
        mcc = matthews_corrcoef(y_true, yhat) if len(np.unique(yhat)) > 1 else -1.0
        if mcc > best_mcc:
            best_mcc, best_th = mcc, th
    return float(best_th), float(best_mcc)

calibrators = {}   # per-ticker
thresholds  = {}

for t in np.unique(tkr_val_last):
    idx = (tkr_val_last == t)
    n   = idx.sum()
    pos = int(y_val_dir[idx].sum())
    neg = int(n - pos)

    if n < 30 or pos < 5 or neg < 5:
        calibrators[int(t)] = None
        thresholds[str(int(t))] = 0.5
        continue

    p_in = np.clip(p_raw_val[idx], 0.05, 0.95)
    try:
        iso = IsotonicRegression(out_of_bounds='clip').fit(p_in, y_val_dir[idx])
        p_cal = iso.transform(p_in)
    except Exception:
        iso = None
        p_cal = p_in

    th, _ = best_threshold_by_mcc(y_val_dir[idx], p_cal)
    calibrators[int(t)] = iso
    thresholds[str(int(t))] = th

joblib.dump(calibrators, 'dir_calibrators_per_ticker.pkl')
with open('dir_thresholds_per_ticker.json','w') as f: json.dump(thresholds, f, indent=2)

# =============================================================================
# 11) MC Dropout helper
# =============================================================================
def predict_dir_with_mc(model, inputs, n=MC_DIR_SAMPLES):
    preds = []
    for _ in range(n):
        y = model(inputs, training=True)      # enable dropout
        p = y[1].numpy().ravel()              # (batch,)
        preds.append(p)
    p = np.stack(preds, axis=0)               # (n, batch)
    mean = np.mean(p, axis=0)
    std  = np.std(p, axis=0, ddof=0)
    return np.squeeze(mean), np.squeeze(std)

# =============================================================================
# 12) WFV with logging + gates
# =============================================================================
def walk_forward_validation_multi_task_batch(
    model,
    df,
    feature_columns,
    ticker_scalers,
    ticker_encoder,
    market_encoder,
    seq_length=SEQ_LEN,
    retrain_frequency=int(BEST_PARAMS['retrain_frequency']),
    chunk_size=int(BEST_PARAMS['chunk_size']),
    online_learning=True,
    use_mc_dropout=True,
    calibrators=None,
    thresholds=None,
    # gates
    conf_gate=CONF_GATE,
    unc_max=UNC_MAX,
    margin=MARGIN,
    allow_price_online=ALLOW_PRICE_ONLINE,
    # logging
    verbose=True,
    verbose_every=200,
    ticker_limit=None
):
    t0 = time.perf_counter()

    all_predictions = []
    chunk_metrics = []

    tickers = df['Ticker'].unique()
    if ticker_limit is not None:
        tickers = tickers[:int(ticker_limit)]

    if verbose:
        print(
            f"▶️ WFV start: tickers={len(tickers)}, "
            f"chunk_size={chunk_size}, seq_length={seq_length}, "
            f"online_learning={online_learning}, mc_dropout={use_mc_dropout}, "
            f"retrain_freq={retrain_frequency}", flush=True
        )

    for t_idx, ticker in enumerate(tickers, start=1):
        g = df[df['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)
        total_days = len(g)
        if total_days < chunk_size + seq_length:
            if verbose:
                print(f"⚠️  Skip {ticker}: rows={total_days} < {chunk_size + seq_length}", flush=True)
            continue

        num_chunks = total_days // chunk_size
        if (total_days % chunk_size) > seq_length:
            num_chunks += 1

        if verbose:
            print(
                f"\n🧩 [{t_idx}/{len(tickers)}] Ticker={ticker} | rows={total_days} | "
                f"chunks={num_chunks} (size={chunk_size})", flush=True
            )

        ticker_pred_count = 0

        for cidx in range(num_chunks):
            s = cidx * chunk_size
            e = min(s + chunk_size, total_days)
            if (e - s) < seq_length + 1:
                if verbose:
                    print(f"  ⚠️  chunk {cidx+1}/{num_chunks} too small: size={e-s}", flush=True)
                continue

            chunk = g.iloc[s:e].reset_index(drop=True)
            step_total = len(chunk) - seq_length

            if verbose:
                print(
                    f"  📦 Chunk {cidx+1}/{num_chunks} | rows={len(chunk)} | "
                    f"range: {chunk['Date'].min()} → {chunk['Date'].max()} | steps={step_total}",
                    flush=True
                )

            batch_Xf, batch_Xt, batch_Xm = [], [], []
            batch_yp, batch_yd = [], []
            batch_sw_price = []   # sample_weight price
            batch_sw_dir   = []   # sample_weight dir

            preds_in_chunk = []

            for i in range(step_total):
                hist = chunk.iloc[i : i + seq_length]
                targ = chunk.iloc[i + seq_length]

                t_id_last = int(hist['Ticker_ID'].iloc[-1])
                if t_id_last not in ticker_scalers:
                    continue

                fs = ticker_scalers[t_id_last]['feature_scaler']
                ps = ticker_scalers[t_id_last]['price_scaler']
                mode = ticker_scalers[t_id_last].get('price_target_mode', PRICE_TARGET_MODE)

                Xf = fs.transform(hist[feature_columns].values).reshape(1, seq_length, -1)
                Xt = hist['Ticker_ID'].values.reshape(1, seq_length)
                Xm = hist['Market_ID_enc'].values.reshape(1, seq_length)

                # ----- prediction -----
                outs = model.predict([Xf, Xt, Xm], verbose=0)
                price_scaled_pred = outs[0]  # scaled of target (price or logret)

                # (อัปเดต) clip ก่อน inverse transform กันหลุดสเกล
                price_scaled_pred = np.clip(price_scaled_pred, -6.0, 6.0)

                if use_mc_dropout:
                    p_mean, p_std = predict_dir_with_mc(model, [Xf, Xt, Xm], n=MC_DIR_SAMPLES)
                    p_dir = float(np.asarray(p_mean).reshape(-1)[0])
                    p_unc = float(np.asarray(p_std).reshape(-1)[0])
                else:
                    p_dir = float(outs[1].ravel()[0])
                    p_unc = 0.0

                last_close = float(hist['Close'].iloc[-1])

                # inverse transform price head → target_pred
                target_pred = float(ps.inverse_transform(price_scaled_pred)[0][0])
                if mode == 'logret':
                    price_pred = last_close * np.exp(target_pred)
                else:  # 'price'
                    price_pred = target_pred

                # per-ticker calibration & threshold
                if calibrators and (t_id_last in calibrators) and (calibrators[t_id_last] is not None):
                    p_cal = float(calibrators[t_id_last].transform([p_dir])[0])
                else:
                    p_cal = p_dir
                thr = float(thresholds.get(str(t_id_last), thresholds.get(t_id_last, 0.5))) if thresholds else 0.5
                pred_dir = int(p_cal >= thr)

                actual_price = float(targ['Close'])
                actual_dir = int(actual_price > last_close)

                preds_in_chunk.append({
                    'Ticker': ticker,
                    'Date': targ['Date'],
                    'Chunk_Index': cidx + 1,
                    'Position_in_Chunk': i + 1,
                    'Predicted_Price': price_pred,
                    'Actual_Price': actual_price,
                    'Predicted_Dir': pred_dir,
                    'Actual_Dir': actual_dir,
                    'Dir_Prob_Raw': float(p_dir),
                    'Dir_Prob_Cal': float(p_cal),
                    'Dir_Prob_Unc': float(p_unc),
                    'Last_Close': last_close,
                    'Price_Change_Actual': actual_price - last_close,
                    'Price_Change_Pred': price_pred - last_close,
                    'Ticker_ID': t_id_last
                })
                ticker_pred_count += 1

                # ----- periodic progress log -----
                if verbose and ( (i + 1) % max(1, int(verbose_every)) == 0 or (i == step_total - 1) ):
                    print(
                        f"    🔹 step {i+1:>5}/{step_total:<5} "
                        f"| pred_price={price_pred:.3f} "
                        f"| p_cal={p_cal:.3f} (thr={thr:.2f}) → dir={pred_dir} "
                        f"| unc={p_unc:.3f} "
                        f"| online={'Y' if online_learning else 'N'}",
                        flush=True
                    )

                # ----- optional online learning inside chunk -----
                if online_learning:
                    # gate สำหรับอัปเดตทิศทาง
                    ok_dir = True
                    if conf_gate:
                        conf = abs(p_cal - thr)
                        ok_dir = (conf >= margin) and (p_unc <= unc_max)

                    if ok_dir:
                        batch_Xf.append(Xf); batch_Xt.append(Xt); batch_Xm.append(Xm)
                        # direction label
                        batch_yd.append(np.array([actual_dir], float))
                        batch_sw_dir.append(1.0)

                        # price label (ตาม mode) + (อัปเดต) clip true_target ป้องกัน spike
                        if mode == 'logret':
                            true_target = np.log(actual_price / last_close)
                            true_target = float(np.clip(true_target, -0.25, 0.25))
                        else:
                            true_target = actual_price
                        batch_yp.append(ps.transform(np.array([[true_target]], float)))

                        # (อัปเดต) น้ำหนักหัวราคาเบา ๆ เมื่อผ่าน gate
                        if allow_price_online and ok_dir:
                            batch_sw_price.append(0.25)
                        else:
                            batch_sw_price.append(0.0)
                    # ไม่ผ่าน gate → ไม่เพิ่ม sample

                    do_retrain = ((i + 1) % retrain_frequency == 0) or (i == step_total - 1)
                    if do_retrain and len(batch_Xf) >= 5:
                        bf = np.concatenate(batch_Xf, axis=0)
                        bt = np.concatenate(batch_Xt, axis=0)
                        bm = np.concatenate(batch_Xm, axis=0)
                        bp = np.concatenate(batch_yp, axis=0)
                        bd = np.concatenate(batch_yd, axis=0)
                        sw_p = np.array(batch_sw_price, float)
                        sw_d = np.array(batch_sw_dir, float)

                        model.fit(
                            [bf, bt, bm],
                            {'price_output': bp, 'direction_output': bd},
                            sample_weight={'price_output': sw_p, 'direction_output': sw_d},
                            epochs=1, batch_size=len(bf), verbose=0, shuffle=False
                        )
                        if verbose:
                            print(f"    🔄 mini-retrain @ step {i+1} (batch={len(bf)})", flush=True)

                        batch_Xf, batch_Xt, batch_Xm = [], [], []
                        batch_yp, batch_yd = [], []
                        batch_sw_price, batch_sw_dir = [], []

            # ----- end of chunk: metrics & log -----
            if preds_in_chunk:
                cdf = pd.DataFrame(preds_in_chunk)
                a_p, p_p = cdf['Actual_Price'].values, cdf['Predicted_Price'].values
                a_d, p_d = cdf['Actual_Dir'].values, cdf['Predicted_Dir'].values

                mae = mean_absolute_error(a_p, p_p)
                rmse = np.sqrt(mean_squared_error(a_p, p_p))
                r2 = r2_score(a_p, p_p)
                acc = accuracy_score(a_d, p_d)
                f1 = f1_score(a_d, p_d)
                mcc = matthews_corrcoef(a_d, p_d) if len(np.unique(p_d)) > 1 else 0.0

                chunk_metrics.append({
                    'Ticker': ticker, 'Chunk_Index': cidx + 1,
                    'Chunk_Start_Date': cdf['Date'].min(), 'Chunk_End_Date': cdf['Date'].max(),
                    'Predictions_Count': len(cdf),
                    'MAE': mae, 'RMSE': rmse, 'R2_Score': r2,
                    'Direction_Accuracy': acc, 'Direction_F1': f1, 'Direction_MCC': mcc
                })
                if verbose:
                    print(
                        f"  ✅ Chunk {cidx+1}/{num_chunks} done | preds={len(cdf)} "
                        f"| MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f} "
                        f"| ACC={acc:.3f} F1={f1:.3f} MCC={mcc:.3f}",
                        flush=True
                    )
                all_predictions.extend(preds_in_chunk)

        if verbose:
            print(f"🟩 Ticker {ticker} completed | total_preds={ticker_pred_count}", flush=True)

    # ----- finish -----
    if not all_predictions:
        if verbose:
            print("❌ No predictions generated!", flush=True)
        return pd.DataFrame(), {}

    pred_df = pd.DataFrame(all_predictions)
    pred_df.to_csv('predictions_chunk_walkforward.csv', index=False)

    overall = {}
    for tkr, g in pred_df.groupby('Ticker'):
        a_p, p_p = g['Actual_Price'].values, g['Predicted_Price'].values
        a_d, p_d = g['Actual_Dir'].values, g['Predicted_Dir'].values

        mae = mean_absolute_error(a_p, p_p)
        rmse = np.sqrt(mean_squared_error(a_p, p_p))
        r2 = r2_score(a_p, p_p)
        acc = accuracy_score(a_d, p_d)
        f1 = f1_score(a_d, p_d)
        prec = precision_score(a_d, p_d)
        rec = recall_score(a_d, p_d)

        overall[tkr] = {
            'Total_Predictions': len(g),
            'MAE': mae, 'RMSE': rmse, 'R2_Score': r2,
            'Direction_Accuracy': acc, 'Direction_F1_Score': f1,
            'Direction_Precision': prec, 'Direction_Recall': rec
        }

    pd.DataFrame.from_dict(overall, orient='index').to_csv('overall_metrics_per_ticker.csv')
    pd.DataFrame(chunk_metrics).to_csv('chunk_metrics.csv', index=False)

    # console summary (CSV-ready)
    lines = ["===== PER-TICKER SUMMARY (CSV-ready) =====",
             ",Total_Predictions,MAE,RMSE,R2_Score,Direction_Accuracy,Direction_F1_Score,Direction_Precision,Direction_Recall"]
    for tkr, m in overall.items():
        lines.append(f"{tkr},{m['Total_Predictions']},{m['MAE']},{m['RMSE']},{m['R2_Score']},"
                     f"{m['Direction_Accuracy']},{m['Direction_F1_Score']},{m['Direction_Precision']},{m['Direction_Recall']}")
    textsum = "\n".join(lines)
    print("\n" + textsum)
    with open('per_ticker_console_summary.txt','w', encoding='utf-8') as f:
        f.write(textsum + "\n")
    print("\n📝 saved per-ticker console report → per_ticker_console_summary.txt")

    dt = time.perf_counter() - t0
    if verbose:
        print(
            f"\n🏁 WFV done | total_preds={len(pred_df)} | chunks={len(chunk_metrics)} "
            f"| elapsed={dt:.1f}s", flush=True
        )

    return pred_df, overall

# =============================================================================
# 13) Final WFV run with best params (+ verbose logs, per-ticker calib, gates)
# =============================================================================
predictions_df, results_per_ticker = walk_forward_validation_multi_task_batch(
    model=best_model,
    df=test_df,
    feature_columns=feature_columns,
    ticker_scalers=ticker_scalers,
    ticker_encoder=ticker_encoder,
    market_encoder=market_encoder,
    seq_length=int(BEST_PARAMS['seq_length']),
    retrain_frequency=int(BEST_PARAMS['retrain_frequency']),
    chunk_size=int(BEST_PARAMS['chunk_size']),
    online_learning=True,        # online learning เปิด แต่มี gate
    use_mc_dropout=True,
    calibrators=calibrators,     # per-ticker
    thresholds=thresholds,       # per-ticker
    conf_gate=CONF_GATE,
    unc_max=UNC_MAX,
    margin=MARGIN,
    allow_price_online=ALLOW_PRICE_ONLINE,
    verbose=True,
    verbose_every=200,
    ticker_limit=None
)

# บันทึก summary files
if results_per_ticker:
    pd.DataFrame.from_dict(results_per_ticker, orient='index').to_csv('metrics_per_ticker_multi_task.csv', index=True)
if predictions_df is not None and len(predictions_df):
    pred_flat = predictions_df[['Ticker','Date','Actual_Price','Predicted_Price','Actual_Dir',
                                'Predicted_Dir','Dir_Prob_Cal','Dir_Prob_Unc']]
    pred_flat.to_csv('all_predictions_per_day_multi_task.csv', index=False)

# =============================================================================
# 14) Production artifacts & config
# =============================================================================
production_config = {
    'model_config': {
        'seq_length': int(BEST_PARAMS['seq_length']),
        'chunk_size': int(BEST_PARAMS['chunk_size']),
        'retrain_frequency': int(BEST_PARAMS['retrain_frequency']),
        'embedding_dim': int(BEST_PARAMS['embedding_dim']),
        'LSTM_units_1': int(BEST_PARAMS['LSTM_units_1']),
        'LSTM_units_2': int(BEST_PARAMS['LSTM_units_2']),
        'dropout_rate': float(BEST_PARAMS['dropout_rate']),
        'dense_units': int(BEST_PARAMS['dense_units']),
        'learning_rate': float(BEST_PARAMS['learning_rate'])
    },
    'training_config': {
        'batch_size': 33,
        'validation_split': 0.12,
        'expected_epochs': 105,
        'early_stopping_patience': 20
    },
    'inference_config': {
        'mc_dir_samples': MC_DIR_SAMPLES,
        'price_target_mode': PRICE_TARGET_MODE
    }
}
with open('production_model_config.json','w') as f: json.dump(production_config, f, indent=2)

# Save model template with best params (โครง+weights)
best_model.save('best_hypertuned_model.keras')

# =============================================================================
# 15) Serve function (real-time inference for one ticker)
# =============================================================================
def serve_one(df_latest_1ticker, artifacts, seq_length=int(BEST_PARAMS['seq_length']), use_mc=True):
    """
    df_latest_1ticker: DataFrame สำหรับหุ้นเดียว (มี feature_columns ครบและเรียงตาม Date)
                       ต้องมีอย่างน้อย seq_length แถว
    artifacts: {
      'model', 'ticker_scalers', 'ticker_encoder', 'market_encoder',
      'feature_columns', 'calibrators', 'thresholds', 'mc_dir_samples', 'price_target_mode'
    }
    """
    g = df_latest_1ticker.sort_values('Date').tail(seq_length)
    ticker = g['Ticker'].iloc[-1]
    market = g['Market_ID'].iloc[-1]
    t_id = int(artifacts['ticker_encoder'].transform([ticker])[0])
    m_id = int(artifacts['market_encoder'].transform([market])[0])

    fs = artifacts['ticker_scalers'][t_id]['feature_scaler']
    ps = artifacts['ticker_scalers'][t_id]['price_scaler']
    mode = artifacts.get('price_target_mode', PRICE_TARGET_MODE)

    Xf = fs.transform(g[artifacts['feature_columns']].values).reshape(1, seq_length, -1)
    Xt = np.full((1, seq_length), t_id)
    Xm = np.full((1, seq_length), m_id)

    outs = artifacts['model'].predict([Xf, Xt, Xm], verbose=0)
    price_target_pred = float(ps.inverse_transform(outs[0])[0][0])
    last_close = float(g['Close'].iloc[-1])

    if mode == 'logret':
        pred_price = last_close * np.exp(price_target_pred)
    else:
        pred_price = price_target_pred

    if use_mc:
        mean_p, std_p = predict_dir_with_mc(artifacts['model'], [Xf, Xt, Xm], n=artifacts.get('mc_dir_samples', 8))
        p_raw = float(np.asarray(mean_p).reshape(-1)[0])
        p_unc = float(np.asarray(std_p).reshape(-1)[0])
    else:
        p_raw = float(outs[1].ravel()[0])
        p_unc = 0.0

    cal = artifacts['calibrators'].get(t_id, None)
    if cal is not None:
        p_cal = float(cal.transform([p_raw])[0])
    else:
        p_cal = p_raw

    thr = float(artifacts['thresholds'].get(str(t_id), artifacts['thresholds'].get(t_id, 0.5)))
    direction = int(p_cal >= thr)

    return {
        'ticker': ticker,
        'ticker_id': int(t_id),
        'pred_price': float(pred_price),
        'dir_prob_raw': float(p_raw),
        'dir_prob_cal': float(p_cal),
        'dir_prob_unc': float(p_unc),
        'decision': int(direction),
        'threshold_used': float(thr)
    }

# Bundle artifacts for serving
artifacts = {
    'model': best_model,
    'ticker_scalers': ticker_scalers,
    'ticker_encoder': ticker_encoder,
    'market_encoder': market_encoder,
    'feature_columns': feature_columns,
    'calibrators': {int(k):v for k,v in calibrators.items()},
    'thresholds': thresholds,          # keys are str(int(ticker_id))
    'mc_dir_samples': MC_DIR_SAMPLES,
    'price_target_mode': PRICE_TARGET_MODE
}
joblib.dump(artifacts, 'serving_artifacts.pkl')

print("\n✅ All done. Files saved:")
print(" - best_v6_plus_minimal_tuning_v2_final_model.keras")
print(" - best_hypertuned_model.keras")
print(" - v6_plus_minimal_tuning_v2_final_training_history.csv")
print(" - market_encoder.pkl, ticker_encoder.pkl, ticker_scalers.pkl, feature_columns.pkl")
print(" - dir_calibrators_per_ticker.pkl, dir_thresholds_per_ticker.json")
print(" - predictions_chunk_walkforward.csv, chunk_metrics.csv, overall_metrics_per_ticker.csv")
print(" - metrics_per_ticker_multi_task.csv, all_predictions_per_day_multi_task.csv")
print(" - per_ticker_console_summary.txt")
print(" - production_model_config.json, serving_artifacts.pkl")
