# -*- coding: utf-8 -*-
"""
Production Probabilistic Regression (1-head μ,σ) + Direction Boost
- EPS per ticker (grid search) + isotonic calibration + per-ticker threshold
- EMA smoothing บน prob หลังคาลิเบรต
- Adaptive threshold ภายในแต่ละ chunk เมื่อ pred bias เกินกรอบ
- Memory-light WFV: stream-to-CSV, ไม่สะสม DataFrame ใหญ่
"""

# =============================================================================
# 0) Imports & Global Config
# =============================================================================
import os, random, json, joblib, warnings, sys, time, math, csv, gc
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             matthews_corrcoef, confusion_matrix)
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
    'learning_rate': 1.7e-4,
    'retrain_frequency': 9,
    'seq_length': 10
}

# ---------- Memory-light ----------
MEMORY_LIGHT_WFV = True
MC_DIR_SAMPLES_BASE = 8
MC_DIR_SAMPLES = 4 if MEMORY_LIGHT_WFV else MC_DIR_SAMPLES_BASE

# ---------- Online learning gates ----------
CONF_GATE = True
UNC_MAX   = 0.20      # ผ่อนคลายขึ้นเล็กน้อย
MARGIN    = 0.05
ALLOW_PRICE_ONLINE = True

# ---------- EPS (log-return margin) ----------
# ค่า default ถ้าหา per-ticker ไม่ได้
EPS_RET_DEFAULT = 0.002  # 0.2%

# grid สำหรับหา eps ต่อ ticker (หน่วย log-return)
EPS_GRID = np.round(np.arange(0.0, 0.0065, 0.0005), 4)  # 0.00% → 0.65% step 0.05%

# ---------- Threshold search ----------
THRESH_METRIC = 'mcc'   # 'mcc'|'f1'|'bal'
THRESH_MIN    = 0.40

# ---------- Eval options ----------
EVAL_RETHRESH_BALANCED = True
INDIFF_BAND_FOR_EVAL   = 0.0

# ---------- Direction boosters ----------
EMA_ALPHA = 0.30       # smoothing บน prob หลังคาลิเบรต
ADAPTIVE_THR = True
ADAPT_MIN_STEPS = 30
ADAPT_LOW  = 0.25      # ถ้าอัตราทำนายเป็นคลาส 1 < 25% หรือ > 75% → ปรับ
ADAPT_HIGH = 0.75
ADAPT_POS_TARGET = 0.50  # พยายามบาลานซ์สู่ 50/50

# ---------- Paths ----------
STREAM_PRED_PATH     = 'predictions_chunk_walkforward.csv'
STREAM_CHUNK_PATH    = 'chunk_metrics.csv'
STREAM_OVERALL_PATH  = 'overall_metrics_per_ticker.csv'
EPS_PER_TICKER_PATH  = 'eps_ret_per_ticker.json'
CALIBRATORS_PATH     = 'dir_calibrators_per_ticker.pkl'
THRESHOLDS_PATH      = 'dir_thresholds_per_ticker.json'

# =============================================================================
# 2) Losses / Utils
# =============================================================================
@tf.keras.utils.register_keras_serializable()
def gaussian_nll(y_true, y_pred):
    """
    y_pred[...,0] = mu_scaled, y_pred[...,1] = log_sigma_scaled
    y_true = target (scaled ด้วย RobustScaler ของ ticker)
    """
    y_true = tf.cast(y_true, tf.float32)
    mu_s, log_sigma_s = tf.split(y_pred, 2, axis=-1)
    sigma_s = tf.nn.softplus(log_sigma_s) + 1e-6
    z = (y_true - mu_s) / sigma_s
    return tf.reduce_mean(0.5*tf.math.log(2.0*np.pi) + tf.math.log(sigma_s) + 0.5*tf.square(z))

@tf.keras.utils.register_keras_serializable()
def mae_on_mu(y_true, y_pred):
    mu_s, _ = tf.split(y_pred, 2, axis=-1)
    return tf.reduce_mean(tf.abs(tf.cast(y_true, tf.float32) - mu_s))

def sanitize(arr):
    arr = np.asarray(arr, dtype=np.float32)
    mask = np.isfinite(arr)
    if not mask.all():
        median = np.nanmedian(arr[mask])
        arr[~mask] = median
    arr[np.isnan(arr)] = np.nanmedian(arr[np.isfinite(arr)])
    return arr.astype(np.float32, copy=False)

def match_len_vec(w, n):
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    if w.size == 0:
        return np.ones((n,), dtype=np.float32)
    if w.size == n:
        return w
    reps = int(math.ceil(n / float(w.size)))
    return np.tile(w, reps)[:n].astype(np.float32, copy=False)

def softplus_np(x):
    return np.log1p(np.exp(x))

def mu_sigma_to_raw(mu_scaled, log_sigma_scaled, ps):
    """แปลง μ,σ จากสเกล RobustScaler → log-return จริง"""
    sigma_scaled = softplus_np(log_sigma_scaled) + 1e-6
    scale  = getattr(ps, 'scale_',  np.array([1.0], dtype=np.float32))[0]
    center = getattr(ps, 'center_', np.array([0.0], dtype=np.float32))[0]
    mu_raw = mu_scaled * scale + center
    sigma_raw = sigma_scaled * scale
    return float(mu_raw), float(sigma_raw)

def norm_cdf(x):
    return 0.5*(1.0 + math.erf(x / math.sqrt(2.0)))

def best_threshold(y_true, p_prob, metric=THRESH_METRIC):
    ths = np.linspace(0.10, 0.90, 81)
    best_th, best_val = 0.5, -1.0
    for th in ths:
        yhat = (p_prob >= th).astype(int)
        if len(np.unique(yhat)) < 2:
            continue
        if metric == 'mcc':
            val = matthews_corrcoef(y_true, yhat)
        elif metric == 'f1':
            val = f1_score(y_true, yhat)
        elif metric == 'bal':
            val = 0.5*matthews_corrcoef(y_true, yhat) + 0.5*f1_score(y_true, yhat)
        else:
            val = f1_score(y_true, yhat)
        if val > best_val:
            best_val, best_th = val, th
    return float(max(THRESH_MIN, best_th)), float(best_val)

# =============================================================================
# 3) Load Data
# =============================================================================
DATA_PATH = '../Preproces/data/Stock/merged_stock_sentiment_financial.csv'
df = pd.read_csv(DATA_PATH)

# =============================================================================
# 4) Clean & Feature Engineering (no leakage)
# =============================================================================
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

df['Sentiment'] = df['Sentiment'].map({'Positive':1, 'Negative':-1, 'Neutral':0}).fillna(0).astype(np.int8)
df['Change'] = df['Close'] - df['Open']
df['Change (%)'] = df.groupby('Ticker')['Close'].pct_change() * 100.0
upper = df['Change (%)'].quantile(0.99); lower = df['Change (%)'].quantile(0.01)
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
    rsi = ta.momentum.RSIIndicator(close=g['Close'], window=14); g['RSI'] = rsi.rsi()
    g['RSI'] = g['RSI'].fillna(g['RSI'].rolling(window=5, min_periods=1).mean())
    g['MACD'] = g['EMA_12'] - g['EMA_26']; g['MACD_Signal'] = g['MACD'].rolling(9, min_periods=1).mean()
    bb = ta.volatility.BollingerBands(close=g['Close'], window=20, window_dev=2)
    g['Bollinger_High'] = bb.bollinger_hband(); g['Bollinger_Low']  = bb.bollinger_lband()
    atr = ta.volatility.AverageTrueRange(high=g['High'], low=g['Low'], close=g['Close'], window=14)
    g['ATR'] = atr.average_true_range()
    kc = ta.volatility.KeltnerChannel(high=g['High'], low=g['Low'], close=g['Close'], window=20, window_atr=10)
    g['Keltner_High']   = kc.keltner_channel_hband()
    g['Keltner_Low']    = kc.keltner_channel_lband()
    g['Keltner_Middle'] = kc.keltner_channel_mband()
    g['High_Low_Diff'] = g['High'] - g['Low']
    g['High_Low_EMA']  = g['High_Low_Diff'].ewm(span=10, adjust=False).mean()
    g['Chaikin_Vol']   = g['High_Low_EMA'].pct_change(10) * 100.0
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

# Financials ffill
financial_columns = [
    'Total Revenue','QoQ Growth (%)','Earnings Per Share (EPS)','ROE (%)',
    'Net Profit Margin (%)','Debt to Equity','P/E Ratio','P/BV Ratio','Dividend Yield (%)'
]
for c in financial_columns:
    if c not in df.columns: df[c] = np.nan
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

df[feature_columns] = (df.groupby('Ticker')[feature_columns]
                         .apply(lambda g: g.fillna(method='ffill'))
                         .reset_index(level=0, drop=True))
df[feature_columns] = df[feature_columns].fillna(0.0)

# Targets per-ticker
df['TargetPrice'] = df.groupby('Ticker')['Close'].shift(-1)
df = df.dropna(subset=['TargetPrice']).reset_index(drop=True)

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
# 6) Per-ticker Robust Scaler (features + log-return target)
# =============================================================================
SEQ_LEN = int(BEST_PARAMS['seq_length'])

train_df['PriceTargetRaw'] = np.log(train_df['TargetPrice'] / train_df['Close']).astype(np.float32)
test_df['PriceTargetRaw']  = np.log(test_df['TargetPrice']  / test_df['Close']).astype(np.float32)

train_features = sanitize(train_df[feature_columns].values.astype(np.float32))
test_features  = sanitize(test_df[feature_columns].values.astype(np.float32))
train_price_t  = sanitize(train_df['PriceTargetRaw'].values.reshape(-1,1).astype(np.float32))
test_price_t   = sanitize(test_df['PriceTargetRaw'].values.reshape(-1,1).astype(np.float32))

train_ticker_id = train_df['Ticker_ID'].values
test_ticker_id  = test_df['Ticker_ID'].values

train_features_scaled = np.zeros_like(train_features, dtype=np.float32)
test_features_scaled  = np.zeros_like(test_features,  dtype=np.float32)
train_price_scaled    = np.zeros_like(train_price_t,  dtype=np.float32)
test_price_scaled     = np.zeros_like(test_price_t,   dtype=np.float32)

ticker_scalers = {}
id2ticker = {}
for t_id in np.unique(train_ticker_id):
    gmask = (train_ticker_id == t_id)
    X_part = train_features[gmask]
    y_part = train_price_t[gmask]
    fs = RobustScaler(); ps = RobustScaler()
    Xs = fs.fit_transform(X_part).astype(np.float32)
    ys = ps.fit_transform(y_part).astype(np.float32)
    train_features_scaled[gmask] = Xs
    train_price_scaled[gmask]    = ys
    ticker_name = train_df.loc[gmask, 'Ticker'].iloc[0]
    id2ticker[t_id] = ticker_name
    ticker_scalers[t_id] = {'feature_scaler': fs, 'price_scaler': ps, 'ticker': ticker_name}
    del X_part, y_part, Xs, ys; gc.collect()

for t_id in np.unique(test_ticker_id):
    if t_id not in ticker_scalers:
        continue
    gmask = (test_ticker_id == t_id)
    fs = ticker_scalers[t_id]['feature_scaler']
    ps = ticker_scalers[t_id]['price_scaler']
    test_features_scaled[gmask] = fs.transform(test_features[gmask]).astype(np.float32)
    test_price_scaled[gmask]    = ps.transform(test_price_t[gmask]).astype(np.float32)

joblib.dump(ticker_scalers, 'ticker_scalers.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')

# =============================================================================
# 7) Build sequences per ticker (memory-aware)
# =============================================================================
def create_sequences_for_ticker(features, ticker_ids, market_ids, targets_price, seq_length=SEQ_LEN):
    X_features, X_tickers, X_markets, Y_price = [], [], [], []
    for i in range(len(features) - seq_length):
        X_features.append(features[i:i+seq_length])
        X_tickers.append(ticker_ids[i:i+seq_length])
        X_markets.append(market_ids[i:i+seq_length])
        Y_price.append(targets_price[i+seq_length])
    return (np.array(X_features, dtype=np.float32),
            np.array(X_tickers, dtype=np.int32),
            np.array(X_markets, dtype=np.int32),
            np.array(Y_price, dtype=np.float32))

def build_dataset_sequences(base_df, features_scaled, price_scaled, seq_length=SEQ_LEN):
    Xf_list, Xt_list, Xm_list, Yp_list = [], [], [], []
    for t_id in range(num_tickers):
        idx = base_df.index[base_df['Ticker_ID']==t_id].tolist()
        if len(idx) <= seq_length:
            continue
        mask = np.isin(base_df.index, idx)
        f = features_scaled[mask]
        p = price_scaled[mask]
        t = base_df.loc[mask, 'Ticker_ID'].values.astype(np.int32)
        m = base_df.loc[mask, 'Market_ID_enc'].values.astype(np.int32)
        Xf, Xt, Xm, Yp = create_sequences_for_ticker(f, t, m, p, seq_length)
        if len(Xf):
            Xf_list.append(Xf); Xt_list.append(Xt); Xm_list.append(Xm); Yp_list.append(Yp)
        del f, p, t, m, Xf, Xt, Xm, Yp; gc.collect()
    if len(Xf_list)==0:
        return (np.zeros((0,seq_length,len(feature_columns)), dtype=np.float32),)*4
    Xf = np.concatenate(Xf_list, axis=0)
    Xt = np.concatenate(Xt_list, axis=0)
    Xm = np.concatenate(Xm_list, axis=0)
    Yp = np.concatenate(Yp_list, axis=0)
    del Xf_list, Xt_list, Xm_list, Yp_list; gc.collect()
    return Xf, Xt, Xm, Yp

X_price_train, X_ticker_train, X_market_train, y_price_train = build_dataset_sequences(
    train_df, train_features_scaled, train_price_scaled, SEQ_LEN
)
X_price_test, X_ticker_test, X_market_test, y_price_test = build_dataset_sequences(
    test_df, test_features_scaled, test_price_scaled, SEQ_LEN
)

print("Train shapes:",
      X_price_train.shape, X_ticker_train.shape, X_market_train.shape,
      y_price_train.shape)

num_feature = len(feature_columns)

# =============================================================================
# 8) Model (1-head μ,σ)
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
price_params = Dense(2, name="price_params")(price_head)

model = Model(inputs=[features_input, ticker_input, market_input], outputs=[price_params])

# Optimizer & compile (CosineDecay)
BATCH_SIZE = 33
VAL_SPLIT  = 0.12
EPOCHS     = 105
steps_per_epoch = max(1, int(len(X_price_train)*(1-VAL_SPLIT)) // BATCH_SIZE)
decay_steps = max(1, int(steps_per_epoch * EPOCHS * 1.4))

lr_schedule = CosineDecay(initial_learning_rate=float(BEST_PARAMS['learning_rate']),
                          decay_steps=decay_steps, alpha=9e-6)
optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1.4e-5, clipnorm=0.95)

model.compile(optimizer=optimizer, loss={"price_params": gaussian_nll},
              metrics={"price_params":[mae_on_mu]})

print("\nModel Summary:")
model.summary(); print("Total params:", model.count_params())

# =============================================================================
# 9) Callbacks & Train
# =============================================================================
class StabilityCallback(Callback):
    def __init__(self): super().__init__(); self.best=float('inf')
    def on_epoch_end(self, epoch, logs=None):
        val_mae = logs.get('val_mae_on_mu', logs.get('val_price_params_mae_on_mu', 0))
        score = val_mae
        if score < self.best:
            self.best = score
            print(f"  🎯 New best validation MAE(μ): {score:.4f}")

early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True,
                               verbose=1, min_delta=1.2e-4, start_from_epoch=12)
checkpoint = ModelCheckpoint("best_v6_plus_minimal_tuning_v2_final_model.keras",
                             monitor="val_loss", save_best_only=True, mode="min", verbose=1)
csv_logger = CSVLogger('v6_plus_minimal_tuning_v2_final_training_log.csv')
callbacks = [early_stopping, checkpoint, csv_logger, StabilityCallback()]

history = model.fit(
    [X_price_train, X_ticker_train, X_market_train],
    {"price_params": y_price_train},
    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, shuffle=False,
    validation_split=VAL_SPLIT, callbacks=callbacks
)

pd.DataFrame(history.history).to_csv('v6_plus_minimal_tuning_v2_final_training_history.csv', index=False)

# Load best model
try:
    best_model = tf.keras.models.load_model(
        "best_v6_plus_minimal_tuning_v2_final_model.keras",
        custom_objects={"gaussian_nll":gaussian_nll, "mae_on_mu":mae_on_mu},
        safe_mode=False
    )
    print("✅ Loaded best model.")
except Exception as e:
    print("⚠️ Could not load best model:", e)
    best_model = model

# =============================================================================
# 10) Calibrate per-ticker: EPS grid → pick best → isotonic + threshold
# =============================================================================
n_total = len(X_price_train)
n_val   = int(np.ceil(n_total * VAL_SPLIT))
val_slice = slice(n_total - n_val, n_total)

Xf_val = X_price_train[val_slice]
Xt_val = X_ticker_train[val_slice]
Xm_val = X_market_train[val_slice]
y_true_scaled_val = y_price_train[val_slice].reshape(-1, 1)

# พยากรณ์บน validation
y_pred_val = best_model.predict([Xf_val, Xt_val, Xm_val], verbose=0)  # (n_val,2)
tkr_val_last = Xt_val[:, -1].astype(int)

# แปลง μ,σ เป็น "raw" และ label จริงทิศทางจาก logret > 0
mu_raw_val = np.zeros((len(Xf_val),), np.float32)
sigma_raw_val = np.zeros((len(Xf_val),), np.float32)
y_dir_true_val = np.zeros((len(Xf_val),), np.int8)

for i in range(len(Xf_val)):
    t_id = int(tkr_val_last[i])
    ps = ticker_scalers[t_id]['price_scaler']
    mu_s, log_sigma_s = float(y_pred_val[i,0]), float(y_pred_val[i,1])
    mu_raw, sigma_raw = mu_sigma_to_raw(mu_s, log_sigma_s, ps)
    mu_raw_val[i] = mu_raw; sigma_raw_val[i] = max(1e-8, sigma_raw)

    y_true_raw = float(ps.inverse_transform(y_true_scaled_val[i:i+1])[0,0])
    y_dir_true_val[i] = 1 if y_true_raw > 0.0 else 0

def prob_up_from_mu_sigma(mu_raw, sigma_raw, eps):
    if sigma_raw <= 1e-8:
        return 1.0 if (mu_raw - eps) > 0.0 else 0.0
    z = (mu_raw - eps)/sigma_raw
    return norm_cdf(z)

eps_per_ticker = {}
calibrators = {}
thresholds  = {}

for t in np.unique(tkr_val_last):
    idx = (tkr_val_last == t)
    n   = int(idx.sum())
    y_true = y_dir_true_val[idx]
    if n < 30 or len(np.unique(y_true)) < 2:
        eps_per_ticker[str(int(t))] = EPS_RET_DEFAULT
        calibrators[int(t)] = None
        thresholds[str(int(t))] = 0.5
        continue

    mu_t = mu_raw_val[idx]; sig_t = sigma_raw_val[idx]

    # 1) เลือก eps ที่ดีที่สุด (จาก prob แบบ raw)
    best_eps, best_score, best_p = EPS_RET_DEFAULT, -1.0, None
    for eps in EPS_GRID:
        p_raw = np.array([prob_up_from_mu_sigma(mu_t[i], sig_t[i], eps) for i in range(n)], dtype=np.float32)
        th_tmp, val_tmp = best_threshold(y_true, p_raw)
        if val_tmp > best_score:
            best_score = val_tmp; best_eps = float(eps); best_p = p_raw

    eps_per_ticker[str(int(t))] = best_eps

    # 2) fit isotonic บน p(best_eps) → calibrator
    try:
        iso = IsotonicRegression(out_of_bounds='clip').fit(best_p, y_true)
        p_cal = iso.transform(best_p)
    except Exception:
        iso = None; p_cal = best_p

    # 3) หา threshold บน prob ที่ calibrate แล้ว
    th, _ = best_threshold(y_true, p_cal)
    thresholds[str(int(t))] = float(th)
    calibrators[int(t)] = iso

# save artifacts
with open(EPS_PER_TICKER_PATH, 'w') as f: json.dump(eps_per_ticker, f, indent=2)
joblib.dump(calibrators, CALIBRATORS_PATH)
with open(THRESHOLDS_PATH, 'w') as f: json.dump(thresholds, f, indent=2)

# =============================================================================
# 11) MC helper (P(UP) mean/std กับ eps เฉพาะ ticker)
# =============================================================================
def predict_pup_with_mc(model, Xf, Xt, Xm, price_scaler, eps, n=MC_DIR_SAMPLES):
    pups = []
    for _ in range(n):
        y = model([Xf, Xt, Xm], training=True).numpy()  # (1,2)
        mu_s = float(y[0,0]); log_sigma_s = float(y[0,1])
        mu_raw, sigma_raw = mu_sigma_to_raw(mu_s, log_sigma_s, price_scaler)
        pups.append(prob_up_from_mu_sigma(mu_raw, sigma_raw, eps))
    pups = np.asarray(pups, dtype=np.float32)
    return float(np.mean(pups)), float(np.std(pups, ddof=0))

# =============================================================================
# 12) WFV streaming (EMA + Adaptive Thr)
# =============================================================================
def walk_forward_validation_prob_batch(
    model,
    df,
    feature_columns,
    ticker_scalers,
    ticker_encoder,
    market_encoder,
    seq_length=int(BEST_PARAMS['seq_length']),
    retrain_frequency=int(BEST_PARAMS['retrain_frequency']),
    chunk_size=int(BEST_PARAMS['chunk_size']),
    online_learning=True,
    use_mc_dropout=True,
    calibrators=None,
    thresholds=None,
    eps_map=None,
    # gates
    conf_gate=CONF_GATE,
    unc_max=UNC_MAX,
    margin=MARGIN,
    allow_price_online=ALLOW_PRICE_ONLINE,
    # logging
    verbose=True,
    verbose_every=200,
    ticker_limit=None,
    # streaming
    stream_preds_path=STREAM_PRED_PATH,
    stream_chunk_path=STREAM_CHUNK_PATH,
    stream_overall_path=STREAM_OVERALL_PATH
):
    t0 = time.perf_counter()

    # prepare CSVs
    with open(stream_preds_path, 'w', newline='', encoding='utf-8') as fpred:
        writer_pred = csv.writer(fpred)
        writer_pred.writerow([
            'Ticker','Date','Chunk_Index','Position_in_Chunk',
            'Predicted_Price','Actual_Price','Predicted_Dir','Actual_Dir',
            'Dir_Prob_Cal_Raw','Dir_Prob_Cal','Dir_Prob_Unc',
            'Thr_Used','Thr_Base','Thr_Adaptive','Eps_Used',
            'Last_Close','Price_Change_Actual','Price_Change_Pred'
        ])

    with open(stream_chunk_path, 'w', newline='', encoding='utf-8') as fchunk:
        writer_chunk = csv.writer(fchunk)
        writer_chunk.writerow([
            'Ticker','Chunk_Index','Chunk_Start_Date','Chunk_End_Date','Predictions_Count',
            'MAE','RMSE','R2_Score','Direction_Accuracy','Direction_F1','Direction_MCC'
        ])

    overall_accum = {}
    tickers = df['Ticker'].unique()
    if ticker_limit is not None:
        tickers = tickers[:int(ticker_limit)]

    if verbose:
        print(
            f"▶️ WFV start: tickers={len(tickers)}, chunk_size={chunk_size}, seq_length={seq_length}, "
            f"online_learning={online_learning}, mc_dropout={use_mc_dropout} (MC={MC_DIR_SAMPLES}), "
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
            print(f"\n🧩 [{t_idx}/{len(tickers)}] Ticker={ticker} | rows={total_days} | chunks={num_chunks} (size={chunk_size})",
                  flush=True)

        ticker_pred_count = 0

        for cidx in range(num_chunks):
            s = cidx * chunk_size
            e = min(s + chunk_size, total_days)
            if (e - s) < seq_length + 1:
                if verbose: print(f"  ⚠️  chunk {cidx+1}/{num_chunks} too small: size={e-s}", flush=True)
                continue

            chunk = g.iloc[s:e].reset_index(drop=True)
            step_total = len(chunk) - seq_length
            if verbose:
                print(f"  📦 Chunk {cidx+1}/{num_chunks} | rows={len(chunk)} | "
                      f"range: {chunk['Date'].min()} → {chunk['Date'].max()} | steps={step_total}", flush=True)

            fpred = open(stream_preds_path, 'a', newline='', encoding='utf-8')
            writer_pred = csv.writer(fpred)

            # mini-batch online update
            batch_Xf, batch_Xt, batch_Xm = [], [], []
            batch_yp, batch_sw_price = [], []

            # metrics arrays (per chunk)
            pred_dir_list, actual_dir_list = [], []
            p_cal_raw_list, p_cal_smooth_list = [], []
            actual_price_list, pred_price_list, last_close_list = [], [], []
            thr_used_list, thr_base_list, thr_adapt_list = [], [], []

            # EMA state per ticker (reset at each chunk)
            ema_prev = 0.5

            # buffers for adaptive-threshold
            p_hist_for_adapt = []

            for i in range(step_total):
                hist = chunk.iloc[i : i + seq_length]
                targ = chunk.iloc[i + seq_length]

                t_id_last = int(hist['Ticker_ID'].iloc[-1])
                if t_id_last not in ticker_scalers:
                    continue
                fs = ticker_scalers[t_id_last]['feature_scaler']
                ps = ticker_scalers[t_id_last]['price_scaler']

                eps_use = float(eps_map.get(str(t_id_last), EPS_RET_DEFAULT)) if eps_map else EPS_RET_DEFAULT
                thr_base = float(thresholds.get(str(t_id_last), thresholds.get(t_id_last, 0.5))) if thresholds else 0.5
                cal = calibrators.get(t_id_last, None) if calibrators else None

                Xf = fs.transform(hist[feature_columns].values.astype(np.float32)).reshape(1, seq_length, -1)
                Xt = hist['Ticker_ID'].values.astype(np.int32).reshape(1, seq_length)
                Xm = hist['Market_ID_enc'].values.astype(np.int32).reshape(1, seq_length)

                # predict μ,σ (scaled) → raw
                y_params = best_model.predict([Xf, Xt, Xm], verbose=0)  # (1,2)
                mu_s = float(y_params[0,0]); log_sigma_s = float(y_params[0,1])
                mu_raw, sigma_raw = mu_sigma_to_raw(mu_s, log_sigma_s, ps)

                last_close = float(hist['Close'].iloc[-1])
                price_pred = float(last_close * math.exp(mu_raw))

                # prob up (analytic + optional MC for uncertainty)
                if sigma_raw <= 1e-8:
                    p_up = 1.0 if (mu_raw - eps_use) > 0.0 else 0.0
                    p_unc = 0.0
                else:
                    z = (mu_raw - eps_use)/sigma_raw
                    p_up = norm_cdf(z)
                    if use_mc_dropout:
                        p_up_mc, p_up_std = predict_pup_with_mc(best_model, Xf, Xt, Xm, ps, eps_use, n=MC_DIR_SAMPLES)
                        p_up, p_unc = p_up_mc, p_up_std
                    else:
                        p_unc = 0.0

                # calibrate
                p_cal_raw = float(cal.transform([p_up])[0]) if cal is not None else float(p_up)

                # EMA smoothing
                p_cal_smooth = float(EMA_ALPHA*p_cal_raw + (1.0-EMA_ALPHA)*ema_prev)
                ema_prev = p_cal_smooth

                # adaptive threshold (ใช้ prob RAW เพื่อวัด bias distribution)
                p_hist_for_adapt.append(p_cal_raw)
                thr_used, thr_adapt = thr_base, 0.0
                if ADAPTIVE_THR and (len(p_hist_for_adapt) >= ADAPT_MIN_STEPS):
                    pos_rate_now = float(np.mean(np.array(p_hist_for_adapt) >= thr_base))
                    if (pos_rate_now < ADAPT_LOW) or (pos_rate_now > ADAPT_HIGH):
                        # ตั้ง threshold ให้ median/quantile ตรง target (เช่น 50%)
                        thr_adapt = float(np.quantile(p_hist_for_adapt, 1.0 - ADAPT_POS_TARGET))
                        thr_used = thr_adapt

                # final decision ใช้ prob "SMOOTHED" กับ threshold ที่เลือก
                pred_dir = int(p_cal_smooth >= thr_used)

                actual_price = float(targ['Close'])
                actual_dir = int(actual_price > last_close)

                # stream
                writer_pred.writerow([
                    ticker, targ['Date'], cidx+1, i+1,
                    price_pred, actual_price, pred_dir, actual_dir,
                    p_cal_raw, p_cal_smooth, p_unc,
                    thr_used, thr_base, thr_adapt, eps_use,
                    last_close, actual_price - last_close, price_pred - last_close
                ])

                # for metrics
                pred_dir_list.append(pred_dir)
                actual_dir_list.append(actual_dir)
                p_cal_raw_list.append(p_cal_raw)
                p_cal_smooth_list.append(p_cal_smooth)
                actual_price_list.append(actual_price)
                pred_price_list.append(price_pred)
                last_close_list.append(last_close)
                thr_used_list.append(thr_used)
                thr_base_list.append(thr_base)
                thr_adapt_list.append(thr_adapt)

                ticker_pred_count += 1

                # ----- online learning (เฉพาะหัวราคา) -----
                if online_learning:
                    # ใช้ความเชื่อมั่นจาก prob "หลัง smoothing" เทียบกับ thr_used
                    conf = abs(p_cal_smooth - thr_used)
                    ok_dir = (conf >= margin) and (p_unc <= unc_max) if conf_gate else True

                    if ok_dir:
                        batch_Xf.append(Xf); batch_Xt.append(Xt); batch_Xm.append(Xm)
                        true_logret = float(np.log(actual_price / last_close))
                        true_logret = float(np.clip(true_logret, -0.25, 0.25))
                        batch_yp.append(ps.transform(np.array([[true_logret]], np.float32)))
                        batch_sw_price.append(0.25)

                    do_retrain = ((i + 1) % retrain_frequency == 0) or (i == step_total - 1)
                    if do_retrain and len(batch_Xf) >= 5:
                        bf = np.concatenate(batch_Xf, axis=0).astype(np.float32)
                        bt = np.concatenate(batch_Xt, axis=0).astype(np.int32)
                        bm = np.concatenate(batch_Xm, axis=0).astype(np.int32)
                        bp = np.concatenate(batch_yp, axis=0).astype(np.float32)
                        sw_p = match_len_vec(batch_sw_price, len(bf))
                        best_model.fit([bf, bt, bm], {'price_params': bp},
                                       sample_weight=sw_p.reshape(-1,1),
                                       epochs=1, batch_size=len(bf), verbose=0, shuffle=False)
                        del bf, bt, bm, bp, sw_p
                        batch_Xf, batch_Xt, batch_Xm = [], [], []
                        batch_yp, batch_sw_price = [], []
                        gc.collect()

            fpred.close()

            # ----- metrics per chunk -----
            if len(pred_dir_list) > 0:
                p_pred_chunk = np.asarray(pred_dir_list, dtype=np.int8)
                y_true_chunk = np.asarray(actual_dir_list, dtype=np.int8)
                a_p = np.asarray(actual_price_list, dtype=np.float32)
                p_p = np.asarray(pred_price_list,   dtype=np.float32)

                if verbose:
                    uniq_pred_vals, uniq_pred_cnts = np.unique(p_pred_chunk, return_counts=True)
                    uniq_true_vals, uniq_true_cnts = np.unique(y_true_chunk, return_counts=True)
                    print(f"  🔎 uniq_pred={dict(zip(uniq_pred_vals.tolist(), uniq_pred_cnts.tolist()))} "
                          f"| uniq_true={dict(zip(uniq_true_vals.tolist(), uniq_true_cnts.tolist()))} "
                          f"| thr_base_mean={np.mean(thr_base_list):.3f} "
                          f"| thr_used_mean={np.mean(thr_used_list):.3f}")

                err = a_p - p_p
                mae = float(np.mean(np.abs(err)))
                rmse = float(np.sqrt(np.mean(err**2)))
                y_mean = float(np.mean(a_p))
                ss_res = float(np.sum((a_p - p_p)**2))
                ss_tot = float(np.sum((a_p - y_mean)**2)) if len(a_p) > 1 else 0.0
                r2 = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else 0.0

                # eval metrics ใช้ผลตัดสินใจจริง (หลัง smoothing + adaptive) ที่เราออกไป
                if len(np.unique(p_pred_chunk)) < 2 or len(np.unique(y_true_chunk)) < 2:
                    acc = float(accuracy_score(y_true_chunk, p_pred_chunk))
                    f1  = float(f1_score(y_true_chunk, p_pred_chunk, zero_division=0))
                    mcc = 0.0
                else:
                    acc = float(accuracy_score(y_true_chunk, p_pred_chunk))
                    f1  = float(f1_score(y_true_chunk, p_pred_chunk))
                    mcc = float(matthews_corrcoef(y_true_chunk, p_pred_chunk))
                    if verbose:
                        tn, fp, fn, tp = confusion_matrix(y_true_chunk, p_pred_chunk, labels=[0,1]).ravel()
                        print(f"  📐 CM: TN={tn} FP={fp} FN={fn} TP={tp}")

                with open(stream_chunk_path, 'a', newline='', encoding='utf-8') as fchunk:
                    writer_chunk = csv.writer(fchunk)
                    writer_chunk.writerow([
                        ticker, cidx+1, str(chunk['Date'].min()), str(chunk['Date'].max()), len(a_p),
                        mae, rmse, r2, acc, f1, mcc
                    ])

                # accumulate overall per ticker
                acc_tkr = overall_accum.get(ticker)
                if acc_tkr is None:
                    acc_tkr = {
                        'count': 0,
                        'sum_abs_err': 0.0,
                        'sum_sq_err' : 0.0,
                        'sum_y'      : 0.0,
                        'sum_y2'     : 0.0,
                        'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0
                    }
                acc_tkr['count']       += int(len(a_p))
                acc_tkr['sum_abs_err'] += float(np.sum(np.abs(err)))
                acc_tkr['sum_sq_err']  += float(np.sum(err**2))
                acc_tkr['sum_y']       += float(np.sum(a_p))
                acc_tkr['sum_y2']      += float(np.sum(a_p**2))

                if len(np.unique(p_pred_chunk)) > 1 and len(np.unique(y_true_chunk)) > 1:
                    tn, fp, fn, tp = confusion_matrix(y_true_chunk, p_pred_chunk, labels=[0,1]).ravel()
                    acc_tkr['tp'] += int(tp); acc_tkr['fp'] += int(fp)
                    acc_tkr['tn'] += int(tn); acc_tkr['fn'] += int(fn)

                overall_accum[ticker] = acc_tkr

            # free per chunk
            del chunk, pred_dir_list, actual_dir_list, p_cal_raw_list, p_cal_smooth_list
            del actual_price_list, pred_price_list, last_close_list
            del thr_used_list, thr_base_list, thr_adapt_list, p_hist_for_adapt
            gc.collect()

        if verbose:
            print(f"🟩 Ticker {ticker} completed | total_preds={ticker_pred_count}", flush=True)

        del g; gc.collect()

    # ----- OVERALL (per ticker) -----
    with open(stream_overall_path, 'w', newline='', encoding='utf-8') as foverall:
        writer_overall = csv.writer(foverall)
        writer_overall.writerow([
            'Ticker','Total_Predictions','MAE','RMSE','R2_Score',
            'Direction_Accuracy','Direction_F1_Score','Direction_Precision','Direction_Recall'
        ])
        for tkr, acc_tkr in overall_accum.items():
            n = max(1, acc_tkr['count'])
            mae  = acc_tkr['sum_abs_err'] / n
            rmse = math.sqrt(acc_tkr['sum_sq_err'] / n)
            y_mean = acc_tkr['sum_y'] / n
            ss_tot = acc_tkr['sum_y2'] - n * (y_mean**2)
            r2 = 1.0 - (acc_tkr['sum_sq_err'] / ss_tot) if ss_tot > 1e-9 else 0.0

            tp = acc_tkr['tp']; fp = acc_tkr['fp']; tn = acc_tkr['tn']; fn = acc_tkr['fn']
            total_cm = tp + fp + tn + fn
            if total_cm > 0:
                acc_cls = (tp + tn) / total_cm
                prec    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1      = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
            else:
                acc_cls = prec = rec = f1 = 0.0

            writer_overall.writerow([tkr, n, mae, rmse, r2, acc_cls, f1, prec, rec])

    dt = time.perf_counter() - t0
    if verbose:
        print(f"\n🏁 WFV done (memory-light) | elapsed={dt:.1f}s")
        print(f"📝 streamed predictions → {stream_preds_path}")
        print(f"📝 streamed chunk metrics → {stream_chunk_path}")
        print(f"📝 streamed overall metrics → {stream_overall_path}")
        print(f"📝 eps per ticker saved → {EPS_PER_TICKER_PATH}")

    return None, overall_accum

# =============================================================================
# 13) Run WFV
# =============================================================================
# load artifacts from step 10
eps_map = {}
try:
    with open(EPS_PER_TICKER_PATH,'r') as f:
        eps_map = json.load(f)
except Exception:
    eps_map = {}

try:
    calibrators = joblib.load(CALIBRATORS_PATH)
except Exception:
    calibrators = {}

try:
    with open(THRESHOLDS_PATH,'r') as f:
        thresholds = json.load(f)
except Exception:
    thresholds = {}

predictions_df, results_per_ticker = walk_forward_validation_prob_batch(
    model=best_model,
    df=test_df,
    feature_columns=feature_columns,
    ticker_scalers=ticker_scalers,
    ticker_encoder=ticker_encoder,
    market_encoder=market_encoder,
    seq_length=int(BEST_PARAMS['seq_length']),
    retrain_frequency=int(BEST_PARAMS['retrain_frequency']),
    chunk_size=int(BEST_PARAMS['chunk_size']),
    online_learning=True,
    use_mc_dropout=True,
    calibrators=calibrators,
    thresholds=thresholds,
    eps_map=eps_map,                      # ← ใช้ EPS ราย ticker
    conf_gate=CONF_GATE,
    unc_max=UNC_MAX,
    margin=MARGIN,
    allow_price_online=ALLOW_PRICE_ONLINE,
    verbose=True,
    verbose_every=200,
    ticker_limit=None
)

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
        'memory_light_wfv': MEMORY_LIGHT_WFV,
        'ema_alpha': EMA_ALPHA,
        'adaptive_threshold': {
            'enabled': ADAPTIVE_THR,
            'min_steps': ADAPT_MIN_STEPS,
            'low': ADAPT_LOW,
            'high': ADAPT_HIGH,
            'target_pos': ADAPT_POS_TARGET
        }
    }
}
with open('production_model_config.json','w') as f: json.dump(production_config, f, indent=2)

best_model.save('best_hypertuned_model.keras')

# =============================================================================
# 15) Serve function (real-time inference for one ticker)
# =============================================================================
def serve_one(df_latest_1ticker, artifacts, seq_length=int(BEST_PARAMS['seq_length']), use_mc=True):
    """
    คืนค่า pred_price, P(UP) (calibrated+smoothed one-shot), uncertainty, decision
    * smoothing ใช้ค่าเดียว (ไม่มี history) → เท่ากับ prob_calibrated เอง
    """
    g = df_latest_1ticker.sort_values('Date').tail(seq_length)
    ticker = g['Ticker'].iloc[-1]
    market = g['Market_ID'].iloc[-1]
    t_id = int(artifacts['ticker_encoder'].transform([ticker])[0])
    m_id = int(artifacts['market_encoder'].transform([market])[0])

    fs = artifacts['ticker_scalers'][t_id]['feature_scaler']
    ps = artifacts['ticker_scalers'][t_id]['price_scaler']
    Xf = fs.transform(g[artifacts['feature_columns']].values.astype(np.float32)).reshape(1, seq_length, -1)
    Xt = np.full((1, seq_length), t_id, dtype=np.int32)
    Xm = np.full((1, seq_length), m_id, dtype=np.int32)

    y_params = artifacts['model'].predict([Xf, Xt, Xm], verbose=0)
    mu_s = float(y_params[0,0]); log_sigma_s = float(y_params[0,1])
    mu_raw, sigma_raw = mu_sigma_to_raw(mu_s, log_sigma_s, ps)

    last_close = float(g['Close'].iloc[-1])
    pred_price = float(last_close * math.exp(mu_raw))

    eps_use = float(artifacts.get('eps_map', {}).get(str(t_id), EPS_RET_DEFAULT))
    if sigma_raw <= 1e-8:
        p_up = 1.0 if (mu_raw - eps_use) > 0.0 else 0.0
        p_unc = 0.0
    else:
        z = (mu_raw - eps_use) / sigma_raw
        p_up = norm_cdf(z)
        if use_mc:
            pups = []
            for _ in range(artifacts.get('mc_dir_samples', MC_DIR_SAMPLES)):
                y = artifacts['model']([Xf, Xt, Xm], training=True).numpy()
                mu_s_i = float(y[0,0]); log_sigma_s_i = float(y[0,1])
                mu_raw_i, sigma_raw_i = mu_sigma_to_raw(mu_s_i, log_sigma_s_i, ps)
                pups.append(prob_up_from_mu_sigma(mu_raw_i, sigma_raw_i, eps_use))
            p_up = float(np.mean(pups)); p_unc = float(np.std(pups, ddof=0))
        else:
            p_unc = 0.0

    cal = artifacts['calibrators'].get(t_id, None)
    p_cal = float(cal.transform([p_up])[0]) if cal is not None else p_up
    thr = float(artifacts['thresholds'].get(str(t_id), artifacts['thresholds'].get(t_id, 0.5)))
    direction = int(p_cal >= thr)

    return {
        'ticker': ticker,
        'ticker_id': int(t_id),
        'pred_price': float(pred_price),
        'dir_prob_raw': float(p_up),
        'dir_prob_cal': float(p_cal),
        'dir_prob_unc': float(p_unc),
        'decision': int(direction),
        'threshold_used': float(thr),
        'eps_used': float(eps_use)
    }

# Bundle artifacts
artifacts = {
    'model': best_model,
    'ticker_scalers': ticker_scalers,
    'ticker_encoder': ticker_encoder,
    'market_encoder': market_encoder,
    'feature_columns': feature_columns,
    'calibrators': {int(k):v for k,v in calibrators.items()},
    'thresholds': thresholds,
    'mc_dir_samples': MC_DIR_SAMPLES,
    'eps_map': eps_map
}
joblib.dump(artifacts, 'serving_artifacts.pkl')

print("\n✅ All done (prob-regression + dir-boost). Files saved:")
print(f" - {STREAM_PRED_PATH}")
print(f" - {STREAM_CHUNK_PATH}")
print(f" - {STREAM_OVERALL_PATH}")
print(f" - {EPS_PER_TICKER_PATH}")
print(" - best_v6_plus_minimal_tuning_v2_final_model.keras, best_hypertuned_model.keras")
print(" - *_training_history.csv, *_encoders.pkl, ticker_scalers.pkl, feature_columns.pkl")
print(f" - {CALIBRATORS_PATH}, {THRESHOLDS_PATH}")
print(" - production_model_config.json, serving_artifacts.pkl")
