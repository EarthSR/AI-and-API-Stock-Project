# -*- coding: utf-8 -*-
"""
Price-only inference (LSTM, GRU, META) + Online Learning (training-parity gates)
- MySQL + dotenv (no SQLAlchemy)
- Default mode: preopen (with "near" time windows, configurable via ENV)
- Online learning state persisted to JSON (per-model) — no DB table for status

Artifacts expected under {MODEL}/logs/models:
  - best_model_static.keras
  - serving_artifacts.pkl  (must have: ticker_scalers, ticker_encoder, market_encoder, feature_columns;
                            may include: iso_cals, meta_lrs, thresholds, val_prev_map, mc_dir_samples)
  - production_model_config.json

META_DIR:
  - xgb_price.json
  - xgb_price.meta.joblib (best_k, q_lo, q_hi)

Writes predictions to StockDetail:
  - PredictionClose_LSTM
  - PredictionClose_GRU
  - PredictionClose_Ensemble
"""

import os, sys, io, math, json, gc, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ----- stdout unicode safe -----
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')

from pathlib import Path
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import joblib
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# register_keras_serializable (fix NameError in some envs)
try:
    from keras.saving import register_keras_serializable
except Exception:
    from tensorflow.keras.utils import register_keras_serializable

# optional psutil for memory-aware MC dropout
try:
    import psutil
except Exception:
    psutil = None

# --- dotenv + mysql ---
from dotenv import load_dotenv
import mysql.connector

# ===================== PATHS / ENV =====================
ROOT = Path(__file__).resolve().parent
DOTENV_PATH = os.getenv("DOTENV_PATH", str(ROOT / "config.env"))
load_dotenv(DOTENV_PATH)

LOCAL_TZ = os.getenv("LOCAL_TZ", "Asia/Bangkok")

LSTM_DIR_DEFAULT = os.getenv("LSTM_DIR", str(ROOT / ".." / "LSTM_model"))
GRU_DIR_DEFAULT  = os.getenv("GRU_DIR",  str(ROOT / ".." / "GRU_model"))
META_DIR_DEFAULT = os.getenv("META_DIR", str(ROOT / ".." / "Ensemble_Model"))

# ===================== PRE-OPEN NEAR WINDOWS =====================
TH_OPEN_LOCAL = os.getenv("TH_OPEN_LOCAL", "08:30")  # HH:MM local
US_OPEN_LOCAL = os.getenv("US_OPEN_LOCAL", "20:30")
PREOPEN_BEFORE_MIN = int(os.getenv("PREOPEN_BEFORE_MIN", "15"))   # minutes before open
PREOPEN_AFTER_MIN  = int(os.getenv("PREOPEN_AFTER_MIN",  "10"))   # minutes after open
PREOPEN_NEAR_MIN   = int(os.getenv("PREOPEN_NEAR_MIN",   "60"))   # consider "near" within minutes

def _today_window(open_hhmm: str, before_min: int, after_min: int):
    tz = ZoneInfo(LOCAL_TZ)
    now_dt = datetime.now(tz)
    hh, mm = map(int, open_hhmm.split(":"))
    open_dt = now_dt.replace(hour=hh, minute=mm, second=0, microsecond=0)
    start_t = (open_dt - timedelta(minutes=before_min)).time()
    end_t   = (open_dt + timedelta(minutes=after_min)).time()
    return start_t, end_t, open_dt

def current_preopen_windows():
    th_s, th_e, th_open = _today_window(TH_OPEN_LOCAL, PREOPEN_BEFORE_MIN, PREOPEN_AFTER_MIN)
    us_s, us_e, us_open = _today_window(US_OPEN_LOCAL, PREOPEN_BEFORE_MIN, PREOPEN_AFTER_MIN)
    return {
        "TH": {"start": th_s, "end": th_e, "open_dt": th_open, "db_market": "Thailand"},
        "US": {"start": us_s, "end": us_e, "open_dt": us_open, "db_market": "America"},
    }

# ===================== ONLINE LEARNING (training-parity) =====================
ONLINE_LEARNING_ENABLED = os.getenv("ONLINE_LEARNING", "true").lower() in ("1","true","yes","y")
ONLINE_TRAIN_STEPS_PER_TICKER = int(os.getenv("ONLINE_TRAIN_STEPS_PER_TICKER", "1"))

# pacing
ONLINE_UPDATE_EVERY    = int(os.getenv("ONLINE_UPDATE_EVERY", "16"))    # general
ONLINE_UPDATE_EVERY_US = int(os.getenv("ONLINE_UPDATE_EVERY_US", "24")) # US special
ONLINE_UPDATE_MAX_PER_RUN    = int(os.getenv("ONLINE_UPDATE_MAX_PER_RUN", "48"))
ONLINE_UPDATE_MAX_PER_RUN_US = int(os.getenv("ONLINE_UPDATE_MAX_PER_RUN_US", "12"))

# gates
CONF_GATE = True
UNC_MAX   = float(os.getenv("UNC_MAX", "0.10"))
MARGIN    = float(os.getenv("MARGIN", "0.05"))
Z_GATE_ONLINE    = float(os.getenv("Z_GATE_ONLINE", "1.05"))
Z_GATE_ONLINE_US = float(os.getenv("Z_GATE_ONLINE_US", "1.05"))

# market policy
ALLOW_PRICE_ONLINE_MARKET = {'US': True, 'TH': False, 'OTHER': False}

# uncertainty (MC dropout)
MC_TRIGGER_BAND = 0.12
MC_DIR_SAMPLES_DEFAULT = 3
MEM_LOW_MB  = 800.0
MEM_CRIT_MB = 400.0

# inference constants
EPS_RET = 0.0011
SIGMA_VOL_SPLIT = 0.013

# EMA smoothing
USE_EMA_PROB = True
ALPHA_EMA_LOWVOL = 0.68
ALPHA_EMA_HIVOL  = 0.62

# PSC
USE_PSC = True
PRIOR_EMA_ALPHA  = 0.07
TARGET_EMA_ALPHA = 0.15
PRIOR_MIN_N      = 40
ACT_PREV_MIN_N   = 12
PSC_LOGIT_CAP    = 0.20

# Trend prior
USE_TREND_PRIOR   = True
TREND_WIN         = 7
TREND_KAPPA       = 2.0
TREND_W_LOWVOL_TH = 0.08
TREND_W_HIVOL_TH  = 0.12
TREND_W_LOWVOL_US = 0.04
TREND_W_HIVOL_US  = 0.07
TREND_W_OVR = {'GOOGL':0.04,'NVDA':0.05,'AAPL':0.05,'MSFT':0.05, 'ADVANC':0.14}

# thresholds & market deltas
THRESH_MIN = 0.50
THR_CLIP_LOW  = 0.44
THR_CLIP_HIGH = 0.86
THR_CLIP_LOW_TH = 0.448
TH_MARKET_DELTA = {'TH': -0.012, 'US': 0.000, 'OTHER': 0.000}
HIVOL_THR_SHIFT_US = -0.022
HIVOL_THR_SHIFT_TH = -0.017

# smoothing overrides (TH calmer)
ALPHA_EMA_OVR = {
    'ADVANC':0.62,'DITTO':0.62,'HUMAN':0.62,'INET':0.62,'JAS':0.62,
    'DIF':0.60,'TRUE':0.60,'INSET':0.62,'JMART':0.60
}

# per-ticker threshold delta
THR_DELTA_OVR = {
    'NVDA': -0.006, 'GOOGL': -0.006, 'AVGO': -0.006, 'AAPL': -0.006, 'MSFT': -0.006,
    'TSM':  -0.006, 'AMD':  -0.006, 'META': -0.010, 'AMZN': -0.010, 'TSLA': +0.006,
    'INSET': -0.004, 'JAS': 0.000, 'ADVANC': -0.010, 'DITTO': 0.000, 'DIF': 0.000,
    'TRUE':  0.000, 'HUMAN': 0.000, 'INET': 0.000, 'JMART': 0.000,
}

# precision_tune (subset of keys relevant at inference)
PRECISION_TUNE = {
    'AAPL':  {'thr_bump': -0.078, 'ema_alpha': 0.46, 'z_gate': 0.96, 'unc_plus': -0.010},
    'GOOGL': {'thr_bump': -0.040, 'ema_alpha': 0.52},
    'AMZN':  {'thr_bump': -0.085, 'ema_alpha': 0.50},
    'META':  {'thr_bump': -0.036, 'ema_alpha': 0.50},
    'TSLA':  {'thr_bump': -0.040, 'ema_alpha': 0.50},
    'TSM':   {'thr_bump': -0.028, 'ema_alpha': 0.50},
    'MSFT':  {'thr_bump': -0.006, 'ema_alpha': 0.50},
    'AMD':   {'thr_bump': -0.048, 'ema_alpha': 0.48, 'z_gate': 0.96, 'unc_plus': -0.008},
    'NVDA':  {'thr_bump': -0.052, 'ema_alpha': 0.48},
    'AVGO':  {'thr_bump': -0.110, 'ema_alpha': 0.50},

    'ADVANC':{'thr_bump': -0.520, 'ema_alpha': 0.18, 'z_gate': 0.68, 'unc_plus': -0.22},
    'TRUE':  {'thr_bump': -0.200, 'ema_alpha': 0.46},
    'JAS':   {'thr_bump': +0.070, 'ema_alpha': 0.60},
    'DITTO': {'thr_bump': -0.080, 'ema_alpha': 0.52},
    'DIF':   {'thr_bump': -0.060, 'ema_alpha': 0.54},
    'JMART': {'thr_bump': -0.090, 'ema_alpha': 0.50},
    'INET':  {'thr_bump': -0.030, 'ema_alpha': 0.52},
    'INSET': {'thr_bump': -0.012, 'ema_alpha': 0.54},
    'HUMAN': {'thr_bump': -0.016, 'ema_alpha': 0.54},
}

# ===================== Custom Loss/Metric =====================
@register_keras_serializable(package="custom")
def gaussian_nll(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    mu_s, log_sigma_s = tf.split(y_pred, 2, axis=-1)
    sigma_s = tf.nn.softplus(log_sigma_s) + 1e-6
    z = (y_true - mu_s) / sigma_s
    return tf.reduce_mean(0.5*tf.math.log(2.0*np.pi) + tf.math.log(sigma_s) + 0.5*tf.square(z))

@register_keras_serializable(package="custom")
def mae_on_mu(y_true, y_pred):
    mu_s, _ = tf.split(y_pred, 2, axis=-1)
    return tf.reduce_mean(tf.abs(tf.cast(y_true, tf.float32) - mu_s))

# ===================== DB =====================
def get_mysql_conn() -> mysql.connector.connection.MySQLConnection:
    db_config = {
        "host": os.getenv("DB_HOST"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME"),
        "port": int(os.getenv("DB_PORT") or 3306),
        "autocommit": True
    }
    try:
        conn = mysql.connector.connect(**db_config)
        cur = conn.cursor(); cur.execute("SELECT 1"); cur.fetchall(); cur.close()
        print("✅ DB connected")
        return conn
    except mysql.connector.Error as e:
        print(f"❌ DB connect failed: {e}")
        sys.exit(1)

# ===================== Columns / Features =====================
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        'StockSymbol': 'StockSymbol',
        'OpenPrice': 'Open',
        'HighPrice': 'High',
        'LowPrice': 'Low',
        'ClosePrice': 'Close',
        'Volume': 'Volume',
        'Changepercen': 'Change (%)',
        'TotalRevenue': 'Total Revenue',
        'QoQGrowth': 'QoQ Growth (%)',
        'EPS': 'Earnings Per Share (EPS)',
        'ROE': 'ROE (%)',
        'NetProfitMargin': 'Net Profit Margin (%)',
        'DebtToEquity': 'Debt to Equity',
        'PERatio': 'P/E Ratio',
        'P_BV_Ratio': 'P/BV Ratio',
        'Dividend_Yield': 'Dividend Yield (%)',
        'positive_news': 'positive_news',
        'negative_news': 'negative_news',
        'neutral_news': 'neutral_news',
        'Sentiment': 'Sentiment',
        'Market': 'Market'
    }
    out = df.rename(columns=rename_map).copy()
    if 'Market' in out.columns:
        out['Market'] = out['Market'].map({'America':'US','Thailand':'TH'}).fillna(out['Market'])
        out['Market'] = out['Market'].where(out['Market'].isin(['US','TH']), 'OTHER')
    out['Date'] = pd.to_datetime(out['Date'], errors='coerce')
    out['StockSymbol'] = out['StockSymbol'].astype(str)
    return out

import ta

FIN_COLS = ['Total Revenue','QoQ Growth (%)','Earnings Per Share (EPS)','ROE (%)',
            'Net Profit Margin (%)','Debt to Equity','P/E Ratio','P/BV Ratio','Dividend Yield (%)']

BASE_FEATURES = [
    'Open','High','Low','Close','Volume','Change (%)','Sentiment',
    'positive_news','negative_news','neutral_news',
    'Total Revenue','QoQ Growth (%)','Earnings Per Share (EPS)','ROE (%)',
    'ATR','Keltner_High','Keltner_Low','Keltner_Middle','Chaikin_Vol',
    'Donchian_High','Donchian_Low','PSAR',
    'Net Profit Margin (%)','Debt to Equity','P/E Ratio','P/BV Ratio','Dividend Yield (%)',
    'RSI','EMA_10','EMA_20','MACD','MACD_Signal',
    'Bollinger_High','Bollinger_Low','SMA_50','SMA_200'
]

def _add_ta(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g['EMA_12']   = g['Close'].ewm(span=12, adjust=False).mean()
    g['EMA_26']   = g['Close'].ewm(span=26, adjust=False).mean()
    g['EMA_10']   = g['Close'].ewm(span=10, adjust=False).mean()
    g['EMA_20']   = g['Close'].ewm(span=20, adjust=False).mean()
    g['SMA_50']   = g['Close'].rolling(50, min_periods=1).mean()
    g['SMA_200']  = g['Close'].rolling(200, min_periods=1).mean()
    try: g['RSI'] = ta.momentum.RSIIndicator(close=g['Close'], window=14).rsi()
    except: g['RSI'] = np.nan
    g['RSI'] = g['RSI'].fillna(g['RSI'].rolling(5, min_periods=1).mean()).fillna(50.0)
    g['MACD'] = g['EMA_12'] - g['EMA_26']
    g['MACD_Signal'] = g['MACD'].rolling(9, min_periods=1).mean()
    try:
        bb = ta.volatility.BollingerBands(close=g['Close'], window=20, window_dev=2)
        g['Bollinger_High'] = bb.bollinger_hband(); g['Bollinger_Low'] = bb.bollinger_lband()
    except:
        g['Bollinger_High'] = g['Close'].rolling(20, min_periods=1).max()
        g['Bollinger_Low']  = g['Close'].rolling(20, min_periods=1).min()
    try:
        atr = ta.volatility.AverageTrueRange(high=g['High'], low=g['Low'], close=g['Close'], window=14)
        g['ATR'] = atr.average_true_rate() if hasattr(atr,'average_true_rate') else atr.average_true_range()
    except:
        g['ATR'] = (g['High']-g['Low']).rolling(14, min_periods=1).mean()
    try:
        kc = ta.volatility.KeltnerChannel(high=g['High'], low=g['Low'], close=g['Close'], window=20, window_atr=10)
        g['Keltner_High']   = kc.keltner_channel_hband()
        g['Keltner_Low']    = kc.keltner_channel_lband()
        g['Keltner_Middle'] = kc.keltner_channel_mband()
    except:
        rng=(g['High']-g['Low']).rolling(20, min_periods=1).mean()
        mid=g['Close'].rolling(20, min_periods=1).mean()
        g['Keltner_High']=mid+rng; g['Keltner_Low']=mid-rng; g['Keltner_Middle']=mid
    g['High_Low_Diff'] = g['High'] - g['Low']
    g['High_Low_EMA']  = g['High_Low_Diff'].ewm(span=10, adjust=False).mean()
    g['Chaikin_Vol']   = g['High_Low_EMA'].pct_change(10)*100.0
    g['Donchian_High'] = g['High'].rolling(20, min_periods=1).max()
    g['Donchian_Low']  = g['Low'].rolling(20, min_periods=1).min()
    try: g['PSAR'] = ta.trend.PSARIndicator(high=g['High'], low=g['Low'], close=g['Close'], step=0.02, max_step=0.2).psar()
    except: g['PSAR'] = (g['High']+g['Low'])/2.0
    return g

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'Ticker' not in out.columns:
        out['Ticker'] = out['StockSymbol'].astype(str)
    out = out.sort_values(['Ticker','Date']).reset_index(drop=True)

    if 'Sentiment' in out.columns:
        out['Sentiment'] = out['Sentiment'].replace({'Positive':1,'Negative':-1,'Neutral':0}).fillna(0).astype(np.int8)
    else:
        out['Sentiment'] = 0

    out['Change']     = out['Close'] - out['Open']
    out['Change (%)'] = out.groupby('Ticker')['Close'].pct_change()*100.0
    upper = out['Change (%)'].quantile(0.99); lower = out['Change (%)'].quantile(0.01)
    out['Change (%)'] = out['Change (%)'].clip(lower, upper)

    _tick_bak = out['Ticker'].values
    try:
        out = out.groupby('Ticker', group_keys=False).apply(_add_ta, include_groups=True)
    except TypeError:
        out = out.groupby('Ticker', group_keys=False).apply(_add_ta)
        if 'Ticker' not in out.columns and len(out) == len(_tick_bak):
            out.insert(0, 'Ticker', _tick_bak)

    for c in FIN_COLS:
        if c not in out.columns: out[c] = np.nan
    out[FIN_COLS] = out[FIN_COLS].replace(0, np.nan)
    out[FIN_COLS] = out.groupby('Ticker')[FIN_COLS].ffill()

    for c in ['positive_news','negative_news','neutral_news']:
        if c not in out.columns: out[c] = 0.0

    for c in BASE_FEATURES:
        if c not in out.columns: out[c] = 0.0
    out[BASE_FEATURES] = (out.groupby('Ticker')[BASE_FEATURES]
                            .apply(lambda g: g.fillna(method='ffill'))
                            .reset_index(level=0, drop=True))
    out[BASE_FEATURES] = out[BASE_FEATURES].fillna(0.0)
    return out

# ===================== FETCH (MySQL) =====================
def fetch_latest_data(conn: mysql.connector.connection.MySQLConnection,
                      market_filter: str | None = None) -> pd.DataFrame:
    if market_filter is None:
        where_mkt = "s.Market IN ('America','Thailand')"
        params = ()
    else:
        where_mkt = "s.Market = %s"
        params = (market_filter,)

    q = f"""
        SELECT 
            sd.Date,
            sd.StockSymbol,
            s.Market,
            sd.OpenPrice  AS OpenPrice,
            sd.HighPrice  AS HighPrice,
            sd.LowPrice   AS LowPrice,
            sd.ClosePrice AS ClosePrice,
            sd.Volume,
            sd.P_BV_Ratio,
            sd.Sentiment,
            sd.Changepercen,
            sd.TotalRevenue,
            sd.QoQGrowth,
            sd.EPS,
            sd.ROE,
            sd.NetProfitMargin,
            sd.DebtToEquity,
            sd.PERatio,
            sd.Dividend_Yield,
            sd.positive_news,
            sd.negative_news,
            sd.neutral_news
        FROM StockDetail sd
        LEFT JOIN Stock s ON sd.StockSymbol = s.StockSymbol
        WHERE {where_mkt}
          AND sd.Date >= CURDATE() - INTERVAL 370 DAY
        ORDER BY sd.StockSymbol, sd.Date;
    """
    cur = conn.cursor()
    cur.execute(q, params)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    cur.close()
    if not rows:
        print("❌ No data returned from DB")
        return pd.DataFrame(columns=['Date','StockSymbol'])
    raw = pd.DataFrame(rows, columns=cols)
    raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce', utc=False)
    df = standardize_columns(raw)

    # fill calendar days per symbol
    filled = []
    for sym, g in df.groupby("StockSymbol", sort=False):
        g = g.sort_values("Date").copy()
        g['Date'] = pd.to_datetime(g['Date'], errors='coerce', utc=False)
        g['StockSymbol'] = g['StockSymbol'].astype(str)
        start, end = g['Date'].min(), g['Date'].max()
        if pd.isna(start) or pd.isna(end): continue
        tmp = pd.DataFrame({"Date": pd.date_range(start, end, freq='D'),
                            "StockSymbol": str(sym)})
        merged = pd.merge(tmp, g, on=["Date","StockSymbol"], how="left")
        if 'Market' in merged.columns:
            mval = g['Market'].dropna().iloc[-1] if g['Market'].notna().any() else None
            merged['Market'] = merged['Market'].ffill().bfill()
            if merged['Market'].isna().any():
                merged['Market'] = merged['Market'].fillna(mval if mval is not None else "US")
        financial = [
            'Total Revenue','QoQ Growth (%)','Earnings Per Share (EPS)','ROE (%)',
            'Net Profit Margin (%)','Debt to Equity','P/E Ratio','P/BV Ratio','Dividend Yield (%)',
            'positive_news','negative_news','neutral_news','Sentiment'
        ]
        for c in financial:
            if c in merged.columns:
                merged[c] = pd.to_numeric(merged[c], errors='coerce').ffill().bfill().fillna(0)
        for c in ['Open','High','Low','Close','Volume']:
            if c in merged.columns:
                merged[c] = pd.to_numeric(merged[c], errors='coerce')
        filled.append(merged)

    df2 = pd.concat(filled, ignore_index=True)
    df2 = compute_indicators(df2)
    need = list(set(['Open','High','Low','Close']).intersection(df2.columns))
    if need: df2 = df2.dropna(subset=need)
    df2 = df2.ffill().bfill().fillna(0)
    return df2

# ===================== SAVE ราคา (MySQL) =====================
def save_predictions_simple(predictions_df: pd.DataFrame,
                            conn: mysql.connector.connection.MySQLConnection) -> bool:
    if predictions_df is None or predictions_df.empty:
        print("❌ ไม่มีข้อมูลพยากรณ์ที่จะบันทึก")
        return False
    try:
        cur = conn.cursor()
        success_count = created_count = updated_count = 0
        for _, row in predictions_df.iterrows():
            sym = str(row['StockSymbol'])
            dt  = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
            cur.execute("SELECT COUNT(*) FROM StockDetail WHERE StockSymbol=%s AND Date=%s", (sym, dt))
            (exists,) = cur.fetchone()

            lstm_price = float(row.get('LSTM_Price', 0.0) or 0.0)
            gru_price  = float(row.get('GRU_Price', 0.0) or 0.0)
            ens_price  = float(row.get('Ensemble_Price', 0.0) or 0.0)

            if exists and int(exists) > 0:
                cur.execute("""
                    UPDATE StockDetail
                       SET PredictionClose_LSTM=%s,
                           PredictionClose_GRU=%s,
                           PredictionClose_Ensemble=%s
                     WHERE StockSymbol=%s AND Date=%s
                """, (lstm_price, gru_price, ens_price, sym, dt))
                updated_count += 1
                print(f"✅ UPDATE {sym} @ {dt}")
            else:
                cur.execute("""
                    INSERT INTO StockDetail
                        (StockSymbol, Date,
                         PredictionClose_LSTM,
                         PredictionClose_GRU,
                         PredictionClose_Ensemble)
                    VALUES (%s, %s, %s, %s, %s)
                """, (sym, dt, lstm_price, gru_price, ens_price))
                created_count += 1
                print(f"✅ INSERT {sym} @ {dt}")

            success_count += 1

        conn.commit()
        cur.close()
        print(f"\n💾 DB upsert done: {success_count} rows (new {created_count}, updated {updated_count})")
        return success_count > 0
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการบันทึก DB: {e}")
        import traceback; traceback.print_exc()
        return False

# ===================== Load artifacts / compile =====================
def load_base_artifacts(model_root_dir: str):
    MODEL_DIR = os.path.join(model_root_dir, "logs", "models")
    def p(name): return os.path.join(MODEL_DIR, name)

    with open(p("production_model_config.json"), 'r', encoding='utf-8') as f:
        cfg_all = json.load(f)
    seq_len = int(cfg_all.get('model_config', {}).get('seq_length', 10))
    lr = float(cfg_all.get('model_config', {}).get('learning_rate', 1.6e-4))

    model = load_model(p("best_model_static.keras"), compile=False, safe_mode=False)
    try:
        model.compile(optimizer=Adam(learning_rate=lr), loss=gaussian_nll, metrics=[mae_on_mu])
    except Exception:
        model.compile(optimizer=Adam(learning_rate=1e-4), loss=gaussian_nll, metrics=[mae_on_mu])

    artifacts = joblib.load(p("serving_artifacts.pkl"))
    mc_samples = int(artifacts.get('mc_dir_samples', MC_DIR_SAMPLES_DEFAULT)) if isinstance(artifacts, dict) else MC_DIR_SAMPLES_DEFAULT

    return dict(
        model_dir=MODEL_DIR,
        seq_len=seq_len,
        model=model,
        ticker_scalers=artifacts['ticker_scalers'],
        ticker_encoder=artifacts['ticker_encoder'],
        market_encoder=artifacts['market_encoder'],
        feature_columns=artifacts['feature_columns'],
        iso_cals = artifacts.get('iso_cals', {}),
        meta_lrs = artifacts.get('meta_lrs', {}),
        thresholds = artifacts.get('thresholds', {}),
        val_prev_map = artifacts.get('val_prev_map', {}),
        mc_dir_samples = mc_samples,
        config=cfg_all
    )

def load_meta_artifacts(meta_dir: str):
    model_path = os.path.join(meta_dir, "xgb_price.json")
    meta_path  = os.path.join(meta_dir, "xgb_price.meta.joblib")
    booster = xgb.Booster(); booster.load_model(model_path)
    meta = joblib.load(meta_path)
    best_k = int(meta.get('best_k', 200))
    q_lo = float(meta.get('q_lo', -0.05)); q_hi = float(meta.get('q_hi', 0.05))
    return booster, best_k, q_lo, q_hi

# ===================== Online state (JSON) =====================
class OnlineStateManager:
    """
    Robust JSON state manager for online learning (per-model).
    Schema:
    {
      "last_snapshot": ISO8601,
      "tickers": {
        "AAPL": {
          "step_ctr": 0,
          "updates_this_run": 0,
          "ema_state": null,
          "pi_pred_ema": 0.5,
          "pi_target_ema": 0.5,
          "n": 0,
          "na": 0,
          "last_online_date": null,
          "last_updated_at": null
        },
        ...
      }
    }
    """
    def __init__(self, model_dir: str, tag: str):
        self.path = os.path.join(model_dir, f"{tag}_online_state.json")
        self.state = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    st = json.load(f)
                if not isinstance(st, dict):
                    raise ValueError("state is not dict")
                if "tickers" not in st or not isinstance(st["tickers"], dict):
                    st["tickers"] = {}
                if "last_snapshot" not in st:
                    st["last_snapshot"] = None
                return st
            except Exception:
                pass
        # default skeleton
        return {"last_snapshot": None, "tickers": {}}

    def get_t(self, ticker: str) -> dict:
        if "tickers" not in self.state or not isinstance(self.state["tickers"], dict):
            self.state["tickers"] = {}
        rec = self.state["tickers"].get(ticker)
        if rec is None or not isinstance(rec, dict):
            rec = {
                "step_ctr": 0,
                "updates_this_run": 0,
                "ema_state": None,
                "pi_pred_ema": 0.5,
                "pi_target_ema": 0.5,
                "n": 0,
                "na": 0,
                "last_online_date": None,
                "last_updated_at": None
            }
            self.state["tickers"][ticker] = rec
        else:
            # ensure required keys exist (backward compatible)
            rec.setdefault("step_ctr", 0)
            rec.setdefault("updates_this_run", 0)
            rec.setdefault("ema_state", None)
            rec.setdefault("pi_pred_ema", 0.5)
            rec.setdefault("pi_target_ema", 0.5)
            rec.setdefault("n", 0)
            rec.setdefault("na", 0)
            rec.setdefault("last_online_date", None)
            rec.setdefault("last_updated_at", None)
        return rec

    def set_t(self, ticker: str, rec: dict):
        if "tickers" not in self.state or not isinstance(self.state["tickers"], dict):
            self.state["tickers"] = {}
        self.state["tickers"][ticker] = rec

    def bump_snapshot(self):
        self.state["last_snapshot"] = datetime.now(ZoneInfo(LOCAL_TZ)).isoformat()

    def save(self):
        try:
            self.bump_snapshot()
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
            print(f"💾 wrote {self.path}")
        except Exception as e:
            print(f"⚠️ failed to write online state: {e}")

# ===================== Prob & gates helpers =====================
def _encode_market(menc, mk_name: str) -> int:
    try:
        return int(menc.transform([mk_name])[0])
    except Exception:
        return 0

def _norm_cdf(x): return 0.5*(1.0 + math.erf(x / math.sqrt(2.0)))
def _softplus(x): return math.log1p(math.exp(x))

def _free_ram_mb():
    try:
        if psutil is None:
            return float('inf')
        return float(psutil.virtual_memory().available) / 1e6
    except Exception:
        return float('inf')

def _trend_weight(ticker: str, sigma_raw: float, market_name: str) -> float:
    if ticker in TREND_W_OVR:
        return TREND_W_OVR[ticker]
    if market_name == 'US':
        return TREND_W_LOWVOL_US if sigma_raw < SIGMA_VOL_SPLIT else TREND_W_HIVOL_US
    else:
        return TREND_W_LOWVOL_TH if sigma_raw < SIGMA_VOL_SPLIT else TREND_W_HIVOL_TH

def _alpha_for(ticker: str, sigma_raw: float) -> float:
    base = ALPHA_EMA_LOWVOL if sigma_raw < SIGMA_VOL_SPLIT else ALPHA_EMA_HIVOL
    return float(ALPHA_EMA_OVR.get(ticker, base))

def _thr_base_for(tid, tname, thresholds: dict, market_name: str) -> float:
    thr = thresholds.get(str(tid), thresholds.get(tid, 0.5)) if thresholds else 0.5
    thr = float(thr)
    if tname in THR_DELTA_OVR:
        thr = thr + THR_DELTA_OVR[tname]
    thr = thr + TH_MARKET_DELTA.get(market_name, 0.0)
    return float(np.clip(thr, THR_CLIP_LOW, THR_CLIP_HIGH))

def _mc_samples(base, need_mc: bool) -> int:
    if not need_mc:
        return 0
    free = _free_ram_mb()
    if free < MEM_CRIT_MB: return 0
    if free < MEM_LOW_MB:  return 1
    return int(max(1, base.get('mc_dir_samples', MC_DIR_SAMPLES_DEFAULT)))

def compute_price_stats(base, hist_df: pd.DataFrame, ticker: str):
    """Return dict with mu_raw, sigma_raw, last_close, market_name, tid, mid, X tensors."""
    m = base['model']; seq_len = base['seq_len']; feats = base['feature_columns']
    tenc = base['ticker_encoder']; menc = base['market_encoder']; scalers = base['ticker_scalers']
    tid = int(tenc.transform([ticker])[0])
    if tid not in scalers: return None
    fs = scalers[tid]['feature_scaler']; ps = scalers[tid]['price_scaler']
    mk_name = str(hist_df['Market'].iloc[-1]) if 'Market' in hist_df.columns else 'OTHER'
    mid = _encode_market(menc, mk_name)
    Xf = fs.transform(hist_df[feats].values.astype(np.float32)).reshape(1, seq_len, -1)
    Xt = np.full((1, seq_len), tid, np.int32)
    Xm = np.full((1, seq_len), mid, np.int32)
    y = m([Xf, Xt, Xm], training=False)
    y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    if y.ndim == 1: y = y.reshape(1, 2)
    mu_s, log_sigma_s = float(y[0,0]), float(y[0,1])
    scale  = getattr(ps, 'scale_',  np.array([1.0], dtype=np.float32))[0]
    center = getattr(ps, 'center_', np.array([0.0], dtype=np.float32))[0]
    sigma_s = max(_softplus(log_sigma_s) + 1e-6, 1e-6)
    mu_raw = mu_s * scale + center
    sigma_raw = sigma_s * scale
    last_close = float(hist_df['Close'].iloc[-1])
    return dict(mu_raw=mu_raw, sigma_raw=sigma_raw, last_close=last_close,
                market_name=('US' if mk_name=='US' else ('TH' if mk_name=='TH' else 'OTHER')),
                tid=tid, mid=mid, Xf=Xf, Xt=Xt, Xm=Xm)

# ====== UPDATED: compute_prob_meta (robust + training-parity) ======
def compute_prob_meta(base, stats, ticker: str, ema_prev: float | None, prior_rec: dict):
    """
    Compute p_use (EMA), p_unc, thr_eff, z, with PSC & trend prior & precision tune.
    Robust to missing calibrators and variable meta-LR feature size.
    """
    mu_raw, sigma_raw = stats['mu_raw'], stats['sigma_raw']
    market_name = stats['market_name']; tid = stats['tid']
    Xf, Xt, Xm = stats['Xf'], stats['Xt'], stats['Xm']

    # raw p_up and z
    if sigma_raw <= 1e-9:
        p_up = 1.0 if (mu_raw - EPS_RET) > 0 else 0.0
        z = 0.0
    else:
        z = (mu_raw - EPS_RET) / max(1e-9, sigma_raw)
        p_up = _norm_cdf(z)

    # MC uncertainty (when needed)
    need_mc = (abs(p_up - 0.5) <= MC_TRIGGER_BAND)
    local_mc = _mc_samples(base, need_mc)
    p_unc = 0.0
    if local_mc > 0:
        pups = []
        for _ in range(local_mc):
            y2 = base['model']([Xf, Xt, Xm], training=True)
            y2 = y2.numpy() if hasattr(y2, "numpy") else np.asarray(y2)
            if y2.ndim == 1: y2 = y2.reshape(1,2)
            mu_s2, log_sigma_s2 = float(y2[0,0]), float(y2[0,1])
            sigma_s2 = max(_softplus(log_sigma_s2)+1e-6, 1e-6)
            # scale/center from price scaler:
            t_scaler = base['ticker_scalers'][tid]['price_scaler']
            scale  = getattr(t_scaler, 'scale_',  np.array([1.0], dtype=np.float32))[0]
            center = getattr(t_scaler, 'center_', np.array([0.0], dtype=np.float32))[0]
            mu_raw2 = mu_s2 * scale + center
            pups.append(_norm_cdf((mu_raw2 - EPS_RET) / (sigma_s2 * scale)))
        p_unc = float(np.std(np.asarray(pups, np.float32), ddof=0))

    # calibration (iso, meta-lr if available)
    iso = None
    try:
        iso = base.get('iso_cals', {}).get(int(tid))
    except Exception:
        iso = None
    p_iso = float(iso.transform([p_up])[0]) if iso is not None else float(p_up)

    lr  = None
    try:
        lr = base.get('meta_lrs', {}).get(int(tid))
    except Exception:
        lr = None

    if lr is not None:
        want = int(getattr(lr, 'n_features_in_', 5))
        if want >= 7:
            X_meta = np.array([[p_iso, z, sigma_raw, mu_raw, p_iso, p_unc, p_unc]], np.float32)
        else:
            X_meta = np.array([[p_iso, z, sigma_raw, mu_raw, p_unc]], np.float32)
        try:
            p_meta = float(lr.predict_proba(X_meta)[0,1])
        except Exception:
            p_meta = p_iso
    else:
        p_meta = p_iso

    # PSC
    try:
        pi_train = float(base.get('val_prev_map', {}).get(int(tid), 0.5))
    except Exception:
        pi_train = 0.5

    if USE_PSC and prior_rec.get('n', 0) >= PRIOR_MIN_N and prior_rec.get('na', 0) >= ACT_PREV_MIN_N:
        def _logit(x, eps=1e-6):
            x=float(np.clip(x,eps,1-eps)); return math.log(x/(1-x))
        delta = _logit(float(prior_rec.get('pi_target_ema', 0.5))) - _logit(pi_train)
        delta = float(np.clip(delta, -PSC_LOGIT_CAP, PSC_LOGIT_CAP))
        try:
            logit_p = math.log(p_meta/(1-p_meta))
            p_meta = 1.0 / (1.0 + math.exp(-(logit_p + delta)))
        except Exception:
            pass

    # Trend prior (light proxy using z)
    if USE_TREND_PRIOR:
        p_trend = _norm_cdf(TREND_KAPPA * z)
        w = _trend_weight(ticker, sigma_raw, market_name)
        p_meta = (1-w)*p_meta + w*p_trend

    # EMA smoothing
    base_alpha = _alpha_for(ticker, sigma_raw)
    tune = PRECISION_TUNE.get(ticker, {})
    alpha = float(tune.get('ema_alpha', base_alpha))
    ema_state = p_meta if (ema_prev is None or not USE_EMA_PROB) else (alpha*ema_prev + (1-alpha)*p_meta)
    p_use = float(np.clip(ema_state, 1e-4, 1-1e-4))

    # effective threshold
    thr_base = _thr_base_for(tid, ticker, base.get('thresholds', {}), market_name)
    thr_base = float(np.clip(thr_base + float(tune.get('thr_bump', 0.0)), THR_CLIP_LOW, THR_CLIP_HIGH))
    hivol_shift = HIVOL_THR_SHIFT_TH if market_name=='TH' else HIVOL_THR_SHIFT_US
    thr_eff = float(np.clip(thr_base + (hivol_shift if sigma_raw>=SIGMA_VOL_SPLIT else 0.0),
                            (THR_CLIP_LOW_TH if market_name=='TH' else THR_CLIP_LOW),
                            THR_CLIP_HIGH))

    # dynamic z_gate / uncertainty relief from precision_tune
    eff_unc_max = max(0.0, float(UNC_MAX - float(tune.get('unc_plus', 0.0))))
    eff_z_gate  = float(tune.get('z_gate', Z_GATE_ONLINE))

    return dict(p_use=p_use, p_unc=p_unc, thr_eff=thr_eff, z=z,
                ema_state=float(ema_state), eff_unc_max=eff_unc_max, eff_z_gate=eff_z_gate)

# ===================== Online one-step update =====================
def maybe_online_update_once_strict(base, state_mgr: OnlineStateManager,
                                    g_for_ticker: pd.DataFrame, ticker: str,
                                    us_mode: bool) -> bool:
    """
    Strict online update with training-like gates.
    Uses the last known (seq_len) as features and last row as target.
    """
    if not ONLINE_LEARNING_ENABLED:
        return False

    seq_len = base['seq_len']
    if len(g_for_ticker) < seq_len + 1:
        return False

    # states
    rec = state_mgr.get_t(ticker)
    step_ctr = int(rec.get("step_ctr", 0))
    updates_this_run = int(rec.get("updates_this_run", 0))
    ema_prev = rec.get("ema_state", None)
    prior_rec = {
        'pi_pred_ema': float(rec.get('pi_pred_ema', 0.5)),
        'pi_target_ema': float(rec.get('pi_target_ema', 0.5)),
        'n': int(rec.get('n', 0)),
        'na': int(rec.get('na', 0))
    }

    # pacing check
    step_ctr += 1
    every = ONLINE_UPDATE_EVERY_US if us_mode else ONLINE_UPDATE_EVERY
    max_per_run = ONLINE_UPDATE_MAX_PER_RUN_US if us_mode else ONLINE_UPDATE_MAX_PER_RUN
    do_pace = (step_ctr % every == 0) and (updates_this_run < max_per_run)
    if not do_pace:
        rec['step_ctr'] = step_ctr
        state_mgr.set_t(ticker, rec)
        return False

    # market policy
    market_name_last = str(g_for_ticker['Market'].iloc[-1]) if 'Market' in g_for_ticker.columns else 'OTHER'
    if not ALLOW_PRICE_ONLINE_MARKET.get(market_name_last, False):
        rec['step_ctr'] = step_ctr
        state_mgr.set_t(ticker, rec)
        return False

    # build hist/target
    hist = g_for_ticker.iloc[-(seq_len+1):-1].copy()
    target_row = g_for_ticker.iloc[-1]
    stats = compute_price_stats(base, hist, ticker)
    if stats is None:
        rec['step_ctr'] = step_ctr
        state_mgr.set_t(ticker, rec)
        return False

    # prob, thresholds, gates
    meta = compute_prob_meta(base, stats, ticker, ema_prev, prior_rec)

    pass_gate = True
    if CONF_GATE:
        conf = abs(meta['p_use'] - meta['thr_eff'])
        zgate = (Z_GATE_ONLINE_US if us_mode else meta['eff_z_gate'])
        pass_gate = (conf >= MARGIN) and (meta['p_unc'] <= meta['eff_unc_max']) and (abs(meta['z']) >= zgate)

    # update PSC predicted prevalence counters (n) each step
    prior_rec['pi_pred_ema'] = (1 - PRIOR_EMA_ALPHA) * prior_rec['pi_pred_ema'] + PRIOR_EMA_ALPHA * meta['p_use']
    prior_rec['n'] += 1

    if not pass_gate:
        rec.update({
            'step_ctr': step_ctr,
            'ema_state': meta['ema_state'],
            'pi_pred_ema': prior_rec['pi_pred_ema'],
            'pi_target_ema': prior_rec['pi_target_ema'],
            'n': prior_rec['n'],
            'na': prior_rec['na']
        })
        state_mgr.set_t(ticker, rec)
        return False

    # train_on_batch with true label
    last_close = float(hist['Close'].iloc[-1])
    actual_next = float(target_row['Close'])
    true_logret = float(np.clip(math.log(actual_next / max(1e-9, last_close)), -0.25, 0.25))

    tid = stats['tid']
    ps = base['ticker_scalers'][tid]['price_scaler']
    y_true_scaled = ps.transform(np.array([[true_logret]], np.float32))

    try:
        for _ in range(max(1, ONLINE_TRAIN_STEPS_PER_TICKER)):
            base['model'].train_on_batch([stats['Xf'], stats['Xt'], stats['Xm']], y_true_scaled)
    except Exception as e:
        print(f"⚠️ online update failed for {ticker}: {e}")
        rec.update({
            'step_ctr': step_ctr,
            'ema_state': meta['ema_state'],
            'pi_pred_ema': prior_rec['pi_pred_ema'],
            'pi_target_ema': prior_rec['pi_target_ema'],
            'n': prior_rec['n'],
            'na': prior_rec['na']
        })
        state_mgr.set_t(ticker, rec)
        return False

    # update PSC target prevalence (we observed actual_dir)
    actual_dir = int(actual_next > last_close)
    prior_rec['pi_target_ema'] = (1 - TARGET_EMA_ALPHA) * prior_rec['pi_target_ema'] + TARGET_EMA_ALPHA * float(actual_dir)
    prior_rec['na'] += 1

    updates_this_run += 1
    rec.update({
        'step_ctr': step_ctr,
        'updates_this_run': updates_this_run,
        'ema_state': meta['ema_state'],
        'pi_pred_ema': prior_rec['pi_pred_ema'],
        'pi_target_ema': prior_rec['pi_target_ema'],
        'n': prior_rec['n'],
        'na': prior_rec['na'],
        'last_online_date': str(pd.to_datetime(target_row['Date']).date()),
        'last_updated_at': datetime.now(ZoneInfo(LOCAL_TZ)).isoformat()
    })
    state_mgr.set_t(ticker, rec)
    return True

# ===================== Prediction helpers =====================
def ensure_feature_columns(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    out = df.copy()
    for c in feature_columns:
        if c not in out.columns:
            out[c] = 0.0
    out[feature_columns] = (out.groupby('Ticker')[feature_columns]
                              .apply(lambda g: g.fillna(method='ffill'))
                              .reset_index(level=0, drop=True))
    out[feature_columns] = out[feature_columns].fillna(0.0)
    return out

def base_predict_price_once(base, hist_df: pd.DataFrame, ticker: str) -> float | None:
    m = base['model']; seq_len = base['seq_len']; feats = base['feature_columns']
    tenc = base['ticker_encoder']; menc = base['market_encoder']; scalers = base['ticker_scalers']

    tid = int(tenc.transform([ticker])[0])
    if tid not in scalers: return None
    fs = scalers[tid]['feature_scaler']; ps = scalers[tid]['price_scaler']

    mk_name = str(hist_df['Market'].iloc[-1]) if 'Market' in hist_df.columns else 'OTHER'
    mid = _encode_market(menc, mk_name)

    Xf = fs.transform(hist_df[feats].values.astype(np.float32)).reshape(1, seq_len, -1)
    Xt = np.full((1, seq_len), tid, np.int32)
    Xm = np.full((1, seq_len), mid, np.int32)

    y = m([Xf, Xt, Xm], training=False)
    y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    if y.ndim == 1: y = y.reshape(1, 2)
    mu_s, log_sigma_s = float(y[0,0]), float(y[0,1])

    scale  = getattr(ps, 'scale_', [1.0])[0]
    center = getattr(ps, 'center_', [0.0])[0]
    mu_raw = mu_s * scale + center
    last_close = float(hist_df['Close'].iloc[-1])
    pred_price = float(last_close * math.exp(mu_raw))
    return pred_price

def meta_price_from_bases(booster, best_k: int, last_close: float,
                          lstm_price: float, gru_price: float, ref_date: pd.Timestamp) -> float:
    ret_lstm = math.log(max(1e-9, lstm_price)/max(1e-9, last_close))
    ret_gru  = math.log(max(1e-9, gru_price) /max(1e-9, last_close))
    ret_mean = 0.5*(ret_lstm + ret_gru)
    ret_diff = ret_lstm - ret_gru
    ema_mae_lstm = abs(lstm_price - last_close)
    ema_mae_gru  = abs(gru_price  - last_close)
    dow = pd.to_datetime(ref_date).weekday()
    dom = pd.to_datetime(ref_date).day
    feat = np.array([[ret_lstm, ret_gru, ret_mean, ret_diff,
                      ema_mae_lstm, ema_mae_gru, dow, dom]], np.float32)
    dmat = xgb.DMatrix(feat)
    ret_hat = float(booster.predict(dmat, iteration_range=(0, best_k))[0])
    return float(last_close * math.exp(ret_hat))

# ===================== MODES =====================
def run_nextday(conn, lstm_dir: str, gru_dir: str, meta_dir: str, market_filter: str | None):
    df_raw = fetch_latest_data(conn, market_filter=market_filter)
    if df_raw.empty:
        print("⚠️ ไม่มีข้อมูลจาก DB")
        return
    df_raw['Ticker'] = df_raw['StockSymbol'].astype(str)

    base_lstm = load_base_artifacts(lstm_dir)
    base_gru  = load_base_artifacts(gru_dir)
    try:
        meta_booster, meta_bestk, _, _ = load_meta_artifacts(meta_dir)
    except Exception:
        meta_booster = None
        meta_bestk   = None

    state_lstm = OnlineStateManager(base_lstm['model_dir'], "LSTM")
    state_gru  = OnlineStateManager(base_gru['model_dir'],  "GRU")

    # reset per-run counters
    for st in (state_lstm, state_gru):
        if "tickers" not in st.state or not isinstance(st.state["tickers"], dict):
            st.state["tickers"] = {}
        for t in list(st.state.get("tickers", {}).keys()):
            rec = st.state["tickers"][t]
            rec["updates_this_run"] = 0
            st.state["tickers"][t] = rec

    df_all = ensure_feature_columns(df_raw, base_lstm['feature_columns'])

    did_update_lstm = did_update_gru = False
    agg = {}
    for tkr, g in df_all.groupby('Ticker'):
        g = g.sort_values('Date').reset_index(drop=True)
        # strict online update once per ticker using last sequence
        us_mode = (str(g['Market'].iloc[-1]) == 'US') if 'Market' in g.columns else False
        if len(g) >= max(base_lstm['seq_len'], base_gru['seq_len']) + 1:
            if maybe_online_update_once_strict(base_lstm, state_lstm, g, tkr, us_mode):
                print(f"🧠 online-updated (LSTM): {tkr}")
                did_update_lstm = True
            if maybe_online_update_once_strict(base_gru, state_gru, g, tkr, us_mode):
                print(f"🧠 online-updated (GRU): {tkr}")
                did_update_gru = True

        # predict next calendar day
        if len(g) < max(base_lstm['seq_len'], base_gru['seq_len']):
            continue
        hist_lstm = g.iloc[-base_lstm['seq_len']:]
        hist_gru  = g.iloc[-base_gru['seq_len']:]
        last_close= float(g['Close'].iloc[-1])
        last_date = pd.to_datetime(g['Date'].iloc[-1])
        pred_date = (last_date + timedelta(days=1)).date()

        p_lstm = base_predict_price_once(base_lstm, hist_lstm, tkr)
        p_gru  = base_predict_price_once(base_gru,  hist_gru,  tkr)

        p_meta = None
        if (p_lstm is not None) and (p_gru is not None) and (meta_booster is not None):
            p_meta = meta_price_from_bases(meta_booster, meta_bestk, last_close, p_lstm, p_gru, last_date)

        agg[(tkr, pred_date)] = {
            'StockSymbol': tkr,
            'Date': pred_date.strftime('%Y-%m-%d'),
            'LSTM_Price': float(p_lstm or 0.0),
            'GRU_Price':  float(p_gru  or 0.0),
            'Ensemble_Price': float(p_meta or 0.0),
        }

    # persist online states + optional model weights
    if did_update_lstm:
        base_lstm['model'].save(os.path.join(base_lstm['model_dir'], "best_model_online.keras"))
        print(f"💾 saved weights -> {os.path.join(base_lstm['model_dir'], 'best_model_online.keras')}")
    if did_update_gru:
        base_gru['model'].save(os.path.join(base_gru['model_dir'], "best_model_online.keras"))
        print(f"💾 saved weights -> {os.path.join(base_gru['model_dir'], 'best_model_online.keras')}")

    state_lstm.save()
    state_gru.save()

    if agg:
        print(f"💾 บันทึกผลลัพธ์ {len(agg)} รายการลง DB ...")
        save_predictions_simple(pd.DataFrame(agg.values()), conn)
    else:
        print("⚠️ nextday: ไม่มีข้อมูลที่จะบันทึก")

def run_backfill(conn, lstm_dir: str, gru_dir: str, meta_dir: str,
                 start: str | None, end: str | None, tickers_csv: str | None, market_filter: str | None):
    df_raw = fetch_latest_data(conn, market_filter=market_filter)
    if df_raw.empty:
        print("⚠️ ไม่มีข้อมูลจาก DB")
        return

    if tickers_csv:
        keep = {t.strip().upper() for t in tickers_csv.split(",") if t.strip()}
        df_raw = df_raw[df_raw['StockSymbol'].str.upper().isin(keep)]
    df_raw['Ticker'] = df_raw['StockSymbol'].astype(str)

    base_lstm = load_base_artifacts(lstm_dir)
    base_gru  = load_base_artifacts(gru_dir)
    try:
        meta_booster, meta_bestk, _, _ = load_meta_artifacts(meta_dir)
    except Exception:
        meta_booster = None
        meta_bestk   = None

    state_lstm = OnlineStateManager(base_lstm['model_dir'], "LSTM")
    state_gru  = OnlineStateManager(base_gru['model_dir'],  "GRU")
    for st in (state_lstm, state_gru):
        if "tickers" not in st.state or not isinstance(st.state["tickers"], dict):
            st.state["tickers"] = {}
        for t in list(st.state.get("tickers", {}).keys()):
            rec = st.state["tickers"][t]
            rec["updates_this_run"] = 0
            st.state["tickers"][t] = rec

    df_all = ensure_feature_columns(df_raw, base_lstm['feature_columns'])
    start_d = pd.to_datetime(start).date() if start else None
    end_d   = pd.to_datetime(end).date()   if end   else None

    did_update_lstm = did_update_gru = False
    agg = {}
    max_seq = max(base_lstm['seq_len'], base_gru['seq_len'])

    for tkr, g in df_all.groupby('Ticker'):
        g = g.sort_values('Date').reset_index(drop=True)
        if len(g) <= max_seq:
            continue
        for i in range(max_seq, len(g) - 1):
            hist_lstm = g.iloc[i - base_lstm['seq_len']: i]
            hist_gru  = g.iloc[i - base_gru['seq_len'] : i]

            ref_date   = pd.to_datetime(g['Date'].iloc[i])
            pred_date  = pd.to_datetime(g['Date'].iloc[i + 1]).date()
            if (start_d and pred_date < start_d) or (end_d and pred_date > end_d):
                continue
            last_close = float(g['Close'].iloc[i])

            # strict online (walk-forward step i -> i+1)
            g_tmp = g.iloc[:i+1]  # up to ref i (has actual Close at i)
            us_mode = (str(g_tmp['Market'].iloc[-1]) == 'US') if 'Market' in g_tmp.columns else False
            if maybe_online_update_once_strict(base_lstm, state_lstm, g_tmp, tkr, us_mode):
                did_update_lstm = True
            if maybe_online_update_once_strict(base_gru,  state_gru,  g_tmp, tkr, us_mode):
                did_update_gru = True

            p_lstm = base_predict_price_once(base_lstm, hist_lstm, tkr)
            p_gru  = base_predict_price_once(base_gru,  hist_gru,  tkr)
            p_meta = None
            if (p_lstm is not None) and (p_gru is not None) and (meta_booster is not None):
                p_meta = meta_price_from_bases(meta_booster, meta_bestk, last_close, p_lstm, p_gru, ref_date)

            key = (tkr, pred_date)
            row = agg.get(key, {
                'StockSymbol': tkr,
                'Date': pred_date.strftime('%Y-%m-%d'),
                'LSTM_Price': 0.0,
                'GRU_Price': 0.0,
                'Ensemble_Price': 0.0,
            })
            if p_lstm is not None: row['LSTM_Price'] = float(p_lstm)
            if p_gru  is not None: row['GRU_Price']  = float(p_gru)
            if p_meta is not None: row['Ensemble_Price'] = float(p_meta)
            agg[key] = row

    if did_update_lstm:
        base_lstm['model'].save(os.path.join(base_lstm['model_dir'], "best_model_online.keras"))
        print(f"💾 saved weights -> {os.path.join(base_lstm['model_dir'], 'best_model_online.keras')}")
    if did_update_gru:
        base_gru['model'].save(os.path.join(base_gru['model_dir'], "best_model_online.keras"))
        print(f"💾 saved weights -> {os.path.join(base_gru['model_dir'], 'best_model_online.keras')}")

    state_lstm.save()
    state_gru.save()

    if agg:
        print(f"💾 บันทึกผลลัพธ์ {len(agg)} รายการลง DB ...")
        save_predictions_simple(pd.DataFrame(agg.values()), conn)
    else:
        print("⚠️ backfill: ไม่มีข้อมูลที่จะบันทึก")

def run_preopen(conn, lstm_dir: str, gru_dir: str, meta_dir: str, strict_window: bool = True):
    wins = current_preopen_windows()
    tz = ZoneInfo(LOCAL_TZ)
    now_dt = datetime.now(tz)
    now_t = now_dt.time()

    def in_window(win):
        return (now_t >= win["start"]) and (now_t <= win["end"])

    def fmt(t):  # time -> HH:MM
        return t.strftime("%H:%M")

    if in_window(wins["TH"]):
        market_filter = "Thailand"
        print(f"⏱️ ภายในช่วง PRE-OPEN ไทย {fmt(wins['TH']['start'])}-{fmt(wins['TH']['end'])} → ทำนายตลาดไทย")
    elif in_window(wins["US"]):
        market_filter = "America"
        print(f"⏱️ ภายในช่วง PRE-OPEN สหรัฐฯ {fmt(wins['US']['start'])}-{fmt(wins['US']['end'])} → ทำนายตลาดสหรัฐฯ")
    else:
        if strict_window:
            print(
                f"⏱️ นอกช่วง PRE-OPEN "
                f"(TH {fmt(wins['TH']['start'])}-{fmt(wins['TH']['end'])} / "
                f"US {fmt(wins['US']['start'])}-{fmt(wins['US']['end'])}) → ไม่รัน"
            )
            return
        # near-mode: choose the closest open within PREOPEN_NEAR_MIN minutes
        def minutes_diff(open_dt):
            return abs((open_dt - now_dt).total_seconds()) / 60.0

        th_diff = minutes_diff(wins["TH"]["open_dt"])
        us_diff = minutes_diff(wins["US"]["open_dt"])

        if min(th_diff, us_diff) <= PREOPEN_NEAR_MIN:
            market_filter = "Thailand" if th_diff <= us_diff else "America"
            print(f"⚠️ นอกกรอบ แต่ใกล้เวลาเปิด ({int(min(th_diff, us_diff))} นาที) → เลือก {market_filter}")
        else:
            # fallback: เช้าเลือก TH, บ่าย/ค่ำเลือก US
            market_filter = "Thailand" if now_t < dtime(12, 0) else "America"
            print(f"⚠️ ไม่อยู่ใกล้เวลาเปิด → fallback ตามช่วงเวลา → {market_filter}")

    run_nextday(conn, lstm_dir, gru_dir, meta_dir, market_filter=market_filter)

# ===================== MAIN =====================
import argparse

def main():
    parser = argparse.ArgumentParser(description="Price-only inference (preopen-first, training-parity online learning)")
    parser.add_argument(
        "--mode",
        choices=["nextday", "backfill", "preopen"],
        default=os.getenv("MODE", "preopen")  # default preopen
    )
    parser.add_argument("--strict-window", action="store_true",
                        default=os.getenv("STRICT_WINDOW","0").lower() in ("1","true","y"))
    parser.add_argument("--lstm-dir", default=LSTM_DIR_DEFAULT)
    parser.add_argument("--gru-dir",  default=GRU_DIR_DEFAULT)
    parser.add_argument("--meta-dir", default=META_DIR_DEFAULT)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end",   default=None)
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--market", choices=["all","th","us"], default="all")

    args = parser.parse_args()

    lstm_dir = args.lstm_dir
    gru_dir  = args.gru_dir
    meta_dir = args.meta_dir

    conn = get_mysql_conn()
    try:
        if args.mode == "nextday":
            mkt = None
            if args.market == "th": mkt = "Thailand"
            elif args.market == "us": mkt = "America"
            run_nextday(conn, lstm_dir, gru_dir, meta_dir, market_filter=mkt)
        elif args.mode == "backfill":
            mkt = None
            if args.market == "th": mkt = "Thailand"
            elif args.market == "us": mkt = "America"
            run_backfill(conn, lstm_dir, gru_dir, meta_dir, args.start, args.end, args.tickers, mkt)
        else:  # preopen (default)
            run_preopen(conn, lstm_dir, gru_dir, meta_dir, strict_window=args.strict_window)
    finally:
        try: conn.close()
        except: pass

if __name__ == "__main__":
    main()
