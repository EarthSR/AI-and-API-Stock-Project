import os
import io
import sys
import json
import math
import joblib
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

import sqlalchemy
from sqlalchemy import text

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

import ta

# ---------------------------------------------------------------------
# Console UTF-8
# ---------------------------------------------------------------------
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  # type: ignore
except Exception:
    pass

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "config.env")
load_dotenv(ENV_PATH)

# Model paths
LSTM_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "LSTM_model", "best_v6_plus_minimal_tuning_v2_final_model.keras"))
GRU_PATH  = os.path.abspath(os.path.join(BASE_DIR, "..", "GRU_Model",  "best_v6_plus_minimal_tuning_v2_final_model.keras"))
SCALER_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "LSTM_model", "ticker_scalers.pkl"))

# XGB meta (optional)
ENSEMBLE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Ensemble_Model"))
XGB_PKL = os.path.join(ENSEMBLE_DIR, "fixed_unified_trading_model.pkl")
XGB_JSON = os.path.join(ENSEMBLE_DIR, "xgb_meta.json")
ISO_CAL = os.path.join(ENSEMBLE_DIR, "meta_isotonic.joblib")

# Production toggles
ENABLE_MINI_RETRAIN = os.getenv("ENABLE_MINI_RETRAIN", "1") == "1"  # เปิดไว้เป็นค่าเริ่มต้น
DEBUG_PER_TICKER = os.getenv("DEBUG_PER_TICKER", "0") == "1"

# Mini-retrain policy
MINI_RETRAIN_EVERY_DAYS   = int(os.getenv("MINI_RETRAIN_EVERY_DAYS", 9))   # ทุก ๆ 9 วัน
MINI_RETRAIN_WINDOW_DAYS  = int(os.getenv("MINI_RETRAIN_WINDOW_DAYS", 120))# ใช้ข้อมูลย้อนหลัง N วัน
MINI_RETRAIN_MIN_SAMPLES  = int(os.getenv("MINI_RETRAIN_MIN_SAMPLES", 5))  # อย่างน้อย 5 ตัวอย่าง
MINI_RETRAIN_BATCH_SIZE   = int(os.getenv("MINI_RETRAIN_BATCH_SIZE", 32))
MINI_RETRAIN_EPOCHS       = int(os.getenv("MINI_RETRAIN_EPOCHS", 1))
MINI_RETRAIN_LR           = float(os.getenv("MINI_RETRAIN_LR", 1e-5))
SAVE_MINI_MODELS          = os.getenv("SAVE_MINI_MODELS", "1") == "1"

# Enforcement & capping
ENFORCE_DIR_BY_PRICE = True
APPLY_XGB_CAP = True
XGB_CAP_SCALE = float(os.getenv("XGB_CAP_SCALE", 1.20))
EPS_MAX = float(os.getenv("EPS_MAX", 0.02))
EPS_MIN = float(os.getenv("EPS_MIN", 0.002))
EPS_FRAC = float(os.getenv("EPS_FRAC", 0.40))
CALIB_STRENGTH = float(os.getenv("CALIB_STRENGTH", 1.15))

# ---------------------------------------------------------------------
# Policy version
# ---------------------------------------------------------------------
def build_policy_version() -> str:
    return (
        f"v{datetime.now().strftime('%Y%m%d')}|"
        f"ENF={int(ENFORCE_DIR_BY_PRICE)}|CAP={int(APPLY_XGB_CAP)}@{XGB_CAP_SCALE:.2f}|"
        f"EPS={EPS_MIN:.3f}/{EPS_FRAC:.2f}/{EPS_MAX:.3f}|CAL={CALIB_STRENGTH:.2f}|"
        f"MR={MINI_RETRAIN_EVERY_DAYS}d/{MINI_RETRAIN_WINDOW_DAYS}w/{MINI_RETRAIN_MIN_SAMPLES}+"
    )
POLICY_VERSION = build_policy_version()

# ---------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------
def build_engine():
    user = os.getenv("DB_USER")
    pw   = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    db   = os.getenv("DB_NAME")
    if not all([user, pw, host, db]):
        print("❌ DB env missing")
        sys.exit(1)
    url = f"mysql+mysqlconnector://{user}:{pw}@{host}/{db}"
    return sqlalchemy.create_engine(url)

# ---------------------------------------------------------------------
# Mini-retrain state (แยกไฟล์ตามตลาด)
# ---------------------------------------------------------------------
STATE_DIR = os.path.join(BASE_DIR, "mini_retrain_state")
os.makedirs(STATE_DIR, exist_ok=True)

def _state_path_for(market: str) -> str:
    code = {"Thailand": "th", "America": "us"}.get(str(market), str(market)).lower()
    return os.path.join(STATE_DIR, f"mini_retrain_state_{code}.json")

def load_state(path: str):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"tickers": {}}

def save_state(path: str, state: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _parse_date(dstr):
    if not dstr: return None
    try:
        return datetime.strptime(dstr, "%Y-%m-%d").date()
    except Exception:
        return None

def due_for_retrain(sym: str, state: dict, today: date, every_days: int) -> bool:
    info = state.get("tickers", {}).get(sym, {})
    last = _parse_date(info.get("last_retrain"))
    if last is None:
        return True  # ไม่เคยทำเลย
    return (today - last).days >= every_days

def mark_retrained(sym: str, state: dict, today: date):
    t = state.setdefault("tickers", {}).setdefault(sym, {"counter": 0, "last_retrain": None})
    t["counter"] = 0
    t["last_retrain"] = today.strftime("%Y-%m-%d")

# ---------------------------------------------------------------------
# Utils: custom loss (for load & mini-retrain)
# ---------------------------------------------------------------------
def focal_weighted_binary_crossentropy(class_weights, gamma=2.0, alpha_pos=0.7):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        eps = tf.constant(1e-7, tf.float32)
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        w = tf.where(tf.equal(y_true, 1.0),
                     tf.cast(class_weights.get(1,1.0), tf.float32),
                     tf.cast(class_weights.get(0,1.0), tf.float32))
        alpha = tf.where(tf.equal(y_true, 1.0),
                         tf.cast(alpha_pos, tf.float32),
                         tf.cast(1 - alpha_pos, tf.float32))
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1 - y_pred)
        focal = tf.pow(1 - pt, tf.cast(gamma, tf.float32))
        bce = tf.cast(tf.keras.losses.binary_crossentropy(y_true, y_pred), tf.float32)
        return tf.reduce_mean(bce * w * alpha * focal)
    return loss

@tf.keras.utils.register_keras_serializable()
def quantile_loss(y_true, y_pred, quantile=0.5):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    q = tf.cast(quantile, tf.float32)
    e = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))

def load_class_weights():
    p = os.path.abspath(os.path.join(BASE_DIR, "..", "LSTM_model", "class_weights.pkl"))
    try:
        import pickle
        with open(p, "rb") as f:
            cw = pickle.load(f)
        print(f"✅ Loaded class_weights: {cw}")
        return cw
    except Exception as e:
        print(f"⚠️ class_weights.pkl not found ({e}) → use balanced weights")
        return {0:1.0, 1:1.0}

# ---------------------------------------------------------------------
# Data prep & indicators
# ---------------------------------------------------------------------
FEATURE_COLUMNS = [
    'Open','High','Low','Close','Volume',
    'Change (%)','Sentiment','positive_news','negative_news','neutral_news',
    'Total Revenue','QoQ Growth (%)','Earnings Per Share (EPS)','ROE (%)',
    'ATR','Keltner_High','Keltner_Low','Keltner_Middle','Chaikin_Vol',
    'Donchian_High','Donchian_Low','PSAR',
    'Net Profit Margin (%)','Debt to Equity','P/E Ratio','P/BV Ratio','Dividend Yield (%)',
    'RSI','EMA_10','EMA_20','MACD','MACD_Signal','Bollinger_High','Bollinger_Low','SMA_50','SMA_200'
]

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {
        'Changepercen': 'Change_Percent',
        'Change_Percent': 'Change_Percent',
        'TotalRevenue': 'Total Revenue',
        'QoQGrowth': 'QoQ Growth (%)',
        'EPS': 'Earnings Per Share (EPS)',
        'ROE': 'ROE (%)',
        'NetProfitMargin': 'Net Profit Margin (%)',
        'DebtToEquity': 'Debt to Equity',
        'PERatio': 'P/E Ratio',
        'P_BV_Ratio': 'P/BV Ratio',
        'Dividend_Yield': 'Dividend Yield (%)',
    }
    for old, new in mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if 'Sentiment' in df.columns:
        df['Sentiment'] = df['Sentiment'].map({'Positive':1,'Negative':-1,'Neutral':0}).fillna(0)
    if 'Change (%)' not in df.columns:
        if 'Change_Percent' in df.columns:
            df['Change (%)'] = pd.to_numeric(df['Change_Percent'], errors='coerce')
        else:
            df['Change (%)'] = df.groupby('StockSymbol')['Close'].pct_change().mul(100)
    for c in FEATURE_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    def _per_symbol(g: pd.DataFrame):
        g = g.sort_values('Date').copy()
        if 'Change (%)' not in g.columns:
            g['Change (%)'] = g['Close'].pct_change()*100
        try:
            g['RSI'] = ta.momentum.RSIIndicator(g['Close'], window=14).rsi()
        except Exception:
            g['RSI'] = 0.0
        g['EMA_10'] = g['Close'].ewm(span=10, adjust=False).mean()
        g['EMA_20'] = g['Close'].ewm(span=20, adjust=False).mean()
        g['SMA_50'] = g['Close'].rolling(50).mean()
        g['SMA_200'] = g['Close'].rolling(200).mean()
        ema12 = g['Close'].ewm(span=12, adjust=False).mean()
        ema26 = g['Close'].ewm(span=26, adjust=False).mean()
        g['MACD'] = ema12 - ema26
        g['MACD_Signal'] = g['MACD'].rolling(9).mean()
        try:
            atr = ta.volatility.AverageTrueRange(g['High'], g['Low'], g['Close'], window=14)
            g['ATR'] = atr.average_true_range()
        except Exception:
            g['ATR'] = 0.0
        try:
            bb = ta.volatility.BollingerBands(g['Close'], window=20, window_dev=2)
            g['Bollinger_High'] = bb.bollinger_hband()
            g['Bollinger_Low']  = bb.bollinger_lband()
        except Exception:
            g['Bollinger_High'] = 0.0
            g['Bollinger_Low'] = 0.0
        try:
            kel = ta.volatility.KeltnerChannel(g['High'], g['Low'], g['Close'], window=20, window_atr=10)
            g['Keltner_High']   = kel.keltner_channel_hband()
            g['Keltner_Low']    = kel.keltner_channel_lband()
            g['Keltner_Middle'] = kel.keltner_channel_mband()
        except Exception:
            g['Keltner_High'] = g['Keltner_Low'] = g['Keltner_Middle'] = 0.0
        try:
            diff = (g['High'] - g['Low']).ewm(span=10, adjust=False).mean()
            g['Chaikin_Vol'] = diff.pct_change(10) * 100
        except Exception:
            g['Chaikin_Vol'] = 0.0
        try:
            g['Donchian_High'] = g['High'].rolling(20).max()
            g['Donchian_Low']  = g['Low'].rolling(20).min()
        except Exception:
            g['Donchian_High'] = 0.0
            g['Donchian_Low'] = 0.0
        try:
            psar = ta.trend.PSARIndicator(g['High'], g['Low'], g['Close'], step=0.02, max_step=0.2)
            g['PSAR'] = psar.psar()
        except Exception:
            g['PSAR'] = 0.0
        for c in FEATURE_COLUMNS:
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors='coerce')
        g[FEATURE_COLUMNS] = g[FEATURE_COLUMNS].ffill().bfill().fillna(0)
        return g
    df = df.groupby("StockSymbol", group_keys=False).apply(_per_symbol).reset_index(drop=True)
    return df

def fetch_latest_data(engine: sqlalchemy.Engine) -> pd.DataFrame:
    q = """
        SELECT 
            sd.Date,
            sd.StockSymbol,
            s.Market,
            sd.OpenPrice  AS Open,
            sd.HighPrice  AS High,
            sd.LowPrice   AS Low,
            sd.ClosePrice AS Close,
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
        WHERE s.Market in ('America','Thailand')
          AND sd.Date >= CURDATE() - INTERVAL 370 DAY
        ORDER BY sd.StockSymbol, sd.Date;
    """
    df = pd.read_sql(q, engine)
    if df.empty:
        print("❌ No data returned from DB")
        return df
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=False)
    df = standardize_columns(df)

    filled = []
    for sym, g in df.groupby("StockSymbol", sort=False):
        g = g.sort_values("Date").copy()
        g['Date'] = pd.to_datetime(g['Date'], errors='coerce', utc=False)
        g['StockSymbol'] = g['StockSymbol'].astype(str)
        start = g['Date'].min()
        end   = g['Date'].max()
        if pd.isna(start) or pd.isna(end):
            continue
        tmp = pd.DataFrame({"Date": pd.date_range(start, end, freq='D'),
                            "StockSymbol": str(sym)})
        merged = pd.merge(tmp, g, on=["Date","StockSymbol"], how="left")
        if 'Market' in merged.columns:
            mval = g['Market'].dropna().iloc[-1] if g['Market'].notna().any() else None
            merged['Market'] = merged['Market'].ffill().bfill()
            if merged['Market'].isna().any():
                merged['Market'] = merged['Market'].fillna(mval if mval is not None else "America")
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
    df = pd.concat(filled, ignore_index=True)
    df = compute_indicators(df)
    need = list(set(['Open','High','Low','Close']).intersection(df.columns))
    if need:
        df = df.dropna(subset=need)
    df = df.ffill().bfill().fillna(0)
    return df

# ---------------------------------------------------------------------
# Scalers & alignment
# ---------------------------------------------------------------------
def load_scalers():
    if not os.path.exists(SCALER_PATH):
        print(f"❌ Scaler file missing: {SCALER_PATH}")
        sys.exit(1)
    sc = joblib.load(SCALER_PATH)
    print(f"✅ Loaded scalers for {len(sc)} tickers")
    return sc

def align_features_to_scaler(df_feat: pd.DataFrame, scaler) -> pd.DataFrame:
    df_feat = df_feat.copy()
    names = getattr(scaler, "feature_names_in_", None)
    if names is not None:
        for c in names:
            if c not in df_feat.columns:
                df_feat[c] = 0.0
        df_feat = df_feat.loc[:, list(names)]
        return df_feat
    for c in FEATURE_COLUMNS:
        if c not in df_feat.columns:
            df_feat[c] = 0.0
    return df_feat.loc[:, FEATURE_COLUMNS]

# ---------------------------------------------------------------------
# Load models (safe)
# ---------------------------------------------------------------------
def load_model_checked(path: str):
    custom = {
        "quantile_loss": quantile_loss,
        "focal_weighted_binary_crossentropy": focal_weighted_binary_crossentropy(load_class_weights())
    }
    try:
        model = tf.keras.models.load_model(path, custom_objects=custom, compile=False, safe_mode=False)
    except Exception as e:
        print(f"❌ load_model_checked error for {path}: {e}")
        return None, (None, None)
    try:
        n_in = len(model.inputs); n_out = len(model.outputs)
        ok = (n_in == 3 and n_out == 2)
        print(f"{'✅' if ok else '⚠️'} {os.path.basename(path)} signature: inputs={n_in}, outputs={n_out}")
        seq_len = model.inputs[0].shape[1] or 10
        n_feat  = model.inputs[0].shape[-1] or len(FEATURE_COLUMNS)
        return model, (int(seq_len), int(n_feat))
    except Exception:
        return model, (10, len(FEATURE_COLUMNS))

# ---------------------------------------------------------------------
# Inverse price (dynamic cap + candidates)
# ---------------------------------------------------------------------
def dynamic_cap_pct(row):
    close = float(row.get("Close", 0) or 0)
    atr   = float(row.get("ATR", 0) or 0)
    if close <= 0 or atr <= 0:
        return 0.05
    base = 2.5 * (atr / close)
    mkt  = str(row.get("Market", "")).lower()
    if mkt in ("america", "us"):  lo, hi = 0.02, 0.08
    elif mkt in ("thailand", "th"): lo, hi = 0.03, 0.12
    else: lo, hi = 0.02, 0.10
    return float(np.clip(base, lo, hi))

def apply_cap(cur_price: float, raw_price: float, cap_pct: float):
    lo = cur_price * (1.0 - cap_pct); hi = cur_price * (1.0 + cap_pct)
    clipped = float(np.clip(raw_price, lo, hi))
    return clipped, bool(raw_price < lo or raw_price > hi)

def price_from_return(cur_price: float, ret: float, mode: str):
    if mode == "log":     return cur_price * float(np.exp(ret))
    elif mode == "simple":return cur_price * float(1.0 + ret)
    else:                 return float(ret)

def choose_price_from_candidates(ticker, cur_price, inv_val, row_ctx):
    cap = dynamic_cap_pct(row_ctx)
    candidates = [
        ("absolute", inv_val, price_from_return(cur_price, inv_val, "absolute")),
        ("log",      inv_val, price_from_return(cur_price, inv_val, "log")),
        ("simple",   inv_val, price_from_return(cur_price, inv_val, "simple")),
    ]
    def score(mode, r, p):
        if not np.isfinite(p) or p <= 0: return -1e9
        atr = float(row_ctx.get("ATR", 0) or 0); pen = 0.0
        move = abs(p - cur_price)
        if atr > 0 and move > 8.0 * atr: pen -= (move/atr)
        pen -= 5.0 * max(0.0, abs(r) - 0.20)
        if mode == "absolute" and atr > 0 and move > 15.0 * atr: pen -= (move/atr)
        return -pen
    candidates.sort(key=lambda x: score(*x), reverse=True)
    mode, inv_r, raw_price = candidates[0]
    clipped, used_clip = apply_cap(cur_price, raw_price, cap)
    print(f"ℹ️ [{ticker}] used {('1+r' if mode=='simple' else 'exp(logret)' if mode=='log' else 'absolute')} (raw={inv_r:+.4f}) -> {clipped:.2f}{' [CLIPPED]' if used_clip else ''}")
    return clipped, mode, cap, used_clip, raw_price

# ---------------------------------------------------------------------
# Calibrator helpers (PATCH1)
# ---------------------------------------------------------------------
class SimpleCalibrator:
    def transform(self, p):
        arr = np.asarray(p, dtype=float).ravel()
        out = 0.5 + 0.9*(arr - 0.5)
        return np.clip(out, 0.0, 1.0)

class IsotonicDictCalibrator:
    def __init__(self, payload):
        self.x, self.y = self._extract_xy(payload)
        self.x = np.asarray(self.x, dtype=float).ravel()
        self.y = np.asarray(self.y, dtype=float).ravel()
        order = np.argsort(self.x); self.x = self.x[order]; self.y = self.y[order]
        self.x = np.clip(self.x, 0.0, 1.0); self.y = np.clip(self.y, 0.0, 1.0)
        self.y = np.maximum.accumulate(self.y)
        if self.x.size < 2: self.x = np.array([0.0, 1.0]); self.y = np.array([0.0, 1.0])
    def _extract_xy(self, obj):
        pairs = [("x","y"),("X","Y"),("X_thresholds_","y_thresholds_"),("X_","y_"),("thresholds","values"),("bin_edges","bin_values")]
        if isinstance(obj, dict):
            for a,b in pairs:
                if a in obj and b in obj:
                    xa, yb = obj[a], obj[b]
                    if isinstance(xa,(list,tuple,np.ndarray)) and isinstance(yb,(list,tuple,np.ndarray)) and len(xa)==len(yb)>=2:
                        return list(xa), list(yb)
        if isinstance(obj,(list,tuple)) and len(obj)==2:
            a,b = obj
            if isinstance(a,(list,tuple,np.ndarray)) and isinstance(b,(list,tuple,np.ndarray)) and len(a)==len(b)>=2:
                return list(a), list(b)
        return [0.0,1.0],[0.0,1.0]
    def transform(self, p):
        arr = np.asarray(p, dtype=float).ravel()
        out = np.interp(arr, self.x, self.y, left=self.y[0], right=self.y[-1])
        return np.clip(out, 0.0, 1.0)

class CallableWrapperCalibrator:
    def __init__(self, fn): self.fn = fn
    def transform(self, p):
        arr = np.asarray(p, dtype=float).ravel()
        out = [self.fn(float(x)) for x in arr]
        return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)

def make_calibrator(obj):
    try:
        if obj is None: return SimpleCalibrator()
        if hasattr(obj,'transform') and callable(getattr(obj,'transform')): return obj
        if isinstance(obj,(dict,list,tuple)): return IsotonicDictCalibrator(obj)
        if callable(obj): return CallableWrapperCalibrator(obj)
    except Exception as e:
        print(f"⚠️ make_calibrator failed: {e} → SimpleCalibrator")
    return SimpleCalibrator()

# ---------------------------------------------------------------------
# XGB meta (optional)
# ---------------------------------------------------------------------
class XGBMeta:
    def __init__(self):
        self.ready = False; self.model = None; self.calib = None; self._printed_calib_err = False
        self._load()
    def _load(self):
        try:
            if os.path.exists(XGB_PKL):
                self.model = joblib.load(XGB_PKL); self.ready = True; print("✅ Loaded XGB meta model")
            elif os.path.exists(XGB_JSON):
                import xgboost as xgb
                self.model = xgb.Booster(); self.model.load_model(XGB_JSON); self.ready = True; print("✅ Loaded XGB meta booster")
            else:
                print("⚠️ XGB meta model not found -> fallback"); self.ready = False
            if os.path.exists(ISO_CAL):
                raw = joblib.load(ISO_CAL); self.calib = make_calibrator(raw)
            else:
                print("⚠️ Isotonic calibrator not found → SimpleCalibrator"); self.calib = SimpleCalibrator()
        except Exception as e:
            print(f"⚠️ XGB load failed: {e}"); self.ready = False; self.model = None; self.calib = SimpleCalibrator()
    def _calibrate(self, p):
        try:
            if self.calib is None: val = float(np.clip(p,0,1))
            elif hasattr(self.calib,'transform'): val = float(np.clip(float(self.calib.transform([p])[0]),0,1))
            elif callable(self.calib): val = float(np.clip(float(self.calib(p)),0,1))
            else: val = float(np.clip(p,0,1))
        except Exception as e:
            if not self._printed_calib_err: print(f"⚠️ Calibrator transform failed: {e} → identity"); self._printed_calib_err=True
            val = float(np.clip(p,0,1))
        val = 0.5 + CALIB_STRENGTH*(val - 0.5)
        return float(np.clip(val,0,1))
    def predict(self, rows):
        out = []
        for r in rows:
            cur, pl, pg, dl, dg = float(r['cur']), float(r['lstm_price']), float(r['gru_price']), float(r['lstm_prob']), float(r['gru_prob'])
            x = np.array([ (pl-cur)/max(cur,1e-9), (pg-cur)/max(cur,1e-9), dl, dg, abs(pl-pg)/max(cur,1e-9) ], dtype=float).reshape(1,-1)
            if self.ready and hasattr(self.model,"predict"):
                try:
                    if hasattr(self.model,"get_params"):
                        pred_change = float(np.ravel(self.model.predict(x))[0]); p_up = (dl+dg)/2.0
                    else:
                        import xgboost as xgb
                        dm = xgb.DMatrix(x); pred_change = float(self.model.predict(dm)[0]); p_up=(dl+dg)/2.0
                except Exception:
                    pred_change = ((pl+pg)/2.0 - cur)/max(cur,1e-9); p_up=(dl+dg)/2.0
            else:
                pred_change = ((pl+pg)/2.0 - cur)/max(cur,1e-9); p_up=(dl+dg)/2.0
            p_up = self._calibrate(p_up)
            price_raw = cur*(1.0+pred_change)
            dir_prob_only = 1 if p_up>=0.5 else 0
            conf_raw = abs(p_up-0.5)*2.0
            out.append({
                "xgb_price_raw": price_raw, "xgb_dir_prob": dir_prob_only,
                "xgb_prob_raw": p_up, "xgb_conf_raw": conf_raw,
                "xgb_price": price_raw, "xgb_dir": dir_prob_only,
                "xgb_prob": p_up, "xgb_conf": conf_raw,
                "pred_change": pred_change,
            })
        return out

# ---------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------
def print_per_ticker_table(rows_sorted):
    print("📈 Per-ticker results (sorted by confidence)")
    print("Ticker          Cur       LSTM        GRU   XGB(raw)  XGB(cap)   Dir    Prob   Conf      Δ%   Cap   Score   Note")
    print("-"*130)
    for r in rows_sorted:
        t = r['ticker']; cur = r['cur']; lstm = r['lstm_price']; gru  = r['gru_price']
        xgb_raw = r.get('xgb_price_raw', r.get('xgb_price', float('nan')))
        xgb_cap = r.get('xgb_price_cap', r.get('xgb_price_raw', r.get('xgb_price', float('nan'))))
        dir_icon = "📈UP" if r['xgb_dir']==1 else "📉DOWN"
        prob = r['xgb_prob']; conf = r['xgb_conf']
        delta_pct = (xgb_cap/cur - 1.0) * 100.0 if cur>0 else 0.0
        cap_txt = f"{r['cap_pct']*100:.1f}%"
        score = r.get('score', abs(delta_pct/100.0) * conf)
        notes = []
        if r.get('lstm_clip'): notes.append('L')
        if r.get('gru_clip'):  notes.append('G')
        if r.get('xgb_clip'):  notes.append('X')
        note = 'CLIPPED('+','.join(notes)+')' if notes else ''
        print(f"{t:<12} {cur:>8.2f} {lstm:>10.2f} {gru:>10.2f} {xgb_raw:>10.2f} {xgb_cap:>10.2f}  {dir_icon:>6}  {prob:>5.3f}  {conf:>5.3f} {delta_pct:>7.2f}  {cap_txt:>5}  {score:>6.3f}  {note}")

# ---------------------------------------------------------------------
# Save predictions to DB (simple upsert for LSTM/GRU/Ensemble)
# ---------------------------------------------------------------------
def save_predictions_simple(predictions_df: pd.DataFrame, engine: sqlalchemy.Engine = None) -> bool:
    """
    บันทึกผลพยากรณ์ลงตาราง StockDetail
      - ถ้ามี (StockSymbol, Date) อยู่แล้ว -> UPDATE 6 คอลัมน์ prediction ทั้งหมด
      - ถ้าไม่มี -> INSERT แถวใหม่พร้อม 6 คอลัมน์ prediction ทั้งหมด
    """
    if predictions_df is None or predictions_df.empty:
        print("❌ ไม่มีข้อมูลพยากรณ์ที่จะบันทึก")
        return False

    try:
        use_engine = engine if engine is not None else build_engine()

        success_count = 0
        created_count = 0
        updated_count = 0

        with use_engine.begin() as conn:
            for _, row in predictions_df.iterrows():
                sym = str(row['StockSymbol'])
                dt  = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')

                exists = conn.execute(sqlalchemy.text("""
                    SELECT COUNT(*) FROM StockDetail
                    WHERE StockSymbol = :symbol AND Date = :date
                """), {'symbol': sym, 'date': dt}).scalar()

                params = {
                    'symbol': sym,
                    'date': dt,
                    'lstm_price': float(row.get('LSTM_Price', 0.0) or 0.0),
                    'lstm_trend': int(row.get('LSTM_Direction', 0) or 0),
                    'gru_price': float(row.get('GRU_Price', 0.0) or 0.0),
                    'gru_trend': int(row.get('GRU_Direction', 0) or 0),
                    'ensemble_price': float(row.get('Ensemble_Price', 0.0) or 0.0),
                    'ensemble_trend': int(row.get('Ensemble_Direction', 0) or 0),
                }

                if exists and int(exists) > 0:
                    conn.execute(sqlalchemy.text("""
                        UPDATE StockDetail
                        SET PredictionClose_LSTM = :lstm_price,
                            PredictionTrend_LSTM = :lstm_trend,
                            PredictionClose_GRU = :gru_price,
                            PredictionTrend_GRU = :gru_trend,
                            PredictionClose_Ensemble = :ensemble_price,
                            PredictionTrend_Ensemble = :ensemble_trend
                        WHERE StockSymbol = :symbol AND Date = :date
                    """), params)
                    updated_count += 1
                    print(f"✅ UPDATE {sym} @ {dt}")
                else:
                    conn.execute(sqlalchemy.text("""
                        INSERT INTO StockDetail
                            (StockSymbol, Date,
                             PredictionClose_LSTM, PredictionTrend_LSTM,
                             PredictionClose_GRU,  PredictionTrend_GRU,
                             PredictionClose_Ensemble, PredictionTrend_Ensemble)
                        VALUES
                            (:symbol, :date,
                             :lstm_price, :lstm_trend,
                             :gru_price,  :gru_trend,
                             :ensemble_price, :ensemble_trend)
                    """), params)
                    created_count += 1
                    print(f"✅ INSERT {sym} @ {dt}")

                success_count += 1

        print(f"\n💾 DB upsert done: {success_count} rows (new {created_count}, updated {updated_count})")
        return success_count > 0

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการบันทึก DB: {e}")
        import traceback; traceback.print_exc()
        return False

# ---------------------------------------------------------------------
# Mini-retrain (per market, per ticker)
# ---------------------------------------------------------------------
def compile_for_miniretrain(model: tf.keras.Model):
    class_weights = load_class_weights()
    model.compile(
        optimizer=Adam(learning_rate=MINI_RETRAIN_LR),
        loss={
            "dense": tf.keras.losses.Huber(),  # placeholder if names unknown; will override below
        },
    )
    # ระบุชื่อเลเยอร์เอาต์พุตจริง ๆ
    try:
        losses = {
            "price_output": tf.keras.losses.Huber(delta=0.75),
            "direction_output": focal_weighted_binary_crossentropy(class_weights, gamma=1.95)
        }
        model.compile(optimizer=Adam(learning_rate=MINI_RETRAIN_LR), loss=losses,
                      loss_weights={"price_output":0.39, "direction_output":0.61})
    except Exception:
        pass

def mini_retrain_for_ticker(symbol: str,
                            df_sym: pd.DataFrame,
                            ticker_id: int,
                            market_id: int,
                            feature_scaler,
                            price_scaler,
                            seq_len: int,
                            lstm_model: tf.keras.Model,
                            gru_model:  tf.keras.Model) -> int:
    """
    คืนจำนวนตัวอย่างที่ใช้เทรน (0 ถ้าไม่ทำ)
    """
    if df_sym.empty: return 0
    end_date = df_sym['Date'].max()
    start_date = end_date - pd.Timedelta(days=MINI_RETRAIN_WINDOW_DAYS)
    g = df_sym[df_sym['Date'].between(start_date, end_date)].sort_values("Date").copy()
    if len(g) < seq_len + 1: return 0

    # เตรียม X (feature scaled), Y_price (log-return scaled), Y_dir
    Xf_list = []; Xt_list=[]; Xm_list=[]; Yp_list=[]; Yd_list=[]
    vals = g[FEATURE_COLUMNS].values.astype(np.float32)
    vals = feature_scaler.transform(align_features_to_scaler(g[FEATURE_COLUMNS], feature_scaler).values.astype(np.float32))
    close = g['Close'].values.astype(np.float32)

    for i in range(len(g)-seq_len):
        last_close = float(close[i+seq_len-1])
        next_close = float(close[i+seq_len])
        # targets
        logret = math.log(max(next_close,1e-9)/max(last_close,1e-9))
        y_price_scaled = float(price_scaler.transform(np.array([[logret]], dtype=np.float32))[0,0])
        y_dir = 1.0 if next_close > last_close else 0.0

        Xf_list.append(vals[i:i+seq_len])
        Xt_list.append(np.full((seq_len,), ticker_id, dtype=np.int32))
        Xm_list.append(np.full((seq_len,), market_id, dtype=np.int32))
        Yp_list.append([y_price_scaled]); Yd_list.append([y_dir])

    Xf = np.asarray(Xf_list, dtype=np.float32)
    Xt = np.asarray(Xt_list, dtype=np.int32)
    Xm = np.asarray(Xm_list, dtype=np.int32)
    Yp = np.asarray(Yp_list, dtype=np.float32)
    Yd = np.asarray(Yd_list, dtype=np.float32)

    n = len(Xf)
    if n < MINI_RETRAIN_MIN_SAMPLES:
        return 0

    # compile (ถ้ายังไม่ compile)
    try:
        if not hasattr(lstm_model, "optimizer") or lstm_model.optimizer is None:
            compile_for_miniretrain(lstm_model)
        if not hasattr(gru_model, "optimizer") or gru_model.optimizer is None:
            compile_for_miniretrain(gru_model)
    except Exception:
        pass

    # เทรนเบา ๆ — 1 epoch, batch เล็ก
    lstm_model.fit([Xf, Xt, Xm], {"price_output":Yp, "direction_output":Yd},
                   epochs=MINI_RETRAIN_EPOCHS, batch_size=min(MINI_RETRAIN_BATCH_SIZE, n),
                   shuffle=False, verbose=0)
    gru_model.fit([Xf, Xt, Xm], {"price_output":Yp, "direction_output":Yd},
                  epochs=MINI_RETRAIN_EPOCHS, batch_size=min(MINI_RETRAIN_BATCH_SIZE, n),
                  shuffle=False, verbose=0)
    return n

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 เริ่มระบบทำนาย (Production) — Inference + XGB Meta (+ optional mini-retrain)")
    print(f"🛡️ Policy: {POLICY_VERSION}")

    # เลือกตลาดจากเวลารัน
    hr = datetime.now().hour
    if 8 <= hr < 18:
        market_filter = "Thailand"
        print("📊 กำลังประมวลผลตลาดหุ้นไทย (SET)...")
    elif 19 <= hr or hr < 5:
        market_filter = "America"
        print("📊 กำลังประมวลผลตลาดหุ้นอเมริกา (NYSE & NASDAQ)...")
    else:
        print("❌ ไม่อยู่ในช่วงเวลาทำการของตลาดหุ้นไทยหรืออเมริกา")
        sys.exit(0)

    # สถานะ mini-retrain (แยกไฟล์ตามตลาด)
    STATE_PATH = _state_path_for(market_filter)
    state = load_state(STATE_PATH)
    print(f"🗂️ Mini-retrain state file: {STATE_PATH}")

    engine = build_engine()
    print("✅ DB connection env OK")

    print(f"✅ LSTM found: {LSTM_PATH}" if os.path.exists(LSTM_PATH) else f"❌ LSTM missing: {LSTM_PATH}")
    print(f"✅ GRU found:  {GRU_PATH}"  if os.path.exists(GRU_PATH)  else f"❌ GRU missing:  {GRU_PATH}")
    print(f"✅ Scalers found: {SCALER_PATH}" if os.path.exists(SCALER_PATH) else f"❌ Scalers missing: {SCALER_PATH}")
    if not (os.path.exists(LSTM_PATH) and os.path.exists(GRU_PATH) and os.path.exists(SCALER_PATH)):
        sys.exit(1)

    ticker_scalers = load_scalers()

    lstm_model, (seq_len_lstm, n_feat_lstm) = load_model_checked(LSTM_PATH)
    gru_model,  (seq_len_gru,  n_feat_gru)  = load_model_checked(GRU_PATH)
    if (lstm_model is None) or (gru_model is None):
        sys.exit(1)

    # ดึงข้อมูล & กรองตลาด
    df = fetch_latest_data(engine)
    if df.empty:
        sys.exit(0)
    df = df[df['Market'] == market_filter].copy()
    if df.empty:
        print(f"❌ ไม่มีข้อมูลสำหรับตลาด {market_filter}")
        sys.exit(0)

    all_syms = df['StockSymbol'].unique().tolist()
    ticker_le = LabelEncoder().fit(all_syms)
    df['Ticker_ID'] = ticker_le.transform(df['StockSymbol'])
    df['Market_ID'] = df['Market'].map(lambda x: 0 if str(x).lower() in ("america","us") else 1 if str(x).lower() in ("thailand","th") else 2)

    print(f"✅ Data ready: {len(df)} rows, {df['StockSymbol'].nunique()} tickers (market={market_filter})")

    seq_len = seq_len_lstm or 10
    n_features = n_feat_lstm or len(FEATURE_COLUMNS)

    # ============= Mini-retrain phase (ก่อน inference) =============
    if ENABLE_MINI_RETRAIN:
        today = datetime.now().date()
        total_trained = 0
        for sym in all_syms:
            if not due_for_retrain(sym, state, today, MINI_RETRAIN_EVERY_DAYS):
                continue
            g = df[df['StockSymbol']==sym].sort_values("Date").copy()
            if len(g) < seq_len + 1:
                continue
            t_id = int(g.iloc[-1]['Ticker_ID'])
            sc = ticker_scalers.get(t_id) or ticker_scalers.get(sym)
            if not sc: continue
            feature_scaler = sc.get('feature_scaler'); price_scaler = sc.get('price_scaler')
            if feature_scaler is None or price_scaler is None: continue
            market_id = int(g.iloc[-1]['Market_ID'])
            n_used = mini_retrain_for_ticker(sym, g, t_id, market_id, feature_scaler, price_scaler,
                                             seq_len, lstm_model, gru_model)
            if n_used >= MINI_RETRAIN_MIN_SAMPLES:
                mark_retrained(sym, state, today)
                total_trained += 1
                print(f"🔄 mini-retrain {sym}: {n_used} samples")
        if total_trained > 0:
            save_state(STATE_PATH, state)
            # บันทึกโมเดล (optional)
            if SAVE_MINI_MODELS:
                stamp = today.strftime("%Y%m%d")
                lstm_out = os.path.join(os.path.dirname(LSTM_PATH), f"best_v6_plus_minimal_tuning_v2_final_model_mini_{market_filter}_{stamp}.keras")
                gru_out  = os.path.join(os.path.dirname(GRU_PATH),  f"best_v6_plus_minimal_tuning_v2_final_model_mini_{market_filter}_{stamp}.keras")
                try:
                    lstm_model.save(lstm_out)
                    gru_model.save(gru_out)
                    print(f"💾 Saved mini LSTM → {os.path.basename(lstm_out)}")
                    print(f"💾 Saved mini GRU  → {os.path.basename(gru_out)}")
                except Exception as e:
                    print(f"⚠️ save mini models failed: {e}")
        else:
            print("ℹ️ ไม่มีสัญลักษณ์ครบกำหนด mini-retrain วันนี้")
    else:
        print("ℹ️ Mini-retrain disabled (ENABLE_MINI_RETRAIN=0)")

    # ===================== Inference phase =======================
    latest = df.groupby("StockSymbol").tail(1).reset_index(drop=True)
    per_ticker_rows = []

    for sym in all_syms:
        g = df[df['StockSymbol']==sym].sort_values("Date")
        if len(g) < seq_len: continue
        t_id = int(g.iloc[-1]['Ticker_ID'])
        sc = ticker_scalers.get(t_id) or ticker_scalers.get(sym)
        if not sc: continue
        feature_scaler = sc.get('feature_scaler'); price_scaler   = sc.get('price_scaler')
        if feature_scaler is None or price_scaler is None: continue

        chunk = g.tail(seq_len).copy()
        feat_df = chunk.loc[:, [c for c in FEATURE_COLUMNS if c in chunk.columns]].copy()
        feat_df = align_features_to_scaler(feat_df, feature_scaler)
        feat_arr = feature_scaler.transform(feat_df.values.astype(np.float32))

        X_feat   = feat_arr.reshape(1, seq_len, -1).astype(np.float32)
        X_ticker = np.full((1, seq_len), t_id, dtype=np.int32)
        market_id = int(chunk.iloc[-1]['Market_ID'])
        X_market = np.full((1, seq_len), market_id, dtype=np.int32)
        cur_price = float(chunk.iloc[-1]['Close'])

        # Predict LSTM/GRU
        lstm_out = lstm_model.predict([X_feat, X_ticker, X_market], verbose=0)
        lstm_price_scaled = float(np.squeeze(lstm_out[0])); lstm_prob = float(np.squeeze(lstm_out[1]))
        gru_out  = gru_model.predict([X_feat, X_ticker, X_market], verbose=0)
        gru_price_scaled  = float(np.squeeze(gru_out[0]));  gru_prob  = float(np.squeeze(gru_out[1]))

        try: inv_lstm_raw = float(price_scaler.inverse_transform(np.array(lstm_price_scaled).reshape(-1,1))[0,0])
        except Exception: inv_lstm_raw = float(lstm_price_scaled)
        try: inv_gru_raw  = float(price_scaler.inverse_transform(np.array(gru_price_scaled).reshape(-1,1))[0,0])
        except Exception: inv_gru_raw  = float(gru_price_scaled)

        row_ctx = latest[latest['StockSymbol']==sym].iloc[0].to_dict()
        lstm_price, lstm_mode, cap_pct1, lstm_clip, lstm_raw_price = choose_price_from_candidates(sym, cur_price, inv_lstm_raw, row_ctx)
        gru_price,  gru_mode,  cap_pct2, gru_clip,  gru_raw_price  = choose_price_from_candidates(sym, cur_price, inv_gru_raw,  row_ctx)
        cap_pct = max(cap_pct1, cap_pct2)

        per_ticker_rows.append({
            "ticker": sym,
            "cur": cur_price,
            "lstm_price": lstm_price,
            "gru_price":  gru_price,
            "lstm_prob":  lstm_prob,
            "gru_prob":   gru_prob,
            "cap_pct": cap_pct,
            "lstm_clip": lstm_clip,
            "gru_clip":  gru_clip,
            "lstm_mode": lstm_mode, "gru_mode":  gru_mode,
            "lstm_raw_price": lstm_raw_price, "gru_raw_price":  gru_raw_price,
            "inv_lstm_raw": inv_lstm_raw, "inv_gru_raw":  inv_gru_raw,
        })

    # XGB meta (raw)
    meta = XGBMeta()
    meta_in = [{
        "cur": r["cur"], "lstm_price": r["lstm_price"], "gru_price":  r["gru_price"],
        "lstm_prob":  r["lstm_prob"], "gru_prob":   r["gru_prob"],
    } for r in per_ticker_rows]
    meta_out = meta.predict(meta_in)

    # Post-process: cap+enforce
    out_rows = []
    for i, r in enumerate(per_ticker_rows):
        m = meta_out[i]; cur = r['cur']
        xgb_price_raw = float(m['xgb_price_raw']); p_up_raw = float(m['xgb_prob_raw']); dir_raw = int(m['xgb_dir_prob'])
        cap_pct_xgb = r['cap_pct'] * float(XGB_CAP_SCALE)
        if APPLY_XGB_CAP: xgb_price_cap, xgb_clipped = apply_cap(cur, xgb_price_raw, cap_pct_xgb)
        else:             xgb_price_cap, xgb_clipped = xgb_price_raw, False

        p_up_final = p_up_raw
        if ENFORCE_DIR_BY_PRICE:
            eps = max(EPS_MIN, EPS_FRAC * max(abs(r['lstm_price']-cur), abs(r['gru_price']-cur))/max(cur,1e-9))
            eps = float(np.clip(eps, EPS_MIN, EPS_MAX))
            pred_dir_price = 1 if xgb_price_cap >= cur*(1.0+eps) else 0 if xgb_price_cap <= cur*(1.0-eps) else dir_raw
            if pred_dir_price != dir_raw:
                p_up_final = 0.5 + 0.5*(p_up_raw - 0.5) if pred_dir_price == 1 else 0.5 - 0.5*(p_up_raw - 0.5)
            dir_final = pred_dir_price
        else:
            dir_final = dir_raw
        conf_final = abs(p_up_final - 0.5) * 2.0
        delta_final = (xgb_price_cap/cur - 1.0) if cur>0 else 0.0
        score_final = abs(delta_final) * conf_final

        merged = {
            **r,
            "xgb_price_raw": xgb_price_raw, "xgb_price_cap": xgb_price_cap, "xgb_clip": xgb_clipped,
            "cap_pct_xgb": cap_pct_xgb,
            "xgb_prob_raw": p_up_raw, "xgb_prob": p_up_final, "xgb_conf": conf_final,
            "xgb_dir_raw": dir_raw, "xgb_dir": dir_final,
            "pred_change": m.get('pred_change', (xgb_price_raw/cur - 1.0) if cur>0 else 0.0),
            "policy_version": POLICY_VERSION, "score": score_final,
        }
        out_rows.append(merged)

    # ===== CSV (ยังเก็บเป็น T+1 ได้ถ้าต้องการส่งออก) =====
    csv_simple = []
    csv_detail = []
    up = down = 0
    conf_sum = 0.0

    for r in out_rows:
        sym = r["ticker"]; cur = r["cur"]
        price_raw = r["xgb_price_raw"]; price_cap = r["xgb_price_cap"]
        dir_f = r["xgb_dir"]; prob_f = r["xgb_prob"]; conf_f = r["xgb_conf"]
        delta_pct_final = (price_cap/cur - 1.0) if cur>0 else 0.0
        if dir_f == 1: up += 1
        else: down += 1
        conf_sum += conf_f

        # ไฟล์ CSV: จะใส่วัน T+1 ก็ได้ตามที่คุณใช้
        csv_simple.append({
            "Policy_Version": POLICY_VERSION, "StockSymbol": sym, "Current_Price": cur,
            "Predicted_Price": price_cap, "Predicted_Direction": dir_f,
            "Direction_Probability": prob_f, "Confidence": conf_f,
            "Score": abs(delta_pct_final) * conf_f,
            "Date": (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
        })
        csv_detail.append({
            "Policy_Version": POLICY_VERSION, "StockSymbol": sym, "Cur": cur,
            "LSTM_RawInverse": r["inv_lstm_raw"], "GRU_RawInverse": r["inv_gru_raw"],
            "LSTM_CandidateRaw": r["lstm_raw_price"], "GRU_CandidateRaw": r["gru_raw_price"],
            "LSTM_Mode": r["lstm_mode"], "GRU_Mode": r["gru_mode"],
            "LSTM_Price": r["lstm_price"], "GRU_Price": r["gru_price"],
            "LSTM_Clipped": int(r.get("lstm_clip", False)), "GRU_Clipped": int(r.get("gru_clip", False)),
            "XGB_Price_Raw": price_raw, "XGB_Price_Capped": price_cap, "XGB_Clipped": int(r.get("xgb_clip", False)),
            "XGB_Prob_Raw": r["xgb_prob_raw"], "XGB_Prob_Final": prob_f,
            "XGB_Dir_Raw": r["xgb_dir_raw"], "XGB_Dir_Final": dir_f, "XGB_Conf_Final": conf_f,
            "DeltaPct_Raw": (price_raw/cur - 1.0) if cur>0 else 0.0, "DeltaPct_Final": delta_pct_final,
            "Cap_pct_LSTM_GRU": r["cap_pct"], "Cap_pct_XGB": r["cap_pct_xgb"],
            "Score": abs(delta_pct_final) * conf_f,
            "Date": (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
        })

    pred_df = pd.DataFrame(csv_simple)
    det_df  = pd.DataFrame(csv_detail)
    pred_path = os.path.join(BASE_DIR, "latest_ensemble_predictions.csv")
    det_path  = os.path.join(BASE_DIR, "latest_ensemble_predictions_detailed.csv")
    pred_df.to_csv(pred_path, index=False)
    det_df.to_csv(det_path, index=False)
    print(f"💾 Saved {os.path.basename(pred_path)} ({len(pred_df)} rows)")
    print(f"💾 Saved {os.path.basename(det_path)}")

    avg_conf = conf_sum / max(len(out_rows), 1)
    print(f"📊 Summary: UP={up} DOWN={down} | Avg confidence={avg_conf:.3f}")
    rows_sorted = sorted(out_rows, key=lambda x: x['xgb_conf'], reverse=True)
    print_per_ticker_table(rows_sorted)

    # ===== บันทึกลง DB: ให้เป็น T+1 =====
    try:
        latest_map = latest.set_index('StockSymbol')['Date'].dt.date.to_dict()

        db_rows = []
        for r in out_rows:
            sym = r['ticker']
            d_base = latest_map.get(sym, datetime.now().date())   # วันล่าสุดจริงใน DB
            d = d_base + timedelta(days=1)                        # → T+1

            db_rows.append({
                'Date': d,
                'StockSymbol': sym,
                'LSTM_Price': float(r['lstm_price']),
                'LSTM_Direction': int(float(r['lstm_prob']) >= 0.5),
                'GRU_Price': float(r['gru_price']),
                'GRU_Direction': int(float(r['gru_prob']) >= 0.5),
                'Ensemble_Price': float(r['xgb_price_cap']),
                'Ensemble_Direction': int(r['xgb_dir']),
            })

        predictions_db_df = pd.DataFrame(db_rows)
        ok = save_predictions_simple(predictions_db_df, engine)
        print("💽 DB upsert status:", "OK" if ok else "FAILED")

    except Exception as e:
        print(f"⚠️ ข้ามการบันทึก DB เพราะพบข้อผิดพลาดในการเตรียมข้อมูล: {e}")

    print("✅ Done.")


