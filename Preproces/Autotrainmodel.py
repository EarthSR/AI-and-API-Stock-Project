# -*- coding: utf-8 -*-
"""
Inference ราคาล้วน (ไม่มีทิศทาง) สำหรับ 3 โมเดล: LSTM, GRU, META (XGBoost จาก LSTM+GRU)
- ใช้ MySQL + dotenv (ไม่ใช้ SQLAlchemy)
- มีเมนูโหมด 1/2/3:
  1) nextday  : ทำนายวันทำการถัดไปของทุกตลาด (US+TH)
  2) backfill : ทำนายย้อนหลังตามช่วงวันที่กำหนด (ถามค่าบนคอนโซล)
  3) preopen  : ทำนายเฉพาะตลาดที่กำลังจะเปิด "อีก 30 นาที"
                - ไทย     : 08:30 (เวลา Asia/Bangkok)
                - อเมริกา : 20:30 (เวลา Asia/Bangkok)
                ระบบจะดึงข้อมูลและทำนาย "เฉพาะหุ้นของตลาดนั้น" แล้วเขียนลง StockDetail

สิ่งที่ต้องมี:
- .env (เช่น ../config.env) เก็บ DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT
- อาร์ติแฟกต์โมเดล:
    {LSTM_DIR|GRU_DIR}/logs/models/
        - best_model_static.keras
        - serving_artifacts.pkl (ต้องมี: ticker_scalers, ticker_encoder, market_encoder, feature_columns)
        - production_model_config.json (มี seq_length)
    META_DIR/
        - xgb_price.json
        - xgb_price.meta.joblib (มี 'best_k','q_lo','q_hi')

คอลัมน์ที่เขียนคืน StockDetail (เฉพาะราคา):
    - PredictionClose_LSTM
    - PredictionClose_GRU
    - PredictionClose_Ensemble
"""

import os, sys, math, json, time, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
from datetime import datetime, time as dtime
from pandas.tseries.offsets import BDay
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import joblib
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- dotenv + mysql ---
from dotenv import load_dotenv
import mysql.connector
from datetime import timedelta

# ===================== CONFIG =====================
ROOT = Path(__file__).resolve().parent
DOTENV_PATH = ('config.env')
LOCAL_TZ = os.getenv("LOCAL_TZ", "Asia/Bangkok")  # ปรับโซนได้ผ่าน ENV
# โฟลเดอร์โมเดล (ปรับได้หรือแก้ในเมนู #4 ด้านล่าง)
LSTM_DIR_DEFAULT = os.path.join(ROOT, "..", "LSTM_model")
GRU_DIR_DEFAULT  = os.path.join(ROOT, "..", "GRU_model")
META_DIR_DEFAULT = os.path.join(ROOT, "..", "Ensemble_Model")

# Pre-open windows (เวลา Asia/Bangkok)
PREOPEN_WINDOWS = {
    "TH": {"start": dtime(8, 25, 0), "end": dtime(8, 40, 0), "db_market": "Thailand"},
    "US": {"start": dtime(20, 25, 0), "end": dtime(20, 40, 0), "db_market": "America"},
}

# ===================== DB (dotenv + mysql.connector) =====================
def get_mysql_conn() -> mysql.connector.connection.MySQLConnection:
    load_dotenv(DOTENV_PATH)
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

# ===================== Standardize + TA/features =====================
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
    """
    market_filter: None=US+TH, 'Thailand' เฉพาะไทย, 'America' เฉพาะอเมริกา
    """
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

    # เติมวัน/ffill ต่อหุ้น
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

# ===================== SAVE ราคาล้วน (MySQL) =====================
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

# ===================== โหลดโมเดล / infer ราคา =====================
def load_base_artifacts(model_root_dir: str):
    MODEL_DIR = os.path.join(model_root_dir, "logs", "models")
    def p(name): return os.path.join(MODEL_DIR, name)
    with open(p("production_model_config.json"), 'r', encoding='utf-8') as f:
        cfg_all = json.load(f)
    seq_len = int(cfg_all.get('model_config', {}).get('seq_length', 10))
    model = load_model(p("best_model_static.keras"), compile=False, safe_mode=False)
    artifacts = joblib.load(p("serving_artifacts.pkl"))
    return dict(
        seq_len=seq_len,
        model=model,
        ticker_scalers=artifacts['ticker_scalers'],
        ticker_encoder=artifacts['ticker_encoder'],
        market_encoder=artifacts['market_encoder'],
        feature_columns=artifacts['feature_columns']
    )

def base_predict_price_once(base, hist_df: pd.DataFrame, ticker: str) -> float | None:
    m = base['model']; seq_len = base['seq_len']; feats = base['feature_columns']
    tenc = base['ticker_encoder']; menc = base['market_encoder']; scalers = base['ticker_scalers']

    tid = int(tenc.transform([ticker])[0])
    if tid not in scalers: return None
    fs = scalers[tid]['feature_scaler']
    ps = scalers[tid]['price_scaler']

    if 'Market' in hist_df.columns:
        mk_name = str(hist_df['Market'].iloc[-1])
        try: mid = int(menc.transform([mk_name])[0])
        except: mid = 0
    else:
        mid = 0

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

def load_meta_artifacts(meta_dir: str):
    model_path = os.path.join(meta_dir, "xgb_price.json")
    meta_path  = os.path.join(meta_dir, "xgb_price.meta.joblib")
    booster = xgb.Booster(); booster.load_model(model_path)
    meta = joblib.load(meta_path)
    best_k = int(meta.get('best_k', 200))
    q_lo = float(meta.get('q_lo', -0.05)); q_hi = float(meta.get('q_hi', 0.05))
    return booster, best_k, q_lo, q_hi

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

# ===================== โหมดทำงาน =====================
def run_nextday(conn, lstm_dir: str, gru_dir: str, meta_dir: str, market_filter: str | None):
    """
    market_filter: None=US+TH, 'Thailand' เฉพาะไทย, 'America' เฉพาะเมกา
    ทำนาย 'วันปฏิทินถัดไป' (+1 day) ของแต่ละหุ้น ตามความต้องการให้ไม่ข้ามเสาร์-อาทิตย์
    """
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

    df_all = ensure_feature_columns(df_raw, base_lstm['feature_columns'])

    agg = {}  # key=(ticker,pred_date) -> row
    for tkr, g in df_all.groupby('Ticker'):
        g = g.sort_values('Date').reset_index(drop=True)
        if len(g) < max(base_lstm['seq_len'], base_gru['seq_len']):
            continue

        hist_lstm = g.iloc[-base_lstm['seq_len']:]
        hist_gru  = g.iloc[-base_gru['seq_len']:]
        last_close= float(g['Close'].iloc[-1])
        last_date = pd.to_datetime(g['Date'].iloc[-1])
        # ใช้ "วันปฏิทิน" ถัดไป ไม่ใช่ business day
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

    if agg:
        print(f"💾 บันทึกผลลัพธ์ {len(agg)} รายการลง DB ...")
        save_predictions_simple(pd.DataFrame(agg.values()), conn)
    else:
        print("⚠️ nextday: ไม่มีข้อมูลที่จะบันทึก")


def run_backfill(conn, lstm_dir: str, gru_dir: str, meta_dir: str,
                 start: str | None, end: str | None, tickers_csv: str | None, market_filter: str | None):
    """
    ทำนายย้อนหลังโดยผูกวันที่ทำนาย = วันที่แถวถัดไปใน DB ของ ticker เดียวกัน (ไม่ข้ามวัน)
    start/end คือช่วงของ 'วันที่ทำนาย' (pred_date) สำหรับการกรอง
    """
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

    df_all = ensure_feature_columns(df_raw, base_lstm['feature_columns'])
    start_d = pd.to_datetime(start).date() if start else None
    end_d   = pd.to_datetime(end).date()   if end   else None

    agg = {}  # key=(ticker,pred_date) -> row
    max_seq = max(base_lstm['seq_len'], base_gru['seq_len'])

    for tkr, g in df_all.groupby('Ticker'):
        g = g.sort_values('Date').reset_index(drop=True)
        if len(g) <= max_seq:
            continue

        # เดินจาก index i (วันอ้างอิง) → ทำนายวันถัดไปใน DB คือ i+1
        for i in range(max_seq, len(g) - 1):
            hist_lstm = g.iloc[i - base_lstm['seq_len']: i]
            hist_gru  = g.iloc[i - base_gru['seq_len'] : i]

            ref_date   = pd.to_datetime(g['Date'].iloc[i])      # วันอ้างอิง
            pred_date  = pd.to_datetime(g['Date'].iloc[i + 1]).date()  # วันถัดไป "ตาม DB"
            if (start_d and pred_date < start_d) or (end_d and pred_date > end_d):
                continue

            last_close = float(g['Close'].iloc[i])

            p_lstm = base_predict_price_once(base_lstm, hist_lstm, tkr)
            p_gru  = base_predict_price_once(base_gru,  hist_gru,  tkr)

            p_meta = None
            if (p_lstm is not None) and (p_gru is not None) and (meta_booster is not None):
                p_meta = meta_price_from_bases(meta_booster, meta_bestk, last_close, p_lstm, p_gru, ref_date)

            key = (tkr, pred_date)
            row = agg.get(key)
            if row is None:
                row = {
                    'StockSymbol': tkr,
                    'Date': pred_date.strftime('%Y-%m-%d'),
                    'LSTM_Price': 0.0,
                    'GRU_Price': 0.0,
                    'Ensemble_Price': 0.0,
                }

            if p_lstm is not None:
                row['LSTM_Price'] = float(p_lstm)
            if p_gru is not None:
                row['GRU_Price']  = float(p_gru)
            if p_meta is not None:
                row['Ensemble_Price'] = float(p_meta)

            agg[key] = row

    if agg:
        print(f"💾 บันทึกผลลัพธ์ {len(agg)} รายการลง DB ...")
        save_predictions_simple(pd.DataFrame(agg.values()), conn)
    else:
        print("⚠️ backfill: ไม่มีข้อมูลที่จะบันทึก")


def run_preopen(conn, lstm_dir: str, gru_dir: str, meta_dir: str, strict_window: bool = True):
    """
    ตรวจเวลาปัจจุบัน (Asia/Bangkok โดยค่าเริ่มต้น) แล้วเลือกตลาด:
        - TH: 08:25–08:40 → ใช้ market_filter='Thailand'
        - US: 20:25–20:40 → ใช้ market_filter='America'
    ถ้า strict_window=True และอยู่นอกหน้าต่างเวลาดังกล่าว จะไม่รัน
    """
    now = datetime.now(ZoneInfo(LOCAL_TZ)).time()

    def in_window(win):
        s, e = win["start"], win["end"]
        return (now >= s) and (now <= e)

    market_filter = None
    if in_window(PREOPEN_WINDOWS["TH"]):
        market_filter = "Thailand"
        print("⏱️ ภายในหน้าต่าง PRE-OPEN ไทย → ทำนายเฉพาะตลาดไทย")
    elif in_window(PREOPEN_WINDOWS["US"]):
        market_filter = "America"
        print("⏱️ ภายในหน้าต่าง PRE-OPEN สหรัฐฯ → ทำนายเฉพาะตลาดสหรัฐฯ")
    else:
        if strict_window:
            print("⏱️ ตอนนี้อยู่นอกช่วงเวลา PRE-OPEN (TH 08:25–08:40 / US 20:25–20:40) → ไม่รัน")
            return
        # ถ้าไม่ strict ให้เลือกตลาดถัดไปแบบ heuristic
        market_filter = "Thailand" if now < dtime(12,0,0) else "America"
        print(f"⚠️ นอกช่วง PRE-OPEN → เลือกตลาดโดยประมาณ: {market_filter}")

    run_nextday(conn, lstm_dir, gru_dir, meta_dir, market_filter=market_filter)

# ===================== MENU (1 / 2 / 3) =====================
def main():
    print("\n=== PRICE-ONLY INFERENCE (MySQL + dotenv) ===")
    print("  1) nextday  : ทำนายวันทำการถัดไป (ทุกตลาด)")
    print("  2) backfill : ทำนายย้อนหลัง (กรอกช่วงวันที่)")
    print("  3) preopen  : ทำนายเฉพาะตลาดที่กำลังจะเปิดอีก 30 นาที (TH 08:30 / US 20:30)")
    print("  4) ตั้งค่า path โมเดล (ปัจจุบันใช้ค่าเริ่มต้น)")
    choice = input("เลือกโหมด (1/2/3/4): ").strip()

    # paths
    lstm_dir = LSTM_DIR_DEFAULT
    gru_dir  = GRU_DIR_DEFAULT
    meta_dir = META_DIR_DEFAULT

    if choice == "4":
        lstm_dir = input(f"ใส่ LSTM dir [{LSTM_DIR_DEFAULT}]: ").strip() or LSTM_DIR_DEFAULT
        gru_dir  = input(f"ใส่ GRU  dir [{GRU_DIR_DEFAULT}]: ").strip() or GRU_DIR_DEFAULT
        meta_dir = input(f"ใส่ META dir [{META_DIR_DEFAULT}]: ").strip() or META_DIR_DEFAULT
        print("\nตั้งค่าเสร็จ ✓\n")
        choice = input("เลือกโหมด (1/2/3): ").strip()

    conn = get_mysql_conn()
    try:
        if choice == "1":
            # ทุกตลาด
            run_nextday(conn, lstm_dir, gru_dir, meta_dir, market_filter=None)
        elif choice == "2":
            start = input("เริ่ม (YYYY-MM-DD) [ว่าง=ทั้งหมด]: ").strip() or None
            end   = input("จบ   (YYYY-MM-DD) [ว่าง=ทั้งหมด]: ").strip() or None
            tks   = input("ระบุหุ้น (คอมม่า) [ว่าง=ทั้งหมด]: ").strip() or None
            mopt  = input("จำกัดตลาด? (all/th/us) [all]: ").strip().lower() or "all"
            market_filter = None
            if mopt == "th": market_filter = "Thailand"
            elif mopt == "us": market_filter = "America"
            run_backfill(conn, lstm_dir, gru_dir, meta_dir, start, end, tks, market_filter)
        elif choice == "3":
            strict = input("บังคับให้อยู่ในหน้าต่างเวลาเป๊ะ? (y/N): ").strip().lower() == "y"
            run_preopen(conn, lstm_dir, gru_dir, meta_dir, strict_window=strict)
        else:
            print("❌ ไม่พบโหมดที่เลือก")
    finally:
        try: conn.close()
        except: pass

if __name__ == "__main__":
    main()
