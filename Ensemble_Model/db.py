# -*- coding: utf-8 -*-
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text

# ---------- .env ----------
# โครงสร้างโปรเจกต์: <root> / Ensemble_Model / (ไฟล์นี้)
# และมี <root>/Preproces/config.env
ROOT = Path(__file__).resolve().parents[1]
dotenv_path = ROOT / 'Preproces' / 'config.env'
load_dotenv(dotenv_path)

# ---------- DB URL ----------
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
    raise RuntimeError("❌ กรุณาเช็ค .env: ต้องมี DB_USER, DB_PASSWORD, DB_HOST, DB_NAME (และ DB_PORT ถ้ามี)")

DB_CONNECTION = (
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?charset=utf8mb4"
)

# ---------- Engine ----------
def get_engine():
    # pool_pre_ping: กัน connection ตาย
    # pool_recycle: รีไซเคิลทุก 1 ชม.
    return create_engine(
        DB_CONNECTION,
        pool_pre_ping=True,
        pool_recycle=3600,
        future=True,
    )

# ---------- Helpers ----------
def _pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"ไม่พบคอลัมน์ใดๆ ในชุด: {candidates}")
    return None

def _to_date_str(series):
    # รับทั้ง string และ datetime → คืน 'YYYY-MM-DD'
    s = pd.to_datetime(series, errors='coerce')
    return s.dt.strftime('%Y-%m-%d')

def _to_nullable_float(v):
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None

def load_predictions(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    # strip ชื่อคอลัมน์กันเว้นวรรคหลงมา
    df.columns = df.columns.str.strip()
    return df

def save_predictions_to_db(pred_df: pd.DataFrame, batch_size: int = 1000):
    """
    ต้องมี (อย่างน้อย):
      - Ticker / หรือ StockSymbol
      - Date
      - Meta_Predicted_Price หรือ XGB_Predicted_Price
      - LSTM_Predicted_Price
      - GRU_Predicted_Price
    อัปเดตลงตาราง stockdetail: PredictionClose_LSTM, PredictionClose_GRU, PredictionClose_Ensemble
    ใช้ ON DUPLICATE KEY UPDATE (ต้องมี unique key (StockSymbol, date))
    """
    # แมพคอลัมน์ (ยืดหยุ่นชื่อ)
    col_symbol = _pick_col(pred_df, ["Ticker", "StockSymbol"])
    col_date   = _pick_col(pred_df, ["Date"])
    col_meta   = _pick_col(pred_df, ["Meta_Predicted_Price", "XGB_Predicted_Price", "PredictionClose_Ensemble"])
    col_lstm   = _pick_col(pred_df, ["LSTM_Predicted_Price", "Predicted_Price_LSTM", "PredictionClose_LSTM"])
    col_gru    = _pick_col(pred_df, ["GRU_Predicted_Price", "Predicted_Price_GRU", "PredictionClose_GRU"])

    # เตรียมข้อมูล
    df = pred_df.copy()
    df["_date_str"] = _to_date_str(df[col_date])

    # records สำหรับ executemany
    records = []
    for _, row in df.iterrows():
        records.append({
            "symbol": str(row[col_symbol]),
            "date":   row["_date_str"],
            "lstm":   _to_nullable_float(row[col_lstm]),
            "gru":    _to_nullable_float(row[col_gru]),
            "meta":   _to_nullable_float(row[col_meta]),
        })

    # ลบเรคอร์ดที่ไม่มี symbol หรือ date
    records = [r for r in records if r["symbol"] and r["date"]]

    if not records:
        print("⚠️ ไม่มีเรคอร์ดที่พร้อมอัปเดต")
        return

    sql = text("""
        INSERT INTO stockdetail (`StockSymbol`, `date`, `PredictionClose_LSTM`, `PredictionClose_GRU`, `PredictionClose_Ensemble`)
        VALUES (:symbol, :date, :lstm, :gru, :meta)
        AS new
        ON DUPLICATE KEY UPDATE
            `PredictionClose_LSTM` = new.`PredictionClose_LSTM`,
            `PredictionClose_GRU` = new.`PredictionClose_GRU`,
            `PredictionClose_Ensemble` = new.`PredictionClose_Ensemble`;
    """)

    engine = get_engine()
    inserted = 0
    with engine.begin() as conn:
        # แบ่ง batch เพื่อประสิทธิภาพ
        for i in range(0, len(records), batch_size):
            chunk = records[i:i+batch_size]
            conn.execute(sql, chunk)
            inserted += len(chunk)

    print(f"✅ upsert เสร็จ: {inserted} rows → stockdetail (PredictionClose_*).")

# ---------- main ----------
if __name__ == "__main__":
    # เปลี่ยนเป็นไฟล์ที่เราได้จาก XGB meta
    file_path = "./meta_price_predictions.csv"   # เดิมคุณใช้ ensemble_predictions.csv
    if not Path(file_path).exists():
        raise FileNotFoundError(f"ไม่พบไฟล์พยากรณ์: {file_path}")

    df_pred = load_predictions(file_path)
    print(f"📄 {file_path} columns = {list(df_pred.columns)}")
    save_predictions_to_db(df_pred)
