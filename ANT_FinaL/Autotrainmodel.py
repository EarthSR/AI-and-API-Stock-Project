import numpy as np
import pandas as pd
import pymysql
import joblib
import pandas_ta as ta
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder

# ✅ เชื่อมต่อฐานข้อมูล MySQL ด้วย SQLAlchemy
db_connection_str = "mysql+pymysql://trademine:trade789@10.10.50.62/TradeMine"
engine = create_engine(db_connection_str)

# ✅ โหลด Scaler และ Encoder
scaler_features = joblib.load("../LSTM_model/scaler_features.pkl")
scaler_target = joblib.load("../LSTM_model/scaler_target.pkl")
ticker_encoder = joblib.load("../LSTM_model/ticker_encoder.pkl")

# ✅ โหลด Weighted Stacking Model
weighted_stacking_price = joblib.load("../Ensemble_Model/weighted_stacking_price.pkl")
weighted_stacking_direction = joblib.load("../Ensemble_Model/weighted_stacking_direction.pkl")

# ✅ โหลด Base Models (XGBoost และ RandomForest)
xgb_model = joblib.load("../Ensemble_Model/xgb_model_price.pkl")
rf_model = joblib.load("../Ensemble_Model/rf_model_price.pkl")
xgb_dir_model = joblib.load("../Ensemble_Model/xgb_model_direction.pkl")
rf_dir_model = joblib.load("../Ensemble_Model/rf_model_direction.pkl")

def calculate_indicators(df):
    """ 📌 คำนวณ Indicators ใน Python """
    df = df.copy()

    print(f"📊 ก่อนคำนวณ Indicator ({df.shape[0]} rows)")

    # ✅ คำนวณ RSI
    df["RSI"] = ta.rsi(df["ClosePrice"], length=14)

    # ✅ คำนวณ EMA
    df["EMA_10"] = ta.ema(df["ClosePrice"], length=10)
    df["EMA_20"] = ta.ema(df["ClosePrice"], length=20)

    # ✅ คำนวณ SMA
    df["SMA_50"] = ta.sma(df["ClosePrice"], length=50)
    df["SMA_200"] = ta.sma(df["ClosePrice"], length=200)

    # ✅ คำนวณ MACD
    macd = ta.macd(df["ClosePrice"])
    df["MACD"] = macd["MACD_12_26_9"] if macd is not None else np.nan
    df["MACD_Signal"] = macd["MACDs_12_26_9"] if macd is not None else np.nan

    # ✅ คำนวณ Bollinger Bands
    bb = ta.bbands(df["ClosePrice"], length=20)
    df["Bollinger_High"] = bb["BBU_20_2.0"] if bb is not None else np.nan
    df["Bollinger_Low"] = bb["BBL_20_2.0"] if bb is not None else np.nan

    # ✅ คำนวณ ATR
    df["ATR"] = ta.atr(df["HighPrice"], df["LowPrice"], df["ClosePrice"], length=14)

    print(f"📊 หลังคำนวณ Indicator ({df.shape[0]} rows)")

    return df.fillna(method="bfill").reset_index(drop=True)

def safe_transform(encoder, ticker):
    """ 📌 แปลงค่า `Ticker` เป็น ID หากไม่พบให้คืน -1 """
    if ticker in encoder.classes_:
        return encoder.transform([ticker])[0]
    else:
        print(f"⚠️ หุ้น {ticker} ไม่มีใน `ticker_encoder`")
        return -1

def walk_forward_validation_multi_task(engine, scaler_features, scaler_target, ticker_encoder, seq_length=10):
    """ 📌 ทำ Walk-Forward Validation ใช้ Weighted Stacking Model """
    all_predictions = []

    # ✅ ดึงลิสต์หุ้นจากฐานข้อมูล
    query_tickers = "SELECT DISTINCT StockSymbol FROM Stock"
    tickers_df = pd.read_sql(query_tickers, engine)
    tickers = tickers_df["StockSymbol"].dropna().unique().tolist()

    for ticker in tickers:
        print(f"\n📈 Processing Ticker: {ticker}")

        # ✅ ตรวจสอบค่า ticker
        if not isinstance(ticker, str) or ticker.strip() == "":
            print(f"⚠️ ค่า `StockSymbol` ผิดพลาด ({ticker}) ข้ามหุ้นนี้ไป...")
            continue

        ticker_id_val = safe_transform(ticker_encoder, ticker)
        if ticker_id_val == -1:
            continue

        # ✅ ดึงข้อมูลราคาหุ้นย้อนหลัง **เพิ่มจำนวนวัน** (เพื่อคำนวณ Indicator)
        query_stock = f"""
            SELECT Date, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, 
                Changepercen, TotalRevenue, QoQGrowth, EPS, ROE, NetProfitMargin, 
                DebtToEquity, PERatio, Dividend_Yield 
            FROM StockDetail 
            WHERE StockSymbol = '{ticker}' 
            ORDER BY Date DESC 
            LIMIT {seq_length + 300}
        """
        df_ticker = pd.read_sql(query_stock, engine)
        print(f"🔍 {ticker}: ข้อมูลจาก SQL ก่อน Indicator: {df_ticker.shape}")  # Debug


        # ✅ ตรวจสอบข้อมูลที่ได้มา
        if df_ticker.empty or len(df_ticker) < seq_length:
            print(f"⚠️ Not enough data for {ticker}, skipping...")
            continue

        df_ticker = df_ticker.sort_values("Date").reset_index(drop=True)

        # ✅ คำนวณ Indicators บนข้อมูลย้อนหลังมากพอ
        df_ticker = calculate_indicators(df_ticker)

        # ✅ ใช้ **เฉพาะ 10 วันล่าสุด** สำหรับการทำนาย
        df_ticker = df_ticker.iloc[-seq_length:]

        # ✅ ตรวจสอบว่าข้อมูล 10 วันสุดท้ายมีครบทุก Indicator หรือไม่
        if df_ticker.isnull().sum().sum() > 0:
            print(f"⚠️ Missing values in indicators for {ticker}, skipping...")
            continue

        feature_columns = df_ticker.columns.tolist()
        features = df_ticker[feature_columns].values
        ticker_ids = np.full((seq_length,), ticker_id_val)

        # ✅ ตรวจสอบว่าข้อมูลไม่ว่างเปล่าหลังคำนวณ Indicators
        if df_ticker.empty or len(df_ticker) < seq_length:
            print(f"⚠️ Not enough data after Indicator Calculation for {ticker}, skipping...")
            continue

        # ✅ ใช้ข้อมูล 10 วันที่ผ่านมาเป็นอินพุต
        historical_data = df_ticker.iloc[-seq_length:]
        features = historical_data[feature_columns].values

        # ✅ ตรวจสอบว่า features ไม่ว่างเปล่า
        if features.shape[0] == 0:
            print(f"⚠️ Features array is empty for {ticker}, skipping...")
            continue

        ticker_ids = np.full((seq_length,), ticker_id_val)

        # ✅ สเกลข้อมูล Feature ก่อนเข้าโมเดล
        features_scaled = scaler_features.transform(features.reshape(-1, len(feature_columns)))
        X_features = features_scaled.reshape(1, seq_length, len(feature_columns))
        X_ticker = ticker_ids.reshape(1, seq_length)


        # ✅ ทำนายราคาปิดด้วย Base Models
        pred_price_xgb = xgb_model.predict(X_features.reshape(1, -1))[0]
        pred_price_rf = rf_model.predict(X_features.reshape(1, -1))[0]

        # ✅ ทำนายทิศทางด้วย Base Models
        pred_dir_xgb = xgb_dir_model.predict(X_features.reshape(1, -1))[0]
        pred_dir_rf = rf_dir_model.predict(X_features.reshape(1, -1))[0]

        # ✅ ย้อนกลับราคาจาก Scaler
        pred_price_xgb_actual = scaler_target.inverse_transform([[pred_price_xgb]])[0][0]
        pred_price_rf_actual = scaler_target.inverse_transform([[pred_price_rf]])[0][0]

        # ✅ ใช้ Weighted Stacking
        w_xgb, w_rf = 0.5, 0.5
        predicted_price = (w_xgb * pred_price_xgb_actual) + (w_rf * pred_price_rf_actual)
        predicted_dir = 1 if ((0.5 * pred_dir_xgb) + (0.5 * pred_dir_rf)) >= 0.5 else 0

        future_date = pd.to_datetime(df_ticker.iloc[-1]["Date"]) + pd.DateOffset(1)

        print(f"✅ ทำนายวันที่ {future_date}: Predicted Price: {predicted_price:.2f}, Direction: {'Up' if predicted_dir == 1 else 'Down'}")

        all_predictions.append({'Ticker': ticker, 'Date': future_date, 'Predicted_Price': predicted_price, 'Predicted_Dir': predicted_dir})

    print("\n✅ การทำนายเสร็จสมบูรณ์!")
    return pd.DataFrame(all_predictions)


predictions_df = walk_forward_validation_multi_task(engine, scaler_features, scaler_target, ticker_encoder, seq_length=10)
predictions_df.to_csv("weighted_stacking_predictions.csv", index=False)
print("\n✅ บันทึกผลลัพธ์ลงไฟล์ 'weighted_stacking_predictions.csv' สำเร็จ!")
