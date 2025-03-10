import numpy as np
import pandas as pd
import pymysql
import joblib
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable

# ✅ โหลด Feature Columns และ Scaler
feature_columns = joblib.load("../LSTM_model/feature_columns.pkl")
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

def walk_forward_validation_multi_task(
    db_connection, feature_columns, scaler_features, scaler_target, ticker_encoder, seq_length=10
):
    """
    ทำ Walk-Forward Validation โดยใช้ Weighted Stacking Model
    อัปเดต `PredictionPrice` และ `PredictionTrend` ในฐานข้อมูล MySQL
    
    Args:
        db_connection: การเชื่อมต่อฐานข้อมูล MySQL
        feature_columns: ชื่อฟีเจอร์ที่ใช้
        scaler_features: Scaler ของฟีเจอร์
        scaler_target: Scaler ของ Price
        ticker_encoder: LabelEncoder ของ Ticker
        seq_length: ความยาว sequence
    """
    all_predictions = []
    
    # ✅ ดึงลิสต์หุ้นทั้งหมดจาก Stock
    query_tickers = "SELECT DISTINCT StockSymbol FROM Stock"
    tickers = pd.read_sql(query_tickers, db_connection)['StockSymbol'].tolist()

    for ticker in tickers:
        print(f"\n📈 Processing Ticker: {ticker}")
        ticker_id_val = ticker_encoder.transform([ticker])[0]

        # ✅ ดึงข้อมูลย้อนหลัง 10 วันล่าสุดจาก StockDetail
        query_stock = f"""
            SELECT Date, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, TotalRevenue, QoQGrowth, EPS,ROE, NetProfitMarginNetProfitMargin,DebtToEquity,PERatio,
            FROM StockDetail 
            WHERE StockSymbol = '{ticker}' 
            ORDER BY Date DESC 
            LIMIT {seq_length}
        """
        df_ticker = pd.read_sql(query_stock, db_connection).sort_values('Date').reset_index(drop=True)

        if len(df_ticker) < seq_length:
            print(f"⚠️ Not enough data for {ticker}, skipping...")
            continue

        # ✅ ใช้ข้อมูล 10 วันที่ผ่านมาเป็นอินพุต
        historical_data = df_ticker.iloc[-seq_length:]
        features = historical_data[feature_columns].values
        ticker_ids = np.full((seq_length,), ticker_id_val)

        # ✅ สเกลข้อมูล Feature ก่อนเข้าโมเดล
        features_scaled = scaler_features.transform(features)
        X_features = features_scaled.reshape(1, seq_length, len(feature_columns))
        X_ticker = ticker_ids.reshape(1, seq_length)

        # ✅ ทำนายราคาปิดด้วย Base Models
        pred_price_xgb = xgb_model.predict(X_features.reshape(1, -1))[0]
        pred_price_rf = rf_model.predict(X_features.reshape(1, -1))[0]

        # ✅ ทำนายทิศทางด้วย Base Models
        pred_dir_xgb = xgb_dir_model.predict(X_features.reshape(1, -1))[0]
        pred_dir_rf = rf_dir_model.predict(X_features.reshape(1, -1))[0]

        # ✅ ย้อนกลับราคาจาก Scaler
        pred_price_xgb_actual = scaler_target.inverse_transform(np.array([[pred_price_xgb]]))[0][0]
        pred_price_rf_actual = scaler_target.inverse_transform(np.array([[pred_price_rf]]))[0][0]

        # ✅ ใช้ Weighted Stacking
        w_lstm, w_gru, w_xgb, w_rf = 0.3, 0.3, 0.2, 0.2
        predicted_price = (w_xgb * pred_price_xgb_actual) + (w_rf * pred_price_rf_actual)
        predicted_dir = (0.2 * pred_dir_xgb) + (0.2 * pred_dir_rf)
        predicted_dir = 1 if predicted_dir >= 0.5 else 0

        # ✅ ดึงข้อมูลวันที่ล่าสุด + 1 วัน
        future_date = pd.to_datetime(df_ticker.iloc[-1]['Date']) + pd.DateOffset(1)

        print(f"✅ ทำนายวันที่ {future_date}:")
        print(f"   - Predicted Price: {predicted_price:.2f}")
        print(f"   - Predicted Direction: {'Up' if predicted_dir == 1 else 'Down'}")

        # ✅ ตรวจสอบว่ามีแถวของ `future_date` ในฐานข้อมูลหรือไม่
        check_query = f"""
            SELECT COUNT(*) FROM StockDetail 
            WHERE StockSymbol = '{ticker}' 
            AND Date = '{future_date.strftime('%Y-%m-%d')}'
        """
        cursor = db_connection.cursor()
        cursor.execute(check_query)
        count = cursor.fetchone()[0]

        if count > 0:
            # ✅ อัปเดต `PredictionPrice` และ `PredictionTrend`
            update_query = f"""
                UPDATE StockDetail 
                SET PredictionPrice = {predicted_price}, PredictionTrend = {predicted_dir}
                WHERE StockSymbol = '{ticker}' 
                AND Date = '{future_date.strftime('%Y-%m-%d')}'
            """
            cursor.execute(update_query)
            db_connection.commit()
            print(f"   ✅ อัปเดตฐานข้อมูลเรียบร้อย ({ticker} - {future_date})")
        else:
            # ❌ ไม่มีแถวของ `future_date`
            print(f"   ❌ ไม่พบข้อมูลของ {ticker} - {future_date} ในฐานข้อมูล")
            db_connection.rollback()

        # ✅ เก็บผลลัพธ์
        all_predictions.append({
            'Ticker': ticker,
            'Date': future_date,
            'Predicted_Price': predicted_price,
            'Predicted_Dir': predicted_dir,
        })

    print("\n✅ การทำนายและอัปเดตฐานข้อมูลเสร็จสมบูรณ์!")

    # ✅ คืนค่า DataFrame ผลลัพธ์
    return pd.DataFrame(all_predictions)

# ✅ เชื่อมต่อฐานข้อมูล MySQL
db_connection = pymysql.connect(
    host="10.10.50.62",
    user="trademine",
    password="trade789",
    database="TradeMine",
    cursorclass=pymysql.cursors.DictCursor
)

# ✅ เรียกใช้ฟังก์ชัน
predictions_df = walk_forward_validation_multi_task(
    db_connection=db_connection,
    feature_columns=feature_columns,
    scaler_features=scaler_features,
    scaler_target=scaler_target,
    ticker_encoder=ticker_encoder,
    seq_length=10
)

# ✅ บันทึกผลลัพธ์
predictions_df.to_csv("weighted_stacking_predictions.csv", index=False)
print("\n✅ บันทึกผลลัพธ์ลงไฟล์ 'weighted_stacking_predictions.csv' สำเร็จ!")
