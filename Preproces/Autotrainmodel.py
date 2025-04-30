import numpy as np
import pandas as pd
import sqlalchemy
import os
import ta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings
from datetime import datetime
import mysql.connector
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------- 1) CONFIG -------------------------
# Load environment variables first
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.env')
load_dotenv(path)

# Get database connection string from environment variables
DB_CONNECTION = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"

# Determine market filter based on current time
current_hour = datetime.now().hour

# if 8 <= current_hour < 18:
#     print("📊 กำลังประมวลผลตลาดหุ้นไทย (SET)...")
#     market_filter = "Thailand"
# elif 19 <= current_hour or current_hour < 5:
#     print("📊 กำลังประมวลผลตลาดหุ้นอเมริกา (NYSE & NASDAQ)...")
#     market_filter = "America"
# else:
#     print("❌ ไม่อยู่ในช่วงเวลาทำการของตลาดหุ้นไทยหรืออเมริกา")
#     exit()
market_filter = "America"
MODEL_LSTM_PATH = "../LSTM_model/best_multi_task_model.keras"
MODEL_GRU_PATH = "../GRU_Model/best_multi_task_model.keras"
SEQ_LENGTH = 10
RETRAIN_FREQUENCY = 5
w_lstm, w_gru = 0.5, 0.5  # ✅ ใช้ Weighted Stacking ระหว่าง LSTM และ GRU

# ------------------------- 2) ดึงข้อมูลจาก Database -------------------------
def fetch_latest_data():
    """
    ดึงข้อมูลหุ้นล่าสุดจากฐานข้อมูล MySQL
    """
    engine = sqlalchemy.create_engine(DB_CONNECTION)

    query = f"""
        SELECT 
            StockDetail.Date, 
            StockDetail.StockSymbol, 
            Stock.Market,  
            StockDetail.OpenPrice AS Open, 
            StockDetail.HighPrice AS High, 
            StockDetail.LowPrice AS Low, 
            StockDetail.ClosePrice AS Close, 
            StockDetail.Volume, 
            StockDetail.P_BV_Ratio,
            StockDetail.Sentiment, 
            StockDetail.Changepercen AS Change_Percent, 
            StockDetail.TotalRevenue, 
            StockDetail.QoQGrowth, 
            StockDetail.EPS, 
            StockDetail.ROE, 
            StockDetail.NetProfitMargin, 
            StockDetail.DebtToEquity, 
            StockDetail.PERatio, 
            StockDetail.Dividend_Yield, 
            StockDetail.positive_news, 
            StockDetail.negative_news, 
            StockDetail.neutral_news 
        FROM StockDetail
        LEFT JOIN Stock ON StockDetail.StockSymbol = Stock.StockSymbol
        WHERE Stock.Market = '{market_filter}'  
        AND StockDetail.Date >= CURDATE() - INTERVAL 365 DAY
        ORDER BY StockDetail.StockSymbol, StockDetail.Date ASC;
    """

    df = pd.read_sql(query, engine)
    engine.dispose()

    if df.empty:
        print("❌ ไม่มีข้อมูลหุ้นสำหรับตลาดที่กำลังเปิดอยู่")
        return df

    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Fill missing dates for each stock
    grouped = df.groupby('StockSymbol')
    filled_dfs = []
    
    for name, group in grouped:
        # Create complete date range for this stock
        all_dates = pd.date_range(start=group['Date'].min(), end=group['Date'].max(), freq='D')
        temp_df = pd.DataFrame({'Date': all_dates})
        temp_df['StockSymbol'] = name
        # Merge with original data
        merged = pd.merge(temp_df, group, on=['StockSymbol', 'Date'], how='left')
        # Forward fill missing values
        merged.fillna(method='ffill', inplace=True)
        filled_dfs.append(merged)
    
    df = pd.concat(filled_dfs)
    
    # Calculate technical indicators for each stock
    def calculate_indicators(group):
        if len(group) < 14:
            return group
            
        # Calculate RSI
        group['RSI'] = ta.momentum.RSIIndicator(group['Close'], window=14).rsi()
        
        # Calculate EMAs
        group['EMA_12'] = group['Close'].ewm(span=12, adjust=False).mean()
        group['EMA_26'] = group['Close'].ewm(span=26, adjust=False).mean()
        group['EMA_10'] = group['Close'].ewm(span=10, adjust=False).mean()
        group['EMA_20'] = group['Close'].ewm(span=20, adjust=False).mean()
        
        # Calculate SMAs
        group['SMA_50'] = group['Close'].rolling(window=50).mean()
        group['SMA_200'] = group['Close'].rolling(window=200).mean()
        
        # Calculate MACD
        group['MACD'] = group['EMA_12'] - group['EMA_26']
        group['MACD_Signal'] = group['MACD'].rolling(window=9).mean()
        
        # Calculate ATR
        if len(group) >= 14:
            atr = ta.volatility.AverageTrueRange(high=group['High'], low=group['Low'], close=group['Close'], window=14)
            group['ATR'] = atr.average_true_range()
        
        # Calculate Bollinger Bands
        bollinger = ta.volatility.BollingerBands(group['Close'], window=20, window_dev=2)
        group['Bollinger_High'] = bollinger.bollinger_hband()
        group['Bollinger_Low'] = bollinger.bollinger_lband()
        
        # Convert Sentiment to numerical values
        group['Sentiment'] = group['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
        
        # Calculate Keltner Channel
        keltner = ta.volatility.KeltnerChannel(high=group['High'], low=group['Low'], close=group['Close'], window=20, window_atr=10)
        group['Keltner_High'] = keltner.keltner_channel_hband()
        group['Keltner_Low'] = keltner.keltner_channel_lband()
        group['Keltner_Middle'] = keltner.keltner_channel_mband()
        
        # Calculate Chaikin Volatility
        window_cv = 10
        group['High_Low_Diff'] = group['High'] - group['Low']
        group['High_Low_EMA'] = group['High_Low_Diff'].ewm(span=window_cv, adjust=False).mean()
        group['Chaikin_Vol'] = group['High_Low_EMA'].pct_change(periods=window_cv) * 100
        
        # Calculate Donchian Channel
        window_dc = 20
        group['Donchian_High'] = group['High'].rolling(window=window_dc).max()
        group['Donchian_Low'] = group['Low'].rolling(window=window_dc).min()
        
        # Calculate PSAR
        psar = ta.trend.PSARIndicator(high=group['High'], low=group['Low'], close=group['Close'], step=0.02, max_step=0.2)
        group['PSAR'] = psar.psar()
        
        return group
    
    # Apply indicators calculation to each stock group
    df = df.groupby('StockSymbol').apply(calculate_indicators)
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df

# ------------------------- 3) โหลดโมเดล -------------------------
try:
    model_lstm = load_model(MODEL_LSTM_PATH, compile=False)
    model_gru = load_model(MODEL_GRU_PATH, compile=False)
    print("✅ โหลดโมเดลสำเร็จ!")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    exit()

# ------------------------- 4) เตรียมข้อมูล -------------------------
feature_columns = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Change_Percent', 'Sentiment','positive_news','negative_news','neutral_news',
    'TotalRevenue', 'QoQGrowth', 'EPS', 'ROE', 'NetProfitMargin', 
    'DebtToEquity', 'PERatio', 'Dividend_Yield', 'P_BV_Ratio',
    'ATR', 'Keltner_High', 'Keltner_Low', 'Keltner_Middle', 'Chaikin_Vol',
    'Donchian_High', 'Donchian_Low', 'PSAR',
    'RSI', 'EMA_10', 'EMA_20', 'MACD', 'MACD_Signal', 
    'Bollinger_High', 'Bollinger_Low', 'SMA_50', 'SMA_200'
]

# ดึงข้อมูล
test_df = fetch_latest_data()

if test_df.empty:
    print("❌ ไม่มีข้อมูลสำหรับประมวลผล")
    exit()

us_stock = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'TSM', 'AMD']
thai_stock = ['ADVANC', 'INTUCH', 'TRUE', 'DITTO', 'DIF', 
           'INSET', 'JMART', 'INET', 'JAS', 'HUMAN']
test_df['Market_ID'] = test_df['StockSymbol'].apply(lambda x: "US" if x in us_stock else "TH" if x in thai_stock else None)

# Initialize scalers and encoder
scaler_features = RobustScaler()
scaler_target = RobustScaler()
ticker_encoder = LabelEncoder()
market_encoder = LabelEncoder()
# Fit and transform
test_df["Ticker_ID"] = ticker_encoder.fit_transform(test_df["StockSymbol"])
test_df['Market_ID'] = market_encoder.fit_transform(test_df['Market_ID'])
scaler_features.fit(test_df[feature_columns])
scaler_target.fit(test_df[["Close"]])

# ------------------------- 5) Predict Next Day -------------------------
def predict_next_day(model_lstm, model_gru, df, feature_columns, scaler_features, scaler_target, ticker_encoder, seq_length):
    all_predictions = []
    tickers = df['StockSymbol'].unique()

    for ticker in tickers:
        print(f"\nProcessing Ticker: {ticker}")
        df_ticker = df[df['StockSymbol'] == ticker].sort_values('Date').reset_index(drop=True)

        if len(df_ticker) < seq_length:
            print(f"⚠️ Not enough data for ticker {ticker}, skipping...")
            continue

        # ใช้ข้อมูล seq_length วันล่าสุด
        historical_data = df_ticker.iloc[-seq_length:]
        last_date = df_ticker.iloc[-1]["Date"]

        # สเกลข้อมูล
        features_scaled = scaler_features.transform(historical_data[feature_columns])
        ticker_ids = historical_data["Ticker_ID"].values
        market_ids = historical_data["Market_ID"].values

        # พยากรณ์ด้วย LSTM และ GRU
        try:
            # พยากรณ์ด้วย LSTM
            pred_price_lstm_output = model_lstm.predict(
                [features_scaled.reshape(1, seq_length, len(feature_columns)), 
                ticker_ids.reshape(1, seq_length),
                market_ids.reshape(1, seq_length)], 
                verbose=0
            )

            # กำจัด list ซ้อน (ถ้ามี) และ squeeze
            if isinstance(pred_price_lstm_output, list):
                pred_price_lstm_scaled = np.squeeze(pred_price_lstm_output[0])
            else:
                pred_price_lstm_scaled = np.squeeze(pred_price_lstm_output)

            
            # พยากรณ์ด้วย GRU
            pred_price_gru_output = model_gru.predict(
                [features_scaled.reshape(1, seq_length, len(feature_columns)), 
                ticker_ids.reshape(1, seq_length),
                market_ids.reshape(1, seq_length)], 
                verbose=0
            )

            if isinstance(pred_price_gru_output, list):
                pred_price_gru_scaled = np.squeeze(pred_price_gru_output[0])
            else:
                pred_price_gru_scaled = np.squeeze(pred_price_gru_output)


            pred_price_lstm = scaler_target.inverse_transform(pred_price_lstm_scaled.reshape(-1, 1)).flatten()[0]
            pred_price_gru = scaler_target.inverse_transform(pred_price_gru_scaled.reshape(-1, 1)).flatten()[0]
            
            # ใช้ Weighted Stacking ระหว่าง LSTM และ GRU
            predicted_price = (w_lstm * pred_price_lstm) + (w_gru * pred_price_gru)

            # ทำนายทิศทาง
            predicted_direction = 1 if predicted_price > df_ticker.iloc[-1]['Close'] else 0

            all_predictions.append({
                'StockSymbol': ticker,
                'Date': last_date + pd.Timedelta(days=1),  # ทำนายสำหรับวันถัดไป
                'Predicted_Price': predicted_price,
                'Predicted_Direction': predicted_direction
            })
            
        except Exception as e:
            print(f"⚠️ เกิดข้อผิดพลาดในการพยากรณ์สำหรับ {ticker}: {e}")
            continue

    predictions_df = pd.DataFrame(all_predictions)
    return predictions_df

# ------------------------- 6) บันทึกผลลัพธ์ลง Database -------------------------
def save_predictions_to_db(predictions_df):
    """
    บันทึกผลลัพธ์การพยากรณ์ลงในตาราง `StockDetail`
    """
    if predictions_df.empty:
        print("❌ ไม่มีข้อมูลพยากรณ์ที่จะบันทึก")
        return

    engine = sqlalchemy.create_engine(DB_CONNECTION)
    
    with engine.connect() as connection:
        for _, row in predictions_df.iterrows():
            # ตรวจสอบว่ามีข้อมูลสำหรับหุ้นนี้ในวันที่ทำนายหรือไม่
            check_query = f"""
                SELECT COUNT(*) FROM StockDetail 
                WHERE StockSymbol = '{row['StockSymbol']}' 
                AND Date = '{row['Date'].strftime('%Y-%m-%d')}'
            """
            exists = connection.execute(check_query).scalar()
            
            if exists > 0:
                # อัปเดตข้อมูลที่มีอยู่
                update_query = f"""
                    UPDATE StockDetail
                    SET PredictionClose = {row['Predicted_Price']}, 
                        PredictionTrend = {row['Predicted_Direction']}
                    WHERE StockSymbol = '{row['StockSymbol']}' 
                    AND Date = '{row['Date'].strftime('%Y-%m-%d')}'
                """
            else:
                # แทรกข้อมูลใหม่
                insert_query = f"""
                    INSERT INTO StockDetail 
                    (StockSymbol, Date, PredictionClose, PredictionTrend)
                    VALUES (
                        '{row['StockSymbol']}', 
                        '{row['Date'].strftime('%Y-%m-%d')}', 
                        {row['Predicted_Price']}, 
                        {row['Predicted_Direction']}
                    )
                """
                update_query = insert_query
            
            try:
                connection.execute(update_query)
            except Exception as e:
                print(f"⚠️ เกิดข้อผิดพลาดในการบันทึกข้อมูลสำหรับ {row['StockSymbol']}: {e}")
                continue
        
        connection.commit()
    
    print("\n✅ บันทึกผลลัพธ์ลงในฐานข้อมูลสำเร็จ!")

# ------------------------- 7) RUN -------------------------
if __name__ == "__main__":
    print("\n🔮 กำลังทำนายราคาหุ้นสำหรับวันถัดไป...")
    predictions_df = predict_next_day(
        model_lstm, model_gru, test_df, feature_columns, 
        scaler_features, scaler_target, ticker_encoder, SEQ_LENGTH
    )
    
    if not predictions_df.empty:
        print("\nผลลัพธ์การทำนาย:")
        print(predictions_df[['StockSymbol', 'Date', 'Predicted_Price', 'Predicted_Direction']])
        # save_predictions_to_db(predictions_df)
    else:
        print("❌ ไม่มีข้อมูลพยากรณ์ที่จะบันทึก")