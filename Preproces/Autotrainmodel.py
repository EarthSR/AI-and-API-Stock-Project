import numpy as np
import pandas as pd
import sqlalchemy
import os
import ta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings
from datetime import datetime, timedelta
import mysql.connector
from dotenv import load_dotenv
import xgboost as xgb
import lightgbm as lgb


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
lgbm_price_model = joblib.load('../Ensemble_Model/lgbm_price_model.pkl')
xgb_dir_model = joblib.load('../Ensemble_Model/xgb_dir_model.pkl')
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
            StockDetail.neutral_news,
            StockDetail.PredictionClose_GRU, 
            StockDetail.PredictionClose_LSTM, 
            StockDetail.PredictionTrend_GRU, 
            StockDetail.PredictionTrend_LSTM 
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
        financial_cols = [
        'TotalRevenue', 'QoQGrowth', 'EPS', 'ROE',
        'NetProfitMargin', 'DebtToEquity', 'PERatio', 'Dividend_Yield'
        ]
        merged[financial_cols] = merged[financial_cols].fillna(0)
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
        
        # Add date-related features
        group['DayOfWeek'] = group['Date'].dt.dayofweek
        group['Is_Day_0'] = (group['Date'].dt.dayofweek == 0).astype(int)  # Monday
        group['Is_Day_4'] = (group['Date'].dt.dayofweek == 4).astype(int)  # Friday
        group['DayOfMonth'] = group['Date'].dt.day
        group['IsFirstHalfOfMonth'] = (group['Date'].dt.day <= 15).astype(int)
        group['IsSecondHalfOfMonth'] = (group['Date'].dt.day > 15).astype(int)
        
        group['Predicted_Price_LSTM'] = group['PredictionClose_LSTM']
        group['Predicted_Price_GRU'] = group['PredictionClose_GRU']
        
        # Calculate prediction directions based on current close vs predicted close
        group['Predicted_Dir_LSTM'] = np.where(group['Predicted_Price_LSTM'] > group['Close'], 1, -1)
        group['Predicted_Dir_GRU'] = np.where(group['Predicted_Price_GRU'] > group['Close'], 1, -1)
        
        # Calculate differences between LSTM and GRU predictions
        group['LSTM_GRU_Price_Diff'] = abs(group['Predicted_Price_LSTM'] - group['Predicted_Price_GRU'])
        group['LSTM_GRU_Dir_Match'] = (group['Predicted_Dir_LSTM'] == group['Predicted_Dir_GRU']).astype(int)
        
        # Calculate 1-day accuracy by comparing prediction with actual next day result
        # We need to shift the data to compare with next day's actual close
        next_day_close = group['Close'].shift(-1)
        group['LSTM_Accuracy_1d'] = ((group['Predicted_Price_LSTM'] > group['Close']) == (next_day_close > group['Close'])).astype(float)
        group['GRU_Accuracy_1d'] = ((group['Predicted_Price_GRU'] > group['Close']) == (next_day_close > group['Close'])).astype(float)
        
        # Calculate 3-day directional accuracy
        # For this we need to see if prediction matches 3-day future trend
        three_day_future_close = group['Close'].shift(-3)
        actual_3d_dir = np.where(three_day_future_close > group['Close'], 1, -1)
        group['LSTM_Dir_Accuracy_3d'] = (group['Predicted_Dir_LSTM'] == actual_3d_dir).astype(float)
        group['GRU_Dir_Accuracy_3d'] = (group['Predicted_Dir_GRU'] == actual_3d_dir).astype(float)
        
        # Calculate price SMA differences
        # Creating 5 and 10 day SMAs for predicted prices
        group['LSTM_Pred_SMA_5'] = group['Predicted_Price_LSTM'].rolling(window=5).mean()
        group['LSTM_Pred_SMA_10'] = group['Predicted_Price_LSTM'].rolling(window=10).mean()
        group['GRU_Pred_SMA_5'] = group['Predicted_Price_GRU'].rolling(window=5).mean()
        group['GRU_Pred_SMA_10'] = group['Predicted_Price_GRU'].rolling(window=10).mean()
        
        group['LSTM_Price_SMA_Diff_5_10'] = group['LSTM_Pred_SMA_5'] - group['LSTM_Pred_SMA_10']
        group['GRU_Price_SMA_Diff_5_10'] = group['GRU_Pred_SMA_5'] - group['GRU_Pred_SMA_10']
        
        # Calculate 3-day directional consistency
        # This measures how consistent the direction predictions are over 3 days
        group['LSTM_Dir_Consistency_3d'] = group['Predicted_Dir_LSTM'].rolling(window=3).apply(lambda x: abs(x.sum()) / 3)
        group['GRU_Dir_Consistency_3d'] = group['Predicted_Dir_GRU'].rolling(window=3).apply(lambda x: abs(x.sum()) / 3)

        
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

feature_columns_ensemble = [
    'Predicted_Price_LSTM', 'Predicted_Price_GRU', 'Predicted_Dir_LSTM', 'Predicted_Dir_GRU', 
    'LSTM_GRU_Price_Diff', 'LSTM_GRU_Dir_Match', 'LSTM_Accuracy_1d', 'GRU_Accuracy_1d', 
    'LSTM_Dir_Accuracy_3d', 'GRU_Dir_Accuracy_3d', 'LSTM_Price_SMA_Diff_5_10', 'GRU_Price_SMA_Diff_5_10', 
    'LSTM_Dir_Consistency_3d', 'GRU_Dir_Consistency_3d', 'DayOfWeek', 'Is_Day_0', 'Is_Day_4', 
    'DayOfMonth', 'IsFirstHalfOfMonth', 'IsSecondHalfOfMonth'
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
scaler_ensemble_features = joblib.load('../Ensemble_Model/scaler_features.pkl')
scaler_ensemble_target = RobustScaler()
scaler_main_features = RobustScaler()
scaler_main_target = RobustScaler()
ticker_encoder = LabelEncoder()
market_encoder = LabelEncoder()
# Fit and transform
test_df["Ticker_ID"] = ticker_encoder.fit_transform(test_df["StockSymbol"])
test_df['Market_ID'] = market_encoder.fit_transform(test_df['Market_ID'])
scaler_ensemble_features.fit(test_df[feature_columns_ensemble])
scaler_ensemble_target.fit(test_df[["Close"]])
scaler_main_features.fit(test_df[feature_columns])
scaler_main_target.fit(test_df[["Close"]])

LAST_TRAINED_PATH = "last_trained.txt"

def should_retrain_main_model():
    if not os.path.exists(LAST_TRAINED_PATH):
        print("📅 ไม่พบไฟล์ last_trained.txt - retrain ครั้งแรก")
        return True

    with open(LAST_TRAINED_PATH, "r") as f:
        last_trained_str = f.read().strip()
    last_trained_date = datetime.strptime(last_trained_str, "%Y-%m-%d")

    days_since_last_train = (datetime.now() - last_trained_date).days
    print(f"📅 วันล่าสุดที่ retrain: {last_trained_date.date()} ({days_since_last_train} วันก่อน)")

    return days_since_last_train >= RETRAIN_FREQUENCY

def update_last_trained_date():
    with open(LAST_TRAINED_PATH, "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d"))

def should_retrain():
    if not os.path.exists(LAST_TRAINED_PATH):
        print("📅 ไม่พบไฟล์ last_trained.txt - retrain ครั้งแรก")
        return True

    with open(LAST_TRAINED_PATH, "r") as f:
        last_trained_str = f.read().strip()
    last_trained_date = datetime.strptime(last_trained_str, "%Y-%m-%d")

    days_since_last_train = (datetime.now() - last_trained_date).days
    print(f"📅 วันล่าสุดที่ retrain: {last_trained_date.date()} ({days_since_last_train} วันก่อน)")

    return days_since_last_train >= RETRAIN_FREQUENCY


def predict_future_day(model_lstm, model_gru, lgbm_price_model, xgb_dir_model, df, 
                     feature_columns, feature_columns_ensemble, scaler_features, 
                     scaler_target, scaler_features_ensemble,scaler_target_ensemble, ticker_encoder, seq_length):
    """ทำนายราคาหุ้นวันถัดไปจากข้อมูลล่าสุด"""
    
    future_predictions = []
    tickers = df['StockSymbol'].unique()
    retraining_enabled = should_retrain()
    
    # สำหรับการรีเทรน
    batch_features, batch_tickers, batch_markets, batch_prices, batch_directions = [], [], [], [], []

    for ticker in tickers:
        print(f"\n📊 กำลังทำนายอนาคตสำหรับหุ้น: {ticker}")
        df_ticker = df[df['StockSymbol'] == ticker].sort_values('Date').reset_index(drop=True)

        if len(df_ticker) < seq_length:
            print(f"⚠️ ไม่มีข้อมูลเพียงพอสำหรับหุ้น {ticker}, ข้ามไป...")
            continue

        # ใช้ข้อมูล seq_length ล่าสุดเพื่อทำนายวันถัดไป
        latest_data = df_ticker.iloc[-seq_length:]
        features_scaled = scaler_features.transform(latest_data[feature_columns])
        ticker_ids = latest_data["Ticker_ID"].values
        market_ids = latest_data["Market_ID"].values

        X_feat = features_scaled.reshape(1, seq_length, -1)
        X_ticker = ticker_ids.reshape(1, seq_length)
        X_market = market_ids.reshape(1, seq_length)

        # พยากรณ์ด้วย LSTM
        pred_output_lstm = model_lstm.predict(
            [X_feat, X_ticker, X_market], 
            verbose=0
        )

        # แยกผลลัพธ์ทำนายราคาและทิศทางจาก LSTM
        pred_price_lstm_scaled = np.squeeze(pred_output_lstm[0])  # ทำนายราคา
        pred_direction_lstm = np.squeeze(pred_output_lstm[1])  # ทำนายทิศทาง

        # กำจัด list ซ้อน (ถ้ามี) และ squeeze
        pred_price_lstm = scaler_target.inverse_transform(pred_price_lstm_scaled.reshape(-1, 1)).flatten()[0]

        # พยากรณ์ด้วย GRU
        pred_output_gru = model_gru.predict(
            [X_feat, X_ticker, X_market], 
            verbose=0
        )

        # แยกผลลัพธ์ทำนายราคาและทิศทางจาก GRU
        pred_price_gru_scaled = np.squeeze(pred_output_gru[0])  # ทำนายราคา
        pred_direction_gru = np.squeeze(pred_output_gru[1])  # ทำนายทิศทาง

        # กำจัด list ซ้อน (ถ้ามี) และ squeeze
        pred_price_gru = scaler_target.inverse_transform(pred_price_gru_scaled.reshape(-1, 1)).flatten()[0]

        # สร้างวันที่ถัดไป
        last_date = df_ticker['Date'].max()
        next_day = last_date + pd.Timedelta(days=1)
        
        # เตรียมฟีเจอร์สำหรับโมเดล Ensemble
        current_row = latest_data.iloc[-1].copy()
        ensemble_features = {
            'Predicted_Price_LSTM': pred_price_lstm,
            'Predicted_Price_GRU': pred_price_gru,
            'Predicted_Dir_LSTM': 1 if pred_direction_lstm > 0.5 else 0,
            'Predicted_Dir_GRU': 1 if pred_direction_gru > 0.5 else 0,
            'LSTM_GRU_Price_Diff': abs(pred_price_lstm - pred_price_gru),
            'LSTM_GRU_Dir_Match': 1 if ((pred_price_lstm > current_row['Close']) == (pred_price_gru > current_row['Close'])) else 0,
            'LSTM_Accuracy_1d': current_row.get('LSTM_Accuracy_1d', 0),
            'GRU_Accuracy_1d': current_row.get('GRU_Accuracy_1d', 0),
            'LSTM_Dir_Accuracy_3d': current_row.get('LSTM_Dir_Accuracy_3d', 0),
            'GRU_Dir_Accuracy_3d': current_row.get('GRU_Dir_Accuracy_3d', 0),
            'LSTM_Price_SMA_Diff_5_10': current_row.get('LSTM_Price_SMA_Diff_5_10', 0),
            'GRU_Price_SMA_Diff_5_10': current_row.get('GRU_Price_SMA_Diff_5_10', 0),
            'LSTM_Dir_Consistency_3d': current_row.get('LSTM_Dir_Consistency_3d', 0),
            'GRU_Dir_Consistency_3d': current_row.get('GRU_Dir_Consistency_3d', 0),
            'DayOfWeek': next_day.dayofweek,  # คำนวณวันในสัปดาห์ของวันถัดไป
            'Is_Day_0': 1 if next_day.dayofweek == 0 else 0,  # Monday
            'Is_Day_4': 1 if next_day.dayofweek == 4 else 0,  # Friday
            'DayOfMonth': next_day.day,
            'IsFirstHalfOfMonth': 1 if next_day.day <= 15 else 0,
            'IsSecondHalfOfMonth': 1 if next_day.day > 15 else 0,
            'Last_Close': current_row['Close']
        }

        # สร้าง DataFrame สำหรับ Ensemble
        ensemble_df = pd.DataFrame([ensemble_features])
        
        # ปรับขนาดฟีเจอร์
        ensemble_features_scaled = scaler_features_ensemble.transform(ensemble_df[feature_columns_ensemble])
        ensemble_df_scaled = pd.DataFrame(ensemble_features_scaled, columns=feature_columns_ensemble)

        # ทำนายด้วยโมเดล Ensemble
        predicted_price = lgbm_price_model.predict(ensemble_df_scaled)[0]
        predicted_price = scaler_target_ensemble.inverse_transform([[predicted_price]])[0][0]

        direction_proba = xgb_dir_model.predict_proba(ensemble_df_scaled)[0][1]
        predicted_direction = 1 if direction_proba >= 0.5 else 0
        
        # เพิ่มผลลัพธ์
        future_predictions.append({
            'StockSymbol': ticker,
            'Date': next_day,
            'Predicted_Price': predicted_price,
            'Predicted_Direction': predicted_direction,
            'Direction_Probability': direction_proba,
            'LSTM_Direction': 1 if pred_direction_lstm > 0.5 else 0,
            'GRU_Direction': 1 if pred_direction_gru > 0.5 else 0,
            'LSTM_Prediction': pred_price_lstm,
            'GRU_Prediction': pred_price_gru,
            'Last_Close': current_row['Close'],
            'Price_Change': predicted_price - current_row['Close'],
            'Price_Change_Percent': (predicted_price - current_row['Close']) / current_row['Close'] * 100,
            'Confidence_Score': direction_proba if predicted_direction == 1 else 1 - direction_proba
        })
        
        # เตรียมข้อมูลสำหรับการรีเทรน
        if retraining_enabled:
            # ใช้ข้อมูลล่าสุดเพื่อรีเทรนโมเดล
            last_seq_data = df_ticker.iloc[-seq_length-1:-1]  # ข้อมูลล่าสุดที่มีค่าจริง
            actual_last_price = df_ticker.iloc[-1]['Close']
            actual_last_direction = 1 if actual_last_price > current_row['Close'] else 0
            
            features_scaled = scaler_features.transform(last_seq_data[feature_columns])
            ticker_ids = last_seq_data["Ticker_ID"].values
            market_ids = last_seq_data["Market_ID"].values
            
            X_feat = features_scaled.reshape(1, seq_length, -1)
            X_ticker = ticker_ids.reshape(1, seq_length)
            X_market = market_ids.reshape(1, seq_length)
            
            y_price = scaler_target.transform(np.array([[actual_last_price]]))
            y_direction = np.array([[actual_last_direction]])
            
            batch_features.append(X_feat)
            batch_tickers.append(X_ticker)
            batch_markets.append(X_market)
            batch_prices.append(y_price)
            batch_directions.append(y_direction)
            
            # รีเทรนหลังจากประมวลผลครบทุกหุ้น
            if len(tickers) - 1 == list(tickers).index(ticker):  # หุ้นสุดท้าย
                if len(batch_features) > 0:
                    Xf = np.concatenate(batch_features, axis=0)
                    Xt = np.concatenate(batch_tickers, axis=0)
                    Xm = np.concatenate(batch_markets, axis=0)
                    Yp = np.concatenate(batch_prices, axis=0)
                    Dr = np.concatenate(batch_directions, axis=0)
                    
                    print(f"🔁 กำลังรีเทรนโมเดลด้วยข้อมูล {len(Xf)} ตัวอย่าง")
                    model_lstm.fit([Xf, Xt, Xm], 
                    {                        
                        'price_output': Yp,
                        'direction_output': Dr}
                    , epochs=1, batch_size=len(Xf)
                    ,verbose=0, 
                    shuffle=False)
                    model_gru.fit([Xf, Xt, Xm], 
                    {                        
                        'price_output': Yp,
                        'direction_output': Dr}
                    , epochs=1, batch_size=len(Xf)
                    ,verbose=0, 
                    shuffle=False)
                    
                    batch_features.clear()
                    batch_tickers.clear()
                    batch_markets.clear()
                    batch_prices.clear()
                
                update_last_trained_date()
                print("✅ อัปเดตวันที่รีเทรนล่าสุดเรียบร้อยแล้ว")

    return pd.DataFrame(future_predictions)

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
    
    
def check_model_features(model, model_type='lgbm'):
    """ตรวจสอบ features ที่โมเดลคาดหวัง"""
    if model_type == 'lgbm':
        try:
            return model.feature_name_
        except AttributeError:
            return [f"Column_{i}" for i in range(len(model.feature_importances_))]
    elif model_type == 'xgb':
        try:
            return model.get_booster().feature_names
        except AttributeError:
            return [f"Column_{i}" for i in range(len(model.feature_importances_))]
    return []

# ------------------------- 7) RUN -------------------------
if __name__ == "__main__":
    print("\n🔮 กำลังทำนายราคาหุ้นสำหรับวันถัดไป...")
    test_df.to_csv('latest_data.csv', index=False)
    print(check_model_features(lgbm_price_model, 'lgbm'))
    print(check_model_features(xgb_dir_model, 'xgb'))
    # 2. เรียกใช้ฟังก์ชันใหม่เพื่อทำนายวันถัดไป (สำคัญ!)
    future_predictions_df = predict_future_day(
        model_lstm, model_gru, lgbm_price_model, xgb_dir_model,
        test_df, feature_columns, feature_columns_ensemble, scaler_main_features, 
        scaler_main_target, scaler_ensemble_features, scaler_ensemble_target, ticker_encoder, SEQ_LENGTH
    )
    
    if not future_predictions_df.empty:
        print("\nผลลัพธ์การทำนายวันถัดไป:")
        to_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'future_predictions.csv')
        future_predictions_df.to_csv(to_csv_path, index=False)
        # แสดงผลลัพธ์ที่สำคัญ
        display_cols = ['StockSymbol', 'Date', 'Last_Close', 'Predicted_Price', 
                        'Price_Change_Percent', 'Predicted_Direction', 'Confidence_Score']
        print(future_predictions_df[display_cols])
        
        # บันทึกลงฐานข้อมูล
        # save_predictions_to_db(future_predictions_df)
    else:
        print("❌ ไม่มีข้อมูลพยากรณ์ที่จะบันทึก")