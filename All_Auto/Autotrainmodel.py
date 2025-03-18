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
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------- 1) CONFIG -------------------------
DB_CONNECTION = "mysql+pymysql://trademine:trade789@10.10.50.62:3306/TradeMine"
MODEL_LSTM_PATH = "./best_multi_task_model_LSTM.keras"
MODEL_GRU_PATH = "./best_multi_task_model_GRU.keras"
SEQ_LENGTH = 10
RETRAIN_FREQUENCY = 1
w_lstm, w_gru = 0.5, 0.5  # ✅ ใช้ Weighted Stacking ระหว่าง LSTM และ GRU

def save_predictions_to_stockdetail(predictions_df):
    """
    อัปเดตค่าพยากรณ์ในตาราง `StockDetail` สำหรับหุ้นแต่ละตัวในวันที่ล่าสุด
    """
    engine = sqlalchemy.create_engine(DB_CONNECTION)

    with engine.connect() as connection:
        for _, row in predictions_df.iterrows():
            ticker = row['Ticker']
            date = row['Date']
            predicted_price = row['Predicted_Price']
            predicted_direction = row['Predicted_Direction']

            # ✅ อัปเดตค่าพยากรณ์ในตาราง `StockDetail` สำหรับวันที่ล่าสุด
            connection.execute(f"""
                UPDATE StockDetail
                SET PredictionClose = {predicted_price}, PredictionTrend = {predicted_direction}
                WHERE StockSymbol = '{ticker}' 
                AND Date = (SELECT MAX(Date) FROM StockDetail WHERE StockSymbol = '{ticker}')
            """)

    print("\n✅ อัปเดตค่าพยากรณ์ใน `StockDetail` สำเร็จ!")


# ------------------------- 2) ดึงข้อมูลจาก Database -------------------------
def fetch_latest_data():
    engine = sqlalchemy.create_engine(DB_CONNECTION)
    import sqlalchemy
import pandas as pd

# ------------------------- 2) ดึงข้อมูลจาก Database -------------------------
def fetch_latest_data():
    """
    ดึงข้อมูลหุ้นล่าสุดจากฐานข้อมูล MySQL
    """
    engine = sqlalchemy.create_engine(DB_CONNECTION)

    query = """
        SELECT 
            Date, 
            StockSymbol, 
            OpenPrice AS Open, 
            HighPrice AS High, 
            LowPrice AS Low, 
            ClosePrice AS Close, 
            Volume, 
            P_BV_Ratio,
            Sentiment, 
            Changepercen AS Change_Percent, 
            TotalRevenue, 
            QoQGrowth, 
            EPS, 
            ROE, 
            NetProfitMargin, 
            DebtToEquity, 
            PERatio, 
            Dividend_Yield 
        FROM StockDetail
        WHERE Date >= CURDATE() - INTERVAL 365 DAY
        ORDER BY Date ASC
    """

    df = pd.read_sql(query, engine)
    engine.dispose()

    # ✅ คำนวณฟีเจอร์ทางเทคนิค
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].rolling(window=9).mean()
    atr = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR'] = atr.average_true_range()
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    df['Sentiment'] = df['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})

    keltner = ta.volatility.KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20, window_atr=10)
    df['Keltner_High'] = keltner.keltner_channel_hband()
    df['Keltner_Low'] = keltner.keltner_channel_lband()
    df['Keltner_Middle'] = keltner.keltner_channel_mband()
    window_cv = 10
    df['High_Low_Diff'] = df['High'] - df['Low']
    df['High_Low_EMA'] = df['High_Low_Diff'].ewm(span=window_cv, adjust=False).mean()
    df['Chaikin_Vol'] = df['High_Low_EMA'].pct_change(periods=window_cv) * 100

    window_dc = 20
    df['Donchian_High'] = df['High'].rolling(window=window_dc).max()
    df['Donchian_Low'] = df['Low'].rolling(window=window_dc).min()
    psar = ta.trend.PSARIndicator(high=df['High'], low=df['Low'], close=df['Close'], step=0.02, max_step=0.2)
    df['PSAR'] = psar.psar()
    # ✅ เติมค่าที่ขาด
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)

    return df

test_df = fetch_latest_data()

# ------------------------- 3) โหลดโมเดลล่าสุด -------------------------
model_lstm = load_model(MODEL_LSTM_PATH, custom_objects={}, safe_mode=False)
model_gru = load_model(MODEL_GRU_PATH, custom_objects={}, safe_mode=False)

# ------------------------- 4) เตรียมข้อมูล -------------------------
feature_columns = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Change_Percent', 'Sentiment',
    'TotalRevenue', 'QoQGrowth', 'EPS', 'ROE', 'NetProfitMargin', 
    'DebtToEquity', 'PERatio', 'Dividend_Yield','P_BV_Ratio',
    'ATR', 'Keltner_High', 'Keltner_Low', 'Keltner_Middle','Chaikin_Vol','Donchian_High', 'Donchian_Low', 'PSAR',
    'RSI', 'EMA_10', 'EMA_20', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'SMA_50', 'SMA_200'
]

scaler_features = RobustScaler()
scaler_target = RobustScaler()
ticker_encoder = LabelEncoder()

test_df["Ticker_ID"] = ticker_encoder.fit_transform(test_df["StockSymbol"])
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

        # ✅ ใช้ข้อมูล 10 วันล่าสุด
        historical_data = df_ticker.iloc[-seq_length:]
        last_date = df_ticker.iloc[-1]["Date"]

        # ✅ สเกลข้อมูล
        features_scaled = pd.DataFrame(scaler_features.transform(historical_data[feature_columns]), columns=feature_columns)
        ticker_ids = historical_data["Ticker_ID"].values

        # ✅ พยากรณ์ด้วย LSTM และ GRU
        pred_price_lstm_scaled = np.array(model_lstm.predict([features_scaled.values.reshape(1, seq_length, len(feature_columns)), 
                                                              ticker_ids.reshape(1, seq_length)], verbose=0)).squeeze()
        pred_price_gru_scaled = np.array(model_gru.predict([features_scaled.values.reshape(1, seq_length, len(feature_columns)), 
                                                            ticker_ids.reshape(1, seq_length)], verbose=0)).squeeze()

        pred_price_lstm = scaler_target.inverse_transform(pred_price_lstm_scaled.reshape(-1, 1)).flatten()[0]
        pred_price_gru = scaler_target.inverse_transform(pred_price_gru_scaled.reshape(-1, 1)).flatten()[0]

        # ✅ ใช้ Weighted Stacking ระหว่าง LSTM และ GRU
        predicted_price = (w_lstm * pred_price_lstm) + (w_gru * pred_price_gru)

        # ✅ ทำนายทิศทาง
        predicted_direction = 1 if predicted_price > df_ticker.iloc[-1]['Close'] else 0

        all_predictions.append({
            'StockSymbol': ticker,
            'Date': last_date,  
            'PredictionClose': predicted_price,
            'PredictionTrend': predicted_direction
        })

    predictions_df = pd.DataFrame(all_predictions)
    return predictions_df

# ------------------------- 6) บันทึกผลลัพธ์ลง Database -------------------------
from sqlalchemy import create_engine, text

def save_predictions_to_db(predictions_df):
    """
    บันทึกผลลัพธ์การพยากรณ์ลงในตาราง `StockDetail`
    """
    engine = create_engine(DB_CONNECTION)

    with engine.connect() as connection:
        for index, row in predictions_df.iterrows():
            sql_query = text("""
                UPDATE StockDetail
                SET PredictionClose = :predicted_price, 
                    PredictionTrend = :predicted_trend
                WHERE StockSymbol = :ticker 
                    AND Date = :date
            """)

            connection.execute(sql_query, {
                "predicted_price": row["PredictionClose"],
                "predicted_trend": row["PredictionTrend"],
                "ticker": row["StockSymbol"],
                "date": row["Date"]
            })

        connection.commit()  # ✅ ต้อง Commit การเปลี่ยนแปลง
    print("\n✅ บันทึกผลลัพธ์ลงในฐานข้อมูลสำเร็จ!")


# ------------------------- 7) RUN -------------------------
predictions_df = predict_next_day(
    model_lstm, model_gru, test_df, feature_columns, scaler_features, scaler_target, ticker_encoder, SEQ_LENGTH
)
save_predictions_to_db(predictions_df)