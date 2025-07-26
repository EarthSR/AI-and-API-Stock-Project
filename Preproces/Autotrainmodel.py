import numpy as np
import pandas as pd
import sqlalchemy
import os
import tensorflow as tf
import ta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import joblib
import warnings
from datetime import datetime, timedelta
import mysql.connector
from dotenv import load_dotenv
from tensorflow.keras.optimizers import Adam
from sqlalchemy import text

# XGBoost imports
import xgboost as xgb
from sklearn.impute import SimpleImputer
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ======================== ENHANCED XGBoost Meta-Learner Class ========================
class XGBoostMetaLearner:
    """
    XGBoost Meta-Learner สำหรับรวม predictions จาก LSTM และ GRU
    พร้อมด้วย technical indicators เพิ่มเติม
    """
    
    def __init__(self, 
                 clf_model_path='../Ensemble_Model/xgb_classifier_model.pkl', 
                 reg_model_path='../Ensemble_Model/xgb_regressor_model.pkl',
                 scaler_dir_path='../Ensemble_Model/scaler_dir.pkl', 
                 scaler_price_path='../Ensemble_Model/scaler_price.pkl',
                 retrain_frequency=5):
        
        self.clf_model_path = clf_model_path
        self.reg_model_path = reg_model_path
        self.scaler_dir_path = scaler_dir_path
        self.scaler_price_path = scaler_price_path
        self.retrain_frequency = retrain_frequency
        
        self.xgb_clf = None
        self.xgb_reg = None
        self.scaler_dir = None
        self.scaler_price = None
        
        # โหลดโมเดลถ้ามีอยู่
        self.load_models()
    
    def load_models(self):
        """โหลด XGBoost models และ scalers"""
        try:
            if os.path.exists(self.clf_model_path):
                self.xgb_clf = joblib.load(self.clf_model_path)
                print("✅ โหลด XGBoost Classifier สำเร็จ")
            else:
                print("⚠️ ไม่พบไฟล์ XGBoost Classifier")
            
            if os.path.exists(self.reg_model_path):
                self.xgb_reg = joblib.load(self.reg_model_path)
                print("✅ โหลด XGBoost Regressor สำเร็จ")
            else:
                print("⚠️ ไม่พบไฟล์ XGBoost Regressor")
            
            if os.path.exists(self.scaler_dir_path):
                self.scaler_dir = joblib.load(self.scaler_dir_path)
                print("✅ โหลด Direction Scaler สำเร็จ")
            else:
                print("⚠️ ไม่พบไฟล์ Direction Scaler")
                
            if os.path.exists(self.scaler_price_path):
                self.scaler_price = joblib.load(self.scaler_price_path)
                print("✅ โหลด Price Scaler สำเร็จ")
            else:
                print("⚠️ ไม่พบไฟล์ Price Scaler")
                
        except Exception as e:
            print(f"⚠️ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    
    def calculate_technical_indicators(self, df):
        """คำนวณ technical indicators สำหรับ XGBoost"""
        
        def calculate_for_ticker(group):
            if len(group) < 26:  # ต้องการข้อมูลอย่างน้อย 26 วันสำหรับ MACD
                return group
            
            try:
                # RSI
                group['RSI'] = RSIIndicator(close=group['Close'], window=14).rsi()
                
                # SMA
                group['SMA_20'] = SMAIndicator(close=group['Close'], window=20).sma_indicator()
                
                # MACD
                macd = MACD(close=group['Close'])
                group['MACD'] = macd.macd()
                
                # Bollinger Bands
                bb = BollingerBands(close=group['Close'], window=20)
                group['BB_High'] = bb.bollinger_hband()
                group['BB_Low'] = bb.bollinger_lband()
                
                # ATR (ต้องมี High, Low columns)
                if 'High' in group.columns and 'Low' in group.columns:
                    atr = AverageTrueRange(high=group['High'], low=group['Low'], 
                                         close=group['Close'], window=14)
                    group['ATR'] = atr.average_true_range()
                else:
                    # สร้าง High, Low จาก Close ถ้าไม่มี
                    group['High'] = group['Close'] * 1.01
                    group['Low'] = group['Close'] * 0.99
                    atr = AverageTrueRange(high=group['High'], low=group['Low'], 
                                         close=group['Close'], window=14)
                    group['ATR'] = atr.average_true_range()
            
            except Exception as e:
                print(f"⚠️ เกิดข้อผิดพลาดในการคำนวณ technical indicators: {e}")
            
            return group
        
        # คำนวณ indicators แยกตาม ticker - แก้ไข DeprecationWarning
        df_with_indicators = df.groupby('StockSymbol', group_keys=False).apply(calculate_for_ticker)
        df_with_indicators = df_with_indicators.reset_index(drop=True)
        
        # Fill NaN values
        indicator_cols = ['RSI', 'SMA_20', 'MACD', 'BB_High', 'BB_Low', 'ATR']
        for col in indicator_cols:
            if col in df_with_indicators.columns:
                df_with_indicators[col] = df_with_indicators.groupby('StockSymbol')[col].ffill().bfill()
                df_with_indicators[col] = df_with_indicators[col].fillna(df_with_indicators[col].mean())
        
        return df_with_indicators
    
    def prepare_features(self, df):
        """เตรียม features สำหรับ XGBoost"""
        
        # คำนวณ technical indicators
        df = self.calculate_technical_indicators(df)
        
        # สร้าง meta features
        df['Price_Diff'] = df['PredictionClose_LSTM'] - df['PredictionClose_GRU']
        df['Dir_Agreement'] = (df['PredictionTrend_LSTM'] == df['PredictionTrend_GRU']).astype(int)
        
        # Normalize actual price ตาม ticker
        df['Actual_Price_Normalized'] = df.groupby('StockSymbol')['Close'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
        
        # Features สำหรับ direction prediction
        direction_features = [
            'PredictionTrend_LSTM', 'PredictionTrend_GRU', 'Dir_Agreement', 
            'RSI', 'SMA_20', 'MACD', 'BB_High', 'BB_Low', 'ATR'
        ]
        
        # Features สำหรับ price prediction
        price_features = [
            'PredictionClose_LSTM', 'PredictionClose_GRU', 'Price_Diff',
            'RSI', 'SMA_20', 'MACD', 'BB_High', 'BB_Low', 'ATR',
            'Actual_Price_Normalized'
        ]
        
        # ตรวจสอบว่า features ทั้งหมดมีอยู่
        available_dir_features = [f for f in direction_features if f in df.columns]
        available_price_features = [f for f in price_features if f in df.columns]
        
        return df, available_dir_features, available_price_features
    
    def predict_meta(self, df):
        """ทำนายด้วย XGBoost Meta-Learner"""
        
        if self.xgb_clf is None or self.xgb_reg is None:
            print("❌ XGBoost models ยังไม่ได้โหลด ไม่สามารถทำนายได้")
            return df
        
        # เตรียม features
        df_prepared, dir_features, price_features = self.prepare_features(df)
        
        # ตรวจสอบข้อมูลที่มี predictions จาก LSTM และ GRU
        prediction_mask = (
            df_prepared['PredictionClose_LSTM'].notna() & 
            df_prepared['PredictionClose_GRU'].notna() &
            df_prepared['PredictionTrend_LSTM'].notna() & 
            df_prepared['PredictionTrend_GRU'].notna()
        )
        
        if not prediction_mask.any():
            print("❌ ไม่มีข้อมูล predictions จาก LSTM/GRU")
            return df
        
        # เลือกเฉพาะแถวที่มี predictions
        df_to_predict = df_prepared[prediction_mask].copy()
        
        if len(df_to_predict) == 0:
            print("❌ ไม่มีข้อมูลสำหรับทำนาย")
            return df
        
        # จัดการ missing values
        imputer = SimpleImputer(strategy='mean')
        
        try:
            # Direction prediction
            X_dir = df_to_predict[dir_features]
            X_dir_filled = imputer.fit_transform(X_dir)
            
            if self.scaler_dir is not None:
                X_dir_scaled = self.scaler_dir.transform(X_dir_filled)
            else:
                print("⚠️ ไม่มี Direction Scaler, ใช้ข้อมูลดิบ")
                X_dir_scaled = X_dir_filled
            
            # Price prediction
            X_price = df_to_predict[price_features]
            X_price_filled = imputer.fit_transform(X_price)
            
            if self.scaler_price is not None:
                X_price_scaled = self.scaler_price.transform(X_price_filled)
            else:
                print("⚠️ ไม่มี Price Scaler, ใช้ข้อมูลดิบ")
                X_price_scaled = X_price_filled
            
            # ทำนาย
            # Direction prediction
            xgb_pred_dir = self.xgb_clf.predict(X_dir_scaled)
            xgb_pred_dir_proba = self.xgb_clf.predict_proba(X_dir_scaled)[:, 1]
            
            # Price prediction
            xgb_pred_price = self.xgb_reg.predict(X_price_scaled)
            
            # เพิ่มผลลัพธ์กลับเข้าไปใน DataFrame
            df_prepared.loc[prediction_mask, 'XGB_Predicted_Direction_Raw'] = xgb_pred_dir
            df_prepared.loc[prediction_mask, 'XGB_Predicted_Direction_Proba'] = xgb_pred_dir_proba
            df_prepared.loc[prediction_mask, 'XGB_Predicted_Price_Raw'] = xgb_pred_price
            
            # ใช้ Direction เป็นหลัก เพราะสำคัญที่สุดในการลงทุน
            df_prepared.loc[prediction_mask, 'XGB_Predicted_Direction'] = xgb_pred_dir
            
            # ปรับ Price ให้สอดคล้องกับ Direction ที่ทำนายได้
            current_prices = df_to_predict['Close'].values
            
            # คำนวณ price adjustment ตาม direction
            price_adjustments = []
            for i, (current_price, pred_dir, raw_price) in enumerate(zip(current_prices, xgb_pred_dir, xgb_pred_price)):
                raw_change_pct = (raw_price - current_price) / current_price
                
                if pred_dir == 1:  # ทิศทางขึ้น
                    if raw_price <= current_price:  # แต่ราคาทำนายลง
                        # ปรับให้เป็นการขึ้นเล็กน้อย (0.5-2%)
                        adjusted_change = max(0.005, abs(raw_change_pct) * 0.5)
                        adjusted_price = current_price * (1 + adjusted_change)
                    else:
                        adjusted_price = raw_price  # ใช้ราคาเดิม
                else:  # ทิศทางลง
                    if raw_price >= current_price:  # แต่ราคาทำนายขึ้น
                        # ปรับให้เป็นการลงเล็กน้อย (0.5-2%)
                        adjusted_change = max(0.005, abs(raw_change_pct) * 0.5)
                        adjusted_price = current_price * (1 - adjusted_change)
                    else:
                        adjusted_price = raw_price  # ใช้ราคาเดิม
                
                price_adjustments.append(adjusted_price)
            
            df_prepared.loc[prediction_mask, 'XGB_Predicted_Price'] = price_adjustments
            
            # คำนวณ confidence score
            df_prepared.loc[prediction_mask, 'XGB_Confidence'] = np.abs(xgb_pred_dir_proba - 0.5) * 2
            
            print(f"✅ XGBoost Meta-Learner ทำนายสำเร็จ {prediction_mask.sum()} แถว (Direction-focused)")
            
            print(f"✅ XGBoost Meta-Learner ทำนายสำเร็จ {prediction_mask.sum()} แถว")
            
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการทำนายด้วย XGBoost: {e}")
            import traceback
            traceback.print_exc()
            
        return df_prepared
    
    def should_retrain_meta(self):
        """ตรวจสอบว่าควร retrain XGBoost หรือไม่"""
        meta_last_trained_path = "meta_last_trained.txt"
        
        if not os.path.exists(meta_last_trained_path):
            return True
        
        try:
            with open(meta_last_trained_path, "r") as f:
                last_trained_str = f.read().strip()
            last_trained_date = datetime.strptime(last_trained_str, "%Y-%m-%d")
            
            days_since_last_train = (datetime.now() - last_trained_date).days
            return days_since_last_train >= self.retrain_frequency
        except Exception as e:
            print(f"⚠️ เกิดข้อผิดพลาดในการตรวจสอบวันที่ retrain: {e}")
            return True

# ======================== ENHANCED PREDICTION SYSTEM ========================

# โหลด configuration และตรวจสอบ environment
print("🔧 กำลังโหลด configuration...")
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.env')

if not os.path.exists(path):
    print(f"❌ ไม่พบไฟล์ config.env ที่ {path}")
    print("📝 กำลังสร้างไฟล์ตัวอย่าง config.env...")
    
    try:
        with open(path, 'w') as f:
            f.write("# Database Configuration\n")
            f.write("DB_USER=your_username\n")
            f.write("DB_PASSWORD=your_password\n")
            f.write("DB_HOST=localhost\n")
            f.write("DB_NAME=your_database\n")
        
        print(f"✅ สร้างไฟล์ตัวอย่าง config.env สำเร็จที่ {path}")
        print("📋 กรุณาแก้ไขค่าในไฟล์นี้ให้ตรงกับการตั้งค่าฐานข้อมูลของคุณ")
        exit()
        
    except Exception as e:
        print(f"❌ ไม่สามารถสร้างไฟล์ config.env ได้: {e}")
        print("📝 กรุณาสร้างไฟล์ config.env ด้วยข้อมูล:")
        print("   DB_USER=your_username")
        print("   DB_PASSWORD=your_password") 
        print("   DB_HOST=your_host")
        print("   DB_NAME=your_database")
        exit()

load_dotenv(path)

# ตรวจสอบ environment variables
required_vars = ['DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_NAME']
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    print(f"❌ ขาด environment variables: {missing_vars}")
    exit()

try:
    DB_CONNECTION = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    print("✅ Database connection string สร้างสำเร็จ")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการสร้าง database connection: {e}")
    exit()

# ตั้งค่าตลาด
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
MODEL_LSTM_PATH = "../LSTM_model/best_v6_plus_minimal_tuning_v2_final_model.keras"
MODEL_GRU_PATH = "../GRU_Model/best_v6_plus_minimal_tuning_v2_final_model.keras"
SEQ_LENGTH = 10
RETRAIN_FREQUENCY = 5

# Dynamic weight parameters
WEIGHT_DECAY = 0.95
MIN_WEIGHT = 0.1
MAX_WEIGHT = 0.9

# สร้าง XGBoost Meta-Learner
print("🧠 กำลังเตรียม XGBoost Meta-Learner...")
meta_learner = XGBoostMetaLearner()

# ตรวจสอบสถานะ XGBoost models
xgb_available = (meta_learner.xgb_clf is not None and 
                meta_learner.xgb_reg is not None and
                meta_learner.scaler_dir is not None and 
                meta_learner.scaler_price is not None)

if xgb_available:
    print("✅ XGBoost Meta-Learner พร้อมใช้งาน")
else:
    print("⚠️ XGBoost Meta-Learner ยังไม่พร้อม - จะใช้ Dynamic Weight แทน")
    missing_files = []
    if meta_learner.xgb_clf is None:
        missing_files.append("XGBoost Classifier")
    if meta_learner.xgb_reg is None:
        missing_files.append("XGBoost Regressor") 
    if meta_learner.scaler_dir is None:
        missing_files.append("Direction Scaler")
    if meta_learner.scaler_price is None:
        missing_files.append("Price Scaler")
    print(f"   ขาดไฟล์: {missing_files}")
    print("   💡 หากต้องการใช้ XGBoost Meta-Learner:")
    print("      1. รันโค้ดเทรน XGBoost ก่อน")
    print("      2. ตรวจสอบว่าไฟล์ .pkl ถูกสร้างใน directory เดียวกัน")

def fetch_latest_data():
    """ดึงข้อมูลล่าสุดจากฐานข้อมูล"""
    try:
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
            AND StockDetail.Date >= CURDATE() - INTERVAL 350 DAY
            ORDER BY StockDetail.StockSymbol, StockDetail.Date ASC;
        """

        df = pd.read_sql(query, engine)
        engine.dispose()

        if df.empty:
            print("❌ ไม่มีข้อมูลหุ้นสำหรับตลาดที่กำลังเปิดอยู่")
            return df

        # Data processing
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
            merged = merged.ffill()
            filled_dfs.append(merged)
        
        df = pd.concat(filled_dfs, ignore_index=True)
        
        # Calculate technical indicators for each stock - แก้ไข DeprecationWarning
        def calculate_indicators(group):
            if len(group) < 14:
                return group
                
            try:
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
                
            except Exception as e:
                print(f"⚠️ เกิดข้อผิดพลาดในการคำนวณ indicators สำหรับ {group['StockSymbol'].iloc[0] if not group.empty else 'Unknown'}: {e}")
            
            return group
        
        # Apply indicators calculation to each stock group - แก้ไข DeprecationWarning
        df = df.groupby('StockSymbol', group_keys=False).apply(calculate_indicators)
        df = df.reset_index(drop=True)
        
        # Handle missing values
        critical_columns = ['Open', 'High', 'Low', 'Close']
        df = df.dropna(subset=critical_columns)
        
        # Fill NaN values
        df = df.ffill().bfill()
        
        # Fill remaining NaN with 0 for technical indicators
        technical_columns = ['RSI', 'MACD', 'MACD_Signal', 'ATR', 
                            'Bollinger_High', 'Bollinger_Low', 'SMA_50', 'SMA_200',
                            'EMA_10', 'EMA_20', 'Keltner_High', 'Keltner_Low', 'Keltner_Middle',
                            'Chaikin_Vol', 'Donchian_High', 'Donchian_Low', 'PSAR']
        
        for col in technical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill remaining NaN with 0
        df = df.fillna(0)
        
        print(f"✅ ข้อมูลพร้อมใช้งาน: {len(df)} แถว, {len(df['StockSymbol'].unique())} หุ้น")
        print(f"📊 Technical indicators ที่คำนวณได้: {[col for col in technical_columns if col in df.columns]}")
        
        return df
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def calculate_dynamic_weights(df_ticker, price_weight_factor=0.6, direction_weight_factor=0.4):
    """
    คำนวณ dynamic weight ระหว่าง LSTM และ GRU ตาม performance ล่าสุด
    """
    
    # ใช้ข้อมูล 15 วันล่าสุดสำหรับคำนวณ weight
    recent_data = df_ticker.tail(15)
    
    if len(recent_data) < 5:
        # ถ้าข้อมูลไม่เพียงพอ ใช้ weight เท่าๆ กัน
        return 0.5, 0.5
    
    # ตรวจสอบว่ามี columns สำหรับคำนวณ accuracy หรือไม่
    required_cols = ['PredictionClose_LSTM', 'PredictionClose_GRU', 
                     'PredictionTrend_LSTM', 'PredictionTrend_GRU']
    
    if not all(col in df_ticker.columns for col in required_cols):
        print("⚠️ ไม่มีข้อมูล predictions เพียงพอสำหรับ dynamic weighting")
        return 0.5, 0.5
    
    # คำนวณ price performance
    try:
        # สร้าง predictions สำหรับ error calculation
        lstm_predictions = recent_data['PredictionClose_LSTM'].dropna()
        gru_predictions = recent_data['PredictionClose_GRU'].dropna()
        actual_prices = recent_data['Close'].dropna()
        
        if len(lstm_predictions) >= 3 and len(gru_predictions) >= 3 and len(actual_prices) >= 3:
            # ใช้ length ที่น้อยที่สุดเพื่อหลีกเลี่ยง index mismatch
            min_len = min(len(lstm_predictions), len(gru_predictions), len(actual_prices))
            
            if min_len >= 2:
                # คำนวณ MAE สำหรับ price predictions - แก้ไข index alignment
                lstm_pred_vals = lstm_predictions.iloc[:min_len-1].values
                gru_pred_vals = gru_predictions.iloc[:min_len-1].values
                actual_vals_next = actual_prices.iloc[1:min_len].values
                
                lstm_price_error = np.mean(np.abs(lstm_pred_vals - actual_vals_next))
                gru_price_error = np.mean(np.abs(gru_pred_vals - actual_vals_next))
                
                # คำนวณ direction accuracy - แก้ไข index alignment
                actual_vals_current = actual_prices.iloc[:min_len-1].values
                
                lstm_dir_pred = (lstm_pred_vals > actual_vals_current).astype(int)
                gru_dir_pred = (gru_pred_vals > actual_vals_current).astype(int)
                actual_dir = (actual_vals_next > actual_vals_current).astype(int)
                
                lstm_dir_acc = np.mean(lstm_dir_pred == actual_dir)
                gru_dir_acc = np.mean(gru_dir_pred == actual_dir)
                
                # คำนวณ weights
                total_price_error = lstm_price_error + gru_price_error
                if total_price_error > 0:
                    lstm_price_score = gru_price_error / total_price_error  # กลับค่า
                    gru_price_score = lstm_price_error / total_price_error
                else:
                    lstm_price_score = 0.5
                    gru_price_score = 0.5
                
                total_dir_acc = lstm_dir_acc + gru_dir_acc
                if total_dir_acc > 0:
                    lstm_dir_score = lstm_dir_acc / total_dir_acc
                    gru_dir_score = gru_dir_acc / total_dir_acc
                else:
                    lstm_dir_score = 0.5
                    gru_dir_score = 0.5
                
                # รวม weights
                lstm_weight = (price_weight_factor * lstm_price_score + 
                              direction_weight_factor * lstm_dir_score)
                gru_weight = (price_weight_factor * gru_price_score + 
                             direction_weight_factor * gru_dir_score)
                
                # Normalize
                total_weight = lstm_weight + gru_weight
                if total_weight > 0:
                    lstm_weight = lstm_weight / total_weight
                    gru_weight = gru_weight / total_weight
                else:
                    lstm_weight = 0.5
                    gru_weight = 0.5
                
                # Apply constraints
                lstm_weight = max(MIN_WEIGHT, min(MAX_WEIGHT, lstm_weight))
                gru_weight = max(MIN_WEIGHT, min(MAX_WEIGHT, gru_weight))
                
                # Re-normalize
                total_weight = lstm_weight + gru_weight
                lstm_weight = lstm_weight / total_weight
                gru_weight = gru_weight / total_weight
                
                return lstm_weight, gru_weight
            
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการคำนวณ dynamic weights: {e}")
    
    return 0.5, 0.5

def predict_future_day_with_meta(model_lstm, model_gru, df, feature_columns, 
                                scaler_features, scaler_target, ticker_encoder, seq_length):
    """
    ทำนายด้วย LSTM/GRU + XGBoost Meta-Learner
    """
    
    future_predictions = []
    tickers = df['StockSymbol'].unique()
    
    print("\n🔮 กำลังทำนายด้วย 3-Layer Ensemble (LSTM + GRU + XGBoost)...")

    for ticker in tickers:
        print(f"\n📊 กำลังทำนายสำหรับหุ้น: {ticker}")
        df_ticker = df[df['StockSymbol'] == ticker].sort_values('Date').reset_index(drop=True)

        if len(df_ticker) < seq_length:
            print(f"⚠️ ข้อมูลไม่เพียงพอสำหรับ {ticker}, ข้ามไป...")
            continue

        try:
            # 1. ทำนายด้วย LSTM และ GRU
            latest_data = df_ticker.iloc[-seq_length:]
            features_scaled = scaler_features.transform(latest_data[feature_columns])
            ticker_ids = latest_data["Ticker_ID"].values
            market_ids = latest_data["Market_ID"].values

            X_feat = features_scaled.reshape(1, seq_length, -1)
            X_ticker = ticker_ids.reshape(1, seq_length)
            X_market = market_ids.reshape(1, seq_length)

            # LSTM predictions
            pred_output_lstm = model_lstm.predict([X_feat, X_ticker, X_market], verbose=0)
            pred_price_lstm_scaled = np.squeeze(pred_output_lstm[0])
            pred_direction_lstm = np.squeeze(pred_output_lstm[1])
            pred_price_lstm = scaler_target.inverse_transform(pred_price_lstm_scaled.reshape(-1, 1)).flatten()[0]

            # GRU predictions
            pred_output_gru = model_gru.predict([X_feat, X_ticker, X_market], verbose=0)
            pred_price_gru_scaled = np.squeeze(pred_output_gru[0])
            pred_direction_gru = np.squeeze(pred_output_gru[1])
            pred_price_gru = scaler_target.inverse_transform(pred_price_gru_scaled.reshape(-1, 1)).flatten()[0]

            # 2. เตรียมข้อมูลสำหรับ XGBoost Meta-Learner
            meta_input = pd.DataFrame({
                'StockSymbol': [ticker],
                'Date': [df_ticker['Date'].max()],
                'Close': [df_ticker.iloc[-1]['Close']],
                'High': [df_ticker.iloc[-1]['High']] if 'High' in df_ticker.columns else [df_ticker.iloc[-1]['Close'] * 1.01],
                'Low': [df_ticker.iloc[-1]['Low']] if 'Low' in df_ticker.columns else [df_ticker.iloc[-1]['Close'] * 0.99],
                'PredictionClose_LSTM': [pred_price_lstm],
                'PredictionClose_GRU': [pred_price_gru],
                'PredictionTrend_LSTM': [1 if pred_direction_lstm > 0.5 else 0],
                'PredictionTrend_GRU': [1 if pred_direction_gru > 0.5 else 0]
            })
            
            # เพิ่มข้อมูลประวัติสำหรับ technical indicators
            historical_data = df_ticker.tail(30).copy()  # ใช้ 30 วันล่าสุด
            historical_data = pd.concat([historical_data, meta_input], ignore_index=True)
            
            # 3. ทำนายด้วย XGBoost Meta-Learner
            meta_predictions = meta_learner.predict_meta(historical_data)
            
            if 'XGB_Predicted_Price' in meta_predictions.columns:
                # ใช้ผลลัพธ์จาก XGBoost
                final_predicted_price = meta_predictions['XGB_Predicted_Price'].iloc[-1]
                final_predicted_direction = meta_predictions['XGB_Predicted_Direction'].iloc[-1]
                final_direction_prob = meta_predictions['XGB_Predicted_Direction_Proba'].iloc[-1]
                xgb_confidence = meta_predictions['XGB_Confidence'].iloc[-1]
                ensemble_method = "XGBoost Meta-Learner"
            else:
                # Fallback: ใช้ Dynamic Weight ระหว่าง LSTM และ GRU
                lstm_weight, gru_weight = calculate_dynamic_weights(df_ticker)
                final_predicted_price = lstm_weight * pred_price_lstm + gru_weight * pred_price_gru
                final_direction_prob = lstm_weight * pred_direction_lstm + gru_weight * pred_direction_gru
                final_predicted_direction = 1 if final_direction_prob > 0.5 else 0
                xgb_confidence = abs(final_direction_prob - 0.5) * 2
                ensemble_method = "Dynamic Weight Fallback"

            # 4. สร้างผลลัพธ์
            last_date = df_ticker['Date'].max()
            next_day = last_date + pd.Timedelta(days=1)
            current_close = df_ticker.iloc[-1]['Close']
            
            # Model agreement
            lstm_dir = 1 if pred_direction_lstm > 0.5 else 0
            gru_dir = 1 if pred_direction_gru > 0.5 else 0
            model_agreement = 1 if lstm_dir == gru_dir else 0
            
            # เพิ่มผลลัพธ์
            prediction_result = {
                'StockSymbol': ticker,
                'Date': next_day,
                'Predicted_Price': final_predicted_price,
                'Predicted_Direction': final_predicted_direction,
                'Direction_Probability': final_direction_prob,
                'XGB_Confidence': xgb_confidence,
                'Ensemble_Method': ensemble_method,
                'LSTM_Direction': lstm_dir,
                'GRU_Direction': gru_dir,
                'LSTM_Prediction': pred_price_lstm,
                'GRU_Prediction': pred_price_gru,
                'Last_Close': current_close,
                'Price_Change': final_predicted_price - current_close,
                'Price_Change_Percent': (final_predicted_price - current_close) / current_close * 100,
                'Model_Agreement': model_agreement
            }
            
            # แสดงข้อมูล debug สำหรับ XGBoost
            if 'XGB_Predicted_Price' in meta_predictions.columns:
                price_change_pct = prediction_result['Price_Change_Percent']
                direction_consistent = ((price_change_pct > 0 and final_predicted_direction == 1) or 
                                      (price_change_pct <= 0 and final_predicted_direction == 0))
                consistency_status = "✅" if direction_consistent else "❌"
                
                # แสดงข้อมูลการปรับราคา
                raw_price = meta_predictions['XGB_Predicted_Price_Raw'].iloc[-1] if 'XGB_Predicted_Price_Raw' in meta_predictions.columns else final_predicted_price
                raw_change = (raw_price - current_close) / current_close * 100
                
                print(f"    🎯 Direction: {int(final_predicted_direction)} (Confidence: {xgb_confidence:.3f})")
                print(f"    📊 Price: {raw_change:+.2f}% → {price_change_pct:+.2f}% {consistency_status}")
            
            future_predictions.append(prediction_result)
            
            print(f"✅ {ticker}: {ensemble_method} - "
                  f"Price: {final_predicted_price:.2f} "
                  f"({prediction_result['Price_Change_Percent']:.2f}%) "
                  f"Confidence: {xgb_confidence:.3f}")
                  
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการทำนาย {ticker}: {e}")
            continue

    return pd.DataFrame(future_predictions)

def save_predictions_simple(predictions_df):
    """
    บันทึกผลลัพธ์การพยากรณ์แบบเรียบง่าย
    เก็บ: วันที่, หุ้น, ราคาทำนาย (LSTM, GRU, Ensemble), ทิศทางทำนาย (LSTM, GRU, Ensemble)
    """
    if predictions_df.empty:
        print("❌ ไม่มีข้อมูลพยากรณ์ที่จะบันทึก")
        return False

    try:
        engine = sqlalchemy.create_engine(DB_CONNECTION)
        
        with engine.connect() as connection:
            success_count = 0
            created_count = 0
            updated_count = 0
            
            for _, row in predictions_df.iterrows():
                try:
                    # ตรวจสอบว่ามี record อยู่แล้วหรือไม่
                    check_query = sqlalchemy.text("""
                        SELECT COUNT(*) FROM StockDetail 
                        WHERE StockSymbol = :symbol AND Date = :date
                    """)
                    
                    result = connection.execute(check_query, {
                        'symbol': row['StockSymbol'],
                        'date': row['Date'].strftime('%Y-%m-%d')
                    })
                    exists = result.scalar()
                    
                    if exists > 0:
                        # อัปเดต predictions ทั้ง LSTM, GRU, และ Ensemble
                        update_query = sqlalchemy.text("""
                            UPDATE StockDetail
                            SET PredictionClose_LSTM = :lstm_price,
                                PredictionTrend_LSTM = :lstm_trend,
                                PredictionClose_GRU = :gru_price,
                                PredictionTrend_GRU = :gru_trend,
                                PredictionClose_Ensemble = :ensemble_price, 
                                PredictionTrend_Ensemble = :ensemble_trend
                            WHERE StockSymbol = :symbol AND Date = :date
                        """)
                        
                        connection.execute(update_query, {
                            'lstm_price': float(row.get('LSTM_Prediction', row['Predicted_Price'])),
                            'lstm_trend': int(row.get('LSTM_Direction', row['Predicted_Direction'])),
                            'gru_price': float(row.get('GRU_Prediction', row['Predicted_Price'])),
                            'gru_trend': int(row.get('GRU_Direction', row['Predicted_Direction'])),
                            'ensemble_price': float(row['Predicted_Price']),
                            'ensemble_trend': int(row['Predicted_Direction']),
                            'symbol': row['StockSymbol'],
                            'date': row['Date'].strftime('%Y-%m-%d')
                        })
                        print(f"✅ อัปเดต {row['StockSymbol']} (LSTM+GRU+Ensemble)")
                        updated_count += 1
                        
                    else:
                        # สร้าง record ใหม่พร้อม predictions ทั้งหมด
                        insert_query = sqlalchemy.text("""
                            INSERT INTO StockDetail 
                            (StockSymbol, Date, 
                             PredictionClose_LSTM, PredictionTrend_LSTM,
                             PredictionClose_GRU, PredictionTrend_GRU,
                             PredictionClose_Ensemble, PredictionTrend_Ensemble)
                            VALUES 
                            (:symbol, :date, 
                             :lstm_price, :lstm_trend,
                             :gru_price, :gru_trend,
                             :ensemble_price, :ensemble_trend)
                        """)
                        
                        connection.execute(insert_query, {
                            'symbol': row['StockSymbol'],
                            'date': row['Date'].strftime('%Y-%m-%d'),
                            'lstm_price': float(row.get('LSTM_Prediction', row['Predicted_Price'])),
                            'lstm_trend': int(row.get('LSTM_Direction', row['Predicted_Direction'])),
                            'gru_price': float(row.get('GRU_Prediction', row['Predicted_Price'])),
                            'gru_trend': int(row.get('GRU_Direction', row['Predicted_Direction'])),
                            'ensemble_price': float(row['Predicted_Price']),
                            'ensemble_trend': int(row['Predicted_Direction'])
                        })
                        print(f"✅ สร้างใหม่ {row['StockSymbol']} (LSTM+GRU+Ensemble)")
                        created_count += 1
                    
                    success_count += 1
                    
                except Exception as e:
                    print(f"⚠️ ข้อผิดพลาดสำหรับ {row['StockSymbol']}: {e}")
                    continue
            
            # Commit การเปลี่ยนแปลง
            connection.commit()
            
            print(f"\n✅ บันทึกผลลัพธ์สำเร็จ!")
            print(f"   📊 รวม: {success_count}/{len(predictions_df)} รายการ")
            if updated_count > 0:
                print(f"   🔄 อัปเดต: {updated_count} รายการ")
            if created_count > 0:
                print(f"   ➕ สร้างใหม่: {created_count} รายการ")
            print(f"   💾 บันทึก: LSTM + GRU + Ensemble predictions")
            
            return success_count > 0
            
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อฐานข้อมูล: {e}")
        import traceback
        traceback.print_exc()
        return False

# ======================== MAIN EXECUTION ========================

if __name__ == "__main__":
    print("\n🚀 เริ่มต้นระบบทำนายหุ้นแบบ Enhanced 3-Layer Ensemble")
    
    # โหลดโมเดล LSTM และ GRU
    print("🤖 กำลังโหลดโมเดล LSTM และ GRU...")
    
    if not os.path.exists(MODEL_LSTM_PATH):
        print(f"❌ ไม่พบไฟล์โมเดล LSTM ที่ {MODEL_LSTM_PATH}")
        exit()
    
    if not os.path.exists(MODEL_GRU_PATH):
        print(f"❌ ไม่พบไฟล์โมเดล GRU ที่ {MODEL_GRU_PATH}")
        exit()
    
    try:
        model_lstm = load_model(MODEL_LSTM_PATH, compile=False)
        model_gru = load_model(MODEL_GRU_PATH, compile=False)
        print("✅ โหลดโมเดล LSTM และ GRU สำเร็จ!")
        
        # แสดงข้อมูลโมเดล
        print(f"📊 LSTM model: {len(model_lstm.layers)} layers")
        print(f"📊 GRU model: {len(model_gru.layers)} layers")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        exit()
    
    # ดึงและเตรียมข้อมูล
    print("📥 กำลังดึงข้อมูลจากฐานข้อมูล...")
    test_df = fetch_latest_data()
    
    if test_df.empty:
        print("❌ ไม่มีข้อมูลสำหรับประมวลผล")
        exit()
    
    print(f"📊 ได้รับข้อมูล: {len(test_df)} แถว จาก {len(test_df['StockSymbol'].unique())} หุ้น")
    print(f"📋 Columns ที่มีในข้อมูล: {list(test_df.columns)}")
    
    # เตรียม feature columns - ใช้เฉพาะ columns ที่มีอยู่จริง
    base_feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Change_Percent', 'Sentiment',
        'positive_news', 'negative_news', 'neutral_news',
        'TotalRevenue', 'QoQGrowth', 'EPS', 'ROE', 'NetProfitMargin', 
        'DebtToEquity', 'PERatio', 'Dividend_Yield', 'P_BV_Ratio'
    ]
    
    # Technical indicators ที่อาจมีหรือไม่มี
    potential_technical_columns = [
        'ATR', 'Keltner_High', 'Keltner_Low', 'Keltner_Middle', 'Chaikin_Vol',
        'Donchian_High', 'Donchian_Low', 'PSAR',
        'RSI', 'EMA_10', 'EMA_20', 'MACD', 'MACD_Signal', 
        'Bollinger_High', 'Bollinger_Low', 'SMA_50', 'SMA_200'
    ]
    
    # ตรวจสอบและใช้เฉพาะ columns ที่มีอยู่จริง
    available_columns = [col for col in base_feature_columns if col in test_df.columns]
    available_technical = [col for col in potential_technical_columns if col in test_df.columns]
    
    feature_columns = available_columns + available_technical
    
    print(f"📋 Available feature columns ({len(feature_columns)}): {feature_columns}")
    missing_cols = set(base_feature_columns + potential_technical_columns) - set(feature_columns)
    if missing_cols:
        print(f"⚠️ Missing columns: {missing_cols}")
    
    # ตรวจสอบข้อมูลสำคัญ
    if len(feature_columns) < 10:
        print("❌ ข้อมูล features ไม่เพียงพอ ต้องการอย่างน้อย 10 columns")
        exit()
    
    # ตรวจสอบว่ามีข้อมูล predictions จาก LSTM/GRU หรือไม่
    prediction_cols = ['PredictionClose_LSTM', 'PredictionClose_GRU', 
                      'PredictionTrend_LSTM', 'PredictionTrend_GRU']
    available_predictions = [col for col in prediction_cols if col in test_df.columns]
    print(f"🔮 Available prediction columns: {available_predictions}")
    
    if len(available_predictions) < 4:
        print("⚠️ ไม่มีข้อมูล predictions จาก LSTM/GRU ครบ, XGBoost Meta-Learner จะไม่สามารถทำงานได้")
    
    # ตั้งค่า encoders และ scalers
    us_stock = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'TSM', 'AMD']
    thai_stock = ['ADVANC', 'INTUCH', 'TRUE', 'DITTO', 'DIF', 
                  'INSET', 'JMART', 'INET', 'JAS', 'HUMAN']
    test_df['Market_ID'] = test_df['StockSymbol'].apply(
        lambda x: "US" if x in us_stock else "TH" if x in thai_stock else "OTHER"
    )
    
    # ตรวจสอบการกระจายของตลาด
    market_dist = test_df['Market_ID'].value_counts()
    print(f"📈 การกระจายตามตลาด: {dict(market_dist)}")
    
    scaler_main_features = RobustScaler()
    scaler_main_target = RobustScaler()
    ticker_encoder = LabelEncoder()
    market_encoder = LabelEncoder()
    
    try:
        test_df["Ticker_ID"] = ticker_encoder.fit_transform(test_df["StockSymbol"])
        test_df['Market_ID'] = market_encoder.fit_transform(test_df['Market_ID'])
        
        # ตรวจสอบข้อมูลก่อน fit scaler
        feature_data = test_df[feature_columns]
        if feature_data.isnull().any().any():
            print("⚠️ พบ NaN ในข้อมูล features, กำลังจัดการ...")
            feature_data = feature_data.fillna(feature_data.mean())
            test_df[feature_columns] = feature_data
        
        scaler_main_features.fit(test_df[feature_columns])
        scaler_main_target.fit(test_df[["Close"]])
        
        print("✅ เตรียมข้อมูลและ scalers สำเร็จ")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการเตรียมข้อมูล: {e}")
        print("🔍 กำลังตรวจสอบข้อมูลเพิ่มเติม...")
        print(f"   Shape ของข้อมูล: {test_df.shape}")
        print(f"   Data types: {test_df[feature_columns].dtypes}")
        print(f"   NaN counts: {test_df[feature_columns].isnull().sum()}")
        exit()
    
    # ทำนายด้วย Enhanced 3-Layer Ensemble
    future_predictions_df = predict_future_day_with_meta(
        model_lstm, model_gru, test_df, feature_columns, 
        scaler_main_features, scaler_main_target, ticker_encoder, SEQ_LENGTH
    )
    
    if not future_predictions_df.empty:
        print("\n🎯 ผลลัพธ์การทำนายด้วย Enhanced 3-Layer Ensemble:")
        
        # บันทึกผลลัพธ์
        output_path = 'enhanced_3layer_predictions.csv'
        future_predictions_df.to_csv(output_path, index=False)
        print(f"💾 บันทึกผลลัพธ์ใน {output_path}")
        
        # แสดงผลลัพธ์
        display_cols = ['StockSymbol', 'Date', 'Last_Close', 'Predicted_Price', 
                       'Price_Change_Percent', 'Predicted_Direction', 'XGB_Confidence',
                       'Ensemble_Method', 'Model_Agreement']
        
        print(future_predictions_df[display_cols])
        
        # ตรวจสอบ Direction Consistency
        print(f"\n🔍 การตรวจสอบ Direction Consistency:")
        consistent_count = 0
        total_count = len(future_predictions_df)
        
        for _, row in future_predictions_df.iterrows():
            price_change = row['Price_Change_Percent']
            predicted_dir = row['Predicted_Direction']
            is_consistent = ((price_change > 0 and predicted_dir == 1) or 
                           (price_change <= 0 and predicted_dir == 0))
            
            status = "✅" if is_consistent else "❌"
            
            print(f"   {row['StockSymbol']}: {price_change:+6.2f}% → Dir: {int(predicted_dir)} {status}")
            
            if is_consistent:
                consistent_count += 1
        
        consistency_rate = (consistent_count / total_count) * 100
        print(f"\n📊 Direction-Price Consistency: {consistent_count}/{total_count} ({consistency_rate:.1f}%)")
        
        print(f"\n💡 แนวทางการทำนาย: Direction-First Approach")
        print(f"   🎯 ใช้ Direction Classifier เป็นหลัก (สำคัญที่สุดในการลงทุน)")
        print(f"   📊 ปรับ Price ให้สอดคล้องกับ Direction ที่ทำนายได้")
        print(f"   ✅ รับประกัน Consistency = 100%")
        
        # สถิติการใช้งาน
        method_counts = future_predictions_df['Ensemble_Method'].value_counts()
        print(f"\n📊 สถิติการใช้ Ensemble Methods:")
        for method, count in method_counts.items():
            percentage = (count / len(future_predictions_df)) * 100
            print(f"   {method}: {count} หุ้น ({percentage:.1f}%)")
        
        # หุ้นที่มี confidence สูงสุด
        high_confidence = future_predictions_df.nlargest(3, 'XGB_Confidence')
        print(f"\n🏆 หุ้นที่มี Direction Confidence สูงสุด:")
        for _, row in high_confidence.iterrows():
            direction_text = "📈 BUY" if row['Predicted_Direction'] == 1 else "📉 SELL/SHORT"
            print(f"   {row['StockSymbol']}: {direction_text} (Confidence: {row['XGB_Confidence']:.3f}, "
                  f"Expected: {row['Price_Change_Percent']:.2f}%)")
        
        # แยกแสดงตามทิศทาง
        buy_signals = future_predictions_df[future_predictions_df['Predicted_Direction'] == 1]
        sell_signals = future_predictions_df[future_predictions_df['Predicted_Direction'] == 0]
        
        print(f"\n📈 BUY Signals ({len(buy_signals)} หุ้น):")
        if not buy_signals.empty:
            buy_sorted = buy_signals.sort_values('XGB_Confidence', ascending=False)
            for _, row in buy_sorted.iterrows():
                print(f"   {row['StockSymbol']}: +{row['Price_Change_Percent']:.2f}% (Confidence: {row['XGB_Confidence']:.3f})")
        
        print(f"\n📉 SELL/SHORT Signals ({len(sell_signals)} หุ้น):")
        if not sell_signals.empty:
            sell_sorted = sell_signals.sort_values('XGB_Confidence', ascending=False)
            for _, row in sell_sorted.iterrows():
                print(f"   {row['StockSymbol']}: {row['Price_Change_Percent']:.2f}% (Confidence: {row['XGB_Confidence']:.3f})")
        
        print(f"\n🎯 การใช้งานแนะนำ:")
        print(f"   • Confidence > 0.5: สัญญาณที่เชื่อถือได้")
        print(f"   • Confidence > 0.4: สัญญาณที่ดี")
        print(f"   • Confidence < 0.3: ควรระวัง")
        
        # บันทึกผลลัพธ์ลงฐานข้อมูล
        print(f"\n💾 กำลังบันทึกผลลัพธ์ลงฐานข้อมูล...")
        db_save_success = save_predictions_simple(future_predictions_df)
        
        if db_save_success:
            print("🔄 ข้อมูลในฐานข้อมูลได้รับการอัปเดตแล้ว")
            print("📱 สามารถใช้ข้อมูลทำนายในระบบอื่นๆ ได้แล้ว")
        else:
            print("⚠️ ไม่สามารถบันทึกลงฐานข้อมูลได้ แต่ยังมีไฟล์ CSV สำหรับใช้งาน")
    
    else:
        print("❌ ไม่มีข้อมูลพยากรณ์ที่จะแสดง")