import importlib
import logging
import sys
import time
import numpy as np
import pandas as pd
import sqlalchemy
import os
import tensorflow as tf
import ta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
from datetime import datetime, timedelta
import mysql.connector
from dotenv import load_dotenv
from tensorflow.keras.optimizers import Adam
from sqlalchemy import text
from sklearn.utils.class_weight import compute_class_weight
# XGBoost imports
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging
import warnings
import os
import xgboost as xgb
from sklearn.impute import SimpleImputer
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
import pickle
import io
import traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

with open('../LSTM_model/class_weights.pkl', 'rb') as f:
    class_weights_dict = pickle.load(f)

def focal_weighted_binary_crossentropy(class_weights, gamma=2.0, alpha_pos=0.7):
    def loss(y_true, y_pred):
        # Cast all inputs to float32 to avoid type mismatch
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        epsilon = tf.constant(1e-7, dtype=tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        
        # Cast class weights to float32
        weights = tf.where(
            tf.equal(y_true, 1.0), 
            tf.cast(class_weights[1], tf.float32), 
            tf.cast(class_weights[0], tf.float32)
        )
        
        # Cast alpha to float32
        alpha = tf.where(
            tf.equal(y_true, 1.0), 
            tf.cast(alpha_pos, tf.float32), 
            tf.cast(1 - alpha_pos, tf.float32)
        )
        
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1 - y_pred)
        
        # Cast gamma to float32
        focal_factor = tf.pow(1 - pt, tf.cast(gamma, tf.float32))
        
        # Ensure BCE is float32
        bce = tf.cast(tf.keras.losses.binary_crossentropy(y_true, y_pred), tf.float32)
        
        # Now all tensors are float32
        weighted_bce = bce * weights * alpha * focal_factor
        return tf.reduce_mean(weighted_bce)
    return loss

# Also update your quantile_loss function:
@tf.keras.utils.register_keras_serializable()
def quantile_loss(y_true, y_pred, quantile=0.5):
    # Cast to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    quantile = tf.cast(quantile, tf.float32)
    
    error = y_true - y_pred
    return tf.keras.backend.mean(tf.keras.backend.maximum(quantile * error, (quantile - 1) * error))

def get_model_name_from_path(model_path):
    """
    🔧 ดึงชื่อโมเดลจาก path
    """
    try:
        import os
        path_parts = model_path.split('/')
        
        # หาส่วนที่มี LSTM หรือ GRU
        for part in path_parts:
            if 'LSTM' in part.upper():
                return 'LSTM_Model'
            elif 'GRU' in part.upper():
                return 'GRU_Model'
        
        # ใช้ชื่อไฟล์
        filename = os.path.basename(model_path).replace('.keras', '').replace('.h5', '')
        return f"{filename}_Model"
    except Exception as e:
        print(f"⚠️ Error extracting model name from path: {e}")
        return "Unknown_Model"

def setup_model_names():
    """Setup model names from paths"""
    lstm_name = get_model_name_from_path(MODEL_LSTM_PATH)
    gru_name = get_model_name_from_path(MODEL_GRU_PATH)
    return lstm_name, gru_name

def assign_model_name(model, model_name):
    """
    🏷️ กำหนดชื่อให้โมเดลอย่างปลอดภัย
    """
    try:
        # ลองหลายวิธีในการตั้งชื่อ
        if hasattr(model, '_name'):
            model._name = model_name
        if hasattr(model, 'name'):
            model.name = model_name
        
        # เพิ่ม custom attribute
        model._model_type = model_name
        model._custom_name = model_name
        
        print(f"✅ กำหนดชื่อโมเดล: {model_name}")
        return True
    except Exception as e:
        print(f"⚠️ ไม่สามารถกำหนดชื่อโมเดลได้: {e}")
        return False

def best_practice_version(raw_price, current_price, direction_prob, model_uncertainty=None):
    """วิธีที่ดีที่สุด - ให้โมเดลเรียนรู้จากข้อผิดพลาดจริง"""
    
    # ใช้ probability แทน binary direction
    predicted_price = raw_price
    predicted_direction_prob = direction_prob
    
    # คำนวณ confidence interval ถ้ามี model uncertainty
    if model_uncertainty is not None:
        # คำนวณ confidence interval จาก model uncertainty
        price_lower = raw_price - (2 * model_uncertainty)  # 95% CI
        price_upper = raw_price + (2 * model_uncertainty)
    else:
        # ใช้ความผันผวนของราคาเป็น proxy
        price_volatility = abs(raw_price - current_price) * 0.1
        price_lower = raw_price - price_volatility
        price_upper = raw_price + price_volatility
    
    # สร้าง prediction object ที่มี uncertainty
    prediction_result = {
        'predicted_price': predicted_price,
        'price_lower_bound': price_lower,
        'price_upper_bound': price_upper,
        'direction_probability': predicted_direction_prob,
        'price_change_percent': (predicted_price - current_price) / current_price * 100,
        'model_confidence': abs(direction_prob - 0.5) * 2,  # 0 = no confidence, 1 = full confidence
        'uncertainty_range': abs(price_upper - price_lower),
        'raw_prediction_used': True,
        'adjustments_made': False
    }
    
    return prediction_result

# ======================== FIXED COLUMN MAPPING ========================

def standardize_column_names_to_training_format(df):
    """
    🔧 แปลง column names จาก database format เป็น training format
    เพื่อให้ scalers ทำงานได้ถูกต้อง - Enhanced version with data cleaning
    """
    print("🔄 Converting column names to match training format...")
    
    # ✅ Column mapping ที่ตรงกับ training code ทุกตัว
    training_column_mapping = {
        # Database format → Training format (ตรงตาม training code)
        'Change_Percent': 'Change (%)',
        'TotalRevenue': 'Total Revenue', 
        'QoQGrowth': 'QoQ Growth (%)',
        'EPS': 'Earnings Per Share (EPS)',
        'ROE': 'ROE (%)',
        'NetProfitMargin': 'Net Profit Margin (%)',
        'DebtToEquity': 'Debt to Equity',
        'PERatio': 'P/E Ratio',
        'P_BV_Ratio': 'P/BV Ratio',
        'Dividend_Yield': 'Dividend Yield (%)',
        # Technical indicators ที่อาจต่างกัน
        'EMA_10': 'EMA_10',  # เหมือนกัน
        'EMA_20': 'EMA_20',  # เหมือนกัน
        'SMA_50': 'SMA_50',  # เหมือนกัน
        'SMA_200': 'SMA_200', # เหมือนกัน
    }
    
    df_fixed = df.copy()
    
    # Apply column mapping with data type validation
    for db_name, training_name in training_column_mapping.items():
        if db_name in df_fixed.columns:
            # Clean and validate data before mapping
            try:
                # Handle mixed data types
                if df_fixed[db_name].dtype == 'object':
                    # Try to convert string data to numeric
                    print(f"   🔧 Converting {db_name} from object to numeric")
                    df_fixed[db_name] = pd.to_numeric(df_fixed[db_name], errors='coerce')
                
                # Replace infinite values
                df_fixed[db_name] = df_fixed[db_name].replace([np.inf, -np.inf], np.nan)
                
                # Fill NaN with appropriate values
                if df_fixed[db_name].isna().any():
                    if db_name in ['TotalRevenue', 'QoQGrowth', 'EPS', 'ROE', 'NetProfitMargin', 
                                   'DebtToEquity', 'PERatio', 'P_BV_Ratio', 'Dividend_Yield']:
                        # For financial metrics, use forward fill then 0
                        df_fixed[db_name] = df_fixed[db_name].ffill().fillna(0)
                    else:
                        # For technical indicators, use 0
                        df_fixed[db_name] = df_fixed[db_name].fillna(0)
                
                # Create the training format column
                df_fixed[training_name] = df_fixed[db_name].astype('float64')
                print(f"   🔄 Mapped: {db_name} → {training_name} (type: {df_fixed[training_name].dtype})")
                
            except Exception as e:
                print(f"   ⚠️ Error processing {db_name}: {e}, using default values")
                df_fixed[training_name] = 0.0
    
    return df_fixed

def use_training_feature_columns():
    """
    ✅ ใช้ feature_columns เหมือนกับ training code ทุกตัว
    """
    # ✅ เหมือนกับใน training code ทุกตัว
    training_feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment','positive_news','negative_news','neutral_news',
        'Total Revenue', 'QoQ Growth (%)','Earnings Per Share (EPS)','ROE (%)',
        'ATR','Keltner_High','Keltner_Low','Keltner_Middle','Chaikin_Vol','Donchian_High','Donchian_Low','PSAR',
        'Net Profit Margin (%)', 'Debt to Equity', 'P/E Ratio',
        'P/BV Ratio', 'Dividend Yield (%)','RSI', 'EMA_10', 'EMA_20', 'MACD', 'MACD_Signal',
        'Bollinger_High', 'Bollinger_Low','SMA_50', 'SMA_200'
    ]
    
    print(f"✅ Using training-compatible feature columns: {len(training_feature_columns)} features")
    return training_feature_columns

def calculate_technical_indicators_training_style(df):
    """
    🔧 คำนวณ technical indicators ตามแบบ training code
    (per-ticker grouping) - แก้ไขปัญหา length mismatch
    """
    print("🔧 Calculating technical indicators (training style)...")
    
    df_with_indicators = df.copy()
    
    # ✅ ตามแบบ training code - per ticker grouping
    df_with_indicators['Change (%)'] = df_with_indicators.groupby('StockSymbol')['Close'].pct_change() * 100
    
    # Clip outliers (ตามแบบ training)
    upper_bound = df_with_indicators['Change (%)'].quantile(0.99)
    lower_bound = df_with_indicators['Change (%)'].quantile(0.01)
    df_with_indicators['Change (%)'] = df_with_indicators['Change (%)'].clip(lower_bound, upper_bound)
    
    # ✅ RSI per ticker (ตามแบบ training) - ใช้ transform แทน apply
    def calculate_rsi_per_ticker(group):
        if len(group) >= 14:
            rsi = ta.momentum.RSIIndicator(group, window=14).rsi()
            # Fill ตามแบบ training
            rsi = rsi.fillna(rsi.rolling(window=5, min_periods=1).mean())
            return rsi
        else:
            return pd.Series([0] * len(group), index=group.index)
    
    # ใช้ transform เพื่อให้ได้ output ที่มีขนาดเท่ากับ input
    df_with_indicators['RSI'] = df_with_indicators.groupby('StockSymbol')['Close'].transform(
        calculate_rsi_per_ticker
    )
    
    # ✅ EMA per ticker (ตามแบบ training)
    df_with_indicators['EMA_12'] = df_with_indicators.groupby('StockSymbol')['Close'].transform(
        lambda x: x.ewm(span=12, adjust=False).mean()
    )
    df_with_indicators['EMA_26'] = df_with_indicators.groupby('StockSymbol')['Close'].transform(
        lambda x: x.ewm(span=26, adjust=False).mean()
    )
    df_with_indicators['EMA_10'] = df_with_indicators.groupby('StockSymbol')['Close'].transform(
        lambda x: x.ewm(span=10, adjust=False).mean()
    )
    df_with_indicators['EMA_20'] = df_with_indicators.groupby('StockSymbol')['Close'].transform(
        lambda x: x.ewm(span=20, adjust=False).mean()
    )
    
    # ✅ SMA per ticker (ตามแบบ training)
    df_with_indicators['SMA_50'] = df_with_indicators.groupby('StockSymbol')['Close'].transform(
        lambda x: x.rolling(window=50).mean()
    )
    df_with_indicators['SMA_200'] = df_with_indicators.groupby('StockSymbol')['Close'].transform(
        lambda x: x.rolling(window=200).mean()
    )
    
    # ✅ MACD (ตามแบบ training)
    df_with_indicators['MACD'] = df_with_indicators['EMA_12'] - df_with_indicators['EMA_26']
    df_with_indicators['MACD_Signal'] = df_with_indicators.groupby('StockSymbol')['MACD'].transform(
        lambda x: x.rolling(window=9).mean()
    )
    
    # ✅ Bollinger Bands per ticker - ใช้วิธีที่ปลอดภัย
    def safe_calculate_bollinger(group):
        try:
            if len(group) >= 20:
                bollinger = ta.volatility.BollingerBands(group, window=20, window_dev=2)
                return pd.DataFrame({
                    'Bollinger_High': bollinger.bollinger_hband(),
                    'Bollinger_Low': bollinger.bollinger_lband()
                }, index=group.index)
            else:
                return pd.DataFrame({
                    'Bollinger_High': [0] * len(group),
                    'Bollinger_Low': [0] * len(group)
                }, index=group.index)
        except Exception as e:
            print(f"      ⚠️ Bollinger calculation error: {e}, using default values")
            return pd.DataFrame({
                'Bollinger_High': [0] * len(group),
                'Bollinger_Low': [0] * len(group)
            }, index=group.index)
    
    # ใช้ apply แต่จัดการ index ให้ถูกต้อง
    bollinger_results = []
    for ticker in df_with_indicators['StockSymbol'].unique():
        ticker_data = df_with_indicators[df_with_indicators['StockSymbol'] == ticker]['Close']
        bollinger_df = safe_calculate_bollinger(ticker_data)
        bollinger_results.append(bollinger_df)
    
    # Concatenate และ sort ตาม original index
    if bollinger_results:
        bollinger_combined = pd.concat(bollinger_results).sort_index()
        df_with_indicators['Bollinger_High'] = bollinger_combined['Bollinger_High']
        df_with_indicators['Bollinger_Low'] = bollinger_combined['Bollinger_Low']
    else:
        df_with_indicators['Bollinger_High'] = 0
        df_with_indicators['Bollinger_Low'] = 0
    
    # ✅ ATR per ticker - ใช้วิธีที่ปลอดภัย
    def safe_calculate_atr(group_data):
        try:
            if len(group_data) >= 14 and all(col in group_data.columns for col in ['High', 'Low', 'Close']):
                atr = ta.volatility.AverageTrueRange(
                    high=group_data['High'], low=group_data['Low'], close=group_data['Close'], window=14
                )
                return atr.average_true_range()
            else:
                return pd.Series([0] * len(group_data), index=group_data.index)
        except Exception as e:
            print(f"      ⚠️ ATR calculation error: {e}, using default values")
            return pd.Series([0] * len(group_data), index=group_data.index)
    
    atr_results = []
    for ticker in df_with_indicators['StockSymbol'].unique():
        ticker_data = df_with_indicators[df_with_indicators['StockSymbol'] == ticker]
        atr_series = safe_calculate_atr(ticker_data)
        atr_results.append(atr_series)
    
    if atr_results:
        atr_combined = pd.concat(atr_results).sort_index()
        df_with_indicators['ATR'] = atr_combined
    else:
        df_with_indicators['ATR'] = 0
    
    # ✅ Keltner Channel per ticker - ใช้วิธีที่ปลอดภัย
    def safe_calculate_keltner(group_data):
        try:
            if len(group_data) >= 20 and all(col in group_data.columns for col in ['High', 'Low', 'Close']):
                keltner = ta.volatility.KeltnerChannel(
                    high=group_data['High'], low=group_data['Low'], close=group_data['Close'], 
                    window=20, window_atr=10
                )
                return pd.DataFrame({
                    'Keltner_High': keltner.keltner_channel_hband(),
                    'Keltner_Low': keltner.keltner_channel_lband(),
                    'Keltner_Middle': keltner.keltner_channel_mband()
                }, index=group_data.index)
            else:
                return pd.DataFrame({
                    'Keltner_High': [0] * len(group_data),
                    'Keltner_Low': [0] * len(group_data),
                    'Keltner_Middle': [0] * len(group_data)
                }, index=group_data.index)
        except Exception as e:
            print(f"      ⚠️ Keltner calculation error: {e}, using default values")
            return pd.DataFrame({
                'Keltner_High': [0] * len(group_data),
                'Keltner_Low': [0] * len(group_data),
                'Keltner_Middle': [0] * len(group_data)
            }, index=group_data.index)
    
    keltner_results = []
    for ticker in df_with_indicators['StockSymbol'].unique():
        ticker_data = df_with_indicators[df_with_indicators['StockSymbol'] == ticker]
        keltner_df = safe_calculate_keltner(ticker_data)
        keltner_results.append(keltner_df)
    
    if keltner_results:
        keltner_combined = pd.concat(keltner_results).sort_index()
        df_with_indicators['Keltner_High'] = keltner_combined['Keltner_High']
        df_with_indicators['Keltner_Low'] = keltner_combined['Keltner_Low']
        df_with_indicators['Keltner_Middle'] = keltner_combined['Keltner_Middle']
    else:
        df_with_indicators['Keltner_High'] = 0
        df_with_indicators['Keltner_Low'] = 0
        df_with_indicators['Keltner_Middle'] = 0
    
    # ✅ Chaikin Volatility per ticker - ใช้วิธีที่ปลอดภัย
    def safe_calculate_chaikin(group_data):
        try:
            window_cv = 10
            if len(group_data) >= window_cv and all(col in group_data.columns for col in ['High', 'Low']):
                high_low_diff = group_data['High'] - group_data['Low']
                high_low_ema = high_low_diff.ewm(span=window_cv, adjust=False).mean()
                chaikin_vol = high_low_ema.pct_change(periods=window_cv) * 100
                return chaikin_vol
            else:
                return pd.Series([0] * len(group_data), index=group_data.index)
        except Exception as e:
            print(f"      ⚠️ Chaikin calculation error: {e}, using default values")
            return pd.Series([0] * len(group_data), index=group_data.index)
    
    chaikin_results = []
    for ticker in df_with_indicators['StockSymbol'].unique():
        ticker_data = df_with_indicators[df_with_indicators['StockSymbol'] == ticker]
        chaikin_series = safe_calculate_chaikin(ticker_data)
        chaikin_results.append(chaikin_series)
    
    if chaikin_results:
        chaikin_combined = pd.concat(chaikin_results).sort_index()
        df_with_indicators['Chaikin_Vol'] = chaikin_combined
    else:
        df_with_indicators['Chaikin_Vol'] = 0
    
    # ✅ Donchian Channel per ticker - ใช้วิธีที่ปลอดภัย
    def safe_calculate_donchian(group_data):
        try:
            window_dc = 20
            if len(group_data) >= window_dc and all(col in group_data.columns for col in ['High', 'Low']):
                return pd.DataFrame({
                    'Donchian_High': group_data['High'].rolling(window=window_dc).max(),
                    'Donchian_Low': group_data['Low'].rolling(window=window_dc).min()
                }, index=group_data.index)
            else:
                return pd.DataFrame({
                    'Donchian_High': [0] * len(group_data),
                    'Donchian_Low': [0] * len(group_data)
                }, index=group_data.index)
        except Exception as e:
            print(f"      ⚠️ Donchian calculation error: {e}, using default values")
            return pd.DataFrame({
                'Donchian_High': [0] * len(group_data),
                'Donchian_Low': [0] * len(group_data)
            }, index=group_data.index)
    
    donchian_results = []
    for ticker in df_with_indicators['StockSymbol'].unique():
        ticker_data = df_with_indicators[df_with_indicators['StockSymbol'] == ticker]
        donchian_df = safe_calculate_donchian(ticker_data)
        donchian_results.append(donchian_df)
    
    if donchian_results:
        donchian_combined = pd.concat(donchian_results).sort_index()
        df_with_indicators['Donchian_High'] = donchian_combined['Donchian_High']
        df_with_indicators['Donchian_Low'] = donchian_combined['Donchian_Low']
    else:
        df_with_indicators['Donchian_High'] = 0
        df_with_indicators['Donchian_Low'] = 0
    
    # ✅ PSAR per ticker - แก้ไขปัญหา length mismatch
    def safe_calculate_psar(group_data):
        try:
            if len(group_data) >= 20 and all(col in group_data.columns for col in ['High', 'Low', 'Close']):
                psar = ta.trend.PSARIndicator(
                    high=group_data['High'], low=group_data['Low'], close=group_data['Close'], 
                    step=0.02, max_step=0.2
                )
                psar_result = psar.psar()
                # ตรวจสอบขนาดและ return เฉพาะ index ที่ตรงกัน
                if len(psar_result) == len(group_data):
                    return psar_result
                else:
                    # ถ้าขนาดไม่ตรงกัน ให้ reindex
                    return psar_result.reindex(group_data.index, fill_value=0)
            else:
                return pd.Series([0] * len(group_data), index=group_data.index)
        except Exception as e:
            print(f"      ⚠️ PSAR calculation error: {e}, using default values")
            return pd.Series([0] * len(group_data), index=group_data.index)
    
    # แก้ไขปัญหา PSAR - ใช้วิธีที่ปลอดภัย
    psar_results = []
    for ticker in df_with_indicators['StockSymbol'].unique():
        ticker_data = df_with_indicators[df_with_indicators['StockSymbol'] == ticker]
        psar_series = safe_calculate_psar(ticker_data)
        psar_results.append(psar_series)
    
    if psar_results:
        psar_combined = pd.concat(psar_results).sort_index()
        # ตรวจสอบขนาดก่อน assign
        if len(psar_combined) == len(df_with_indicators):
            df_with_indicators['PSAR'] = psar_combined
        else:
            print(f"⚠️ PSAR size mismatch: {len(psar_combined)} vs {len(df_with_indicators)}, using default values")
            df_with_indicators['PSAR'] = 0
    else:
        df_with_indicators['PSAR'] = 0
    
    # ✅ Handle infinite values (ตามแบบ training)
    stock_columns = [
        'RSI', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'Bollinger_High',
        'Bollinger_Low', 'ATR', 'Keltner_High', 'Keltner_Low', 'Keltner_Middle',
        'Chaikin_Vol', 'Donchian_High', 'Donchian_Low', 'PSAR', 'SMA_50', 'SMA_200'
    ]
    
    available_stock_cols = [col for col in stock_columns if col in df_with_indicators.columns]
    
    # Forward fill per ticker (ตามแบบ training)
    for ticker in df_with_indicators['StockSymbol'].unique():
        ticker_mask = df_with_indicators['StockSymbol'] == ticker
        df_with_indicators.loc[ticker_mask, available_stock_cols] = \
            df_with_indicators.loc[ticker_mask, available_stock_cols].ffill()

    
    # Final fillna (ตามแบบ training)
    df_with_indicators = df_with_indicators.fillna(0)
    
    print(f"✅ Technical indicators calculated (training style): {len(available_stock_cols)} indicators")
    return df_with_indicators

# ======================== FIXED DATA PROCESSING PIPELINE ========================

def process_data_training_compatible_enhanced(raw_df):
    """
    🔧 ประมวลผลข้อมูลให้เข้ากันได้กับ training scalers 100% - Enhanced version
    """
    print("\n🔧 Processing data for training compatibility (Enhanced)...")
    
    # Step 1: Enhanced column name standardization
    df_processed = standardize_column_names_to_training_format(raw_df)
    
    # Step 2: Calculate technical indicators (training style)
    df_processed = calculate_technical_indicators_training_style(df_processed)
    
    # Step 3: Add required columns ที่ training ต้องการ
    if 'Ticker' not in df_processed.columns and 'StockSymbol' in df_processed.columns:
        df_processed['Ticker'] = df_processed['StockSymbol']
    
    # Step 4: Enhanced data cleaning
    training_features = use_training_feature_columns()
    df_processed = clean_data_for_scalers_enhanced(df_processed, training_features)
    
    # Step 5: Final feature validation
    missing_features = [col for col in training_features if col not in df_processed.columns]
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
        # เพิ่ม missing features ด้วยค่า 0
        for col in missing_features:
            df_processed[col] = 0.0
            print(f"   Added {col} = 0.0")
    
    available_features = [col for col in training_features if col in df_processed.columns]
    print(f"✅ Available training features: {len(available_features)}/{len(training_features)}")
    
    # Step 6: Final data type and value validation
    print("🔍 Final validation...")
    for col in training_features:
        if col in df_processed.columns:
            # Ensure numeric type
            if not pd.api.types.is_numeric_dtype(df_processed[col]):
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
            
            # Ensure float64 type for consistency
            df_processed[col] = df_processed[col].astype('float64')
            
            # Final sanity checks
            if df_processed[col].isna().any() or np.isinf(df_processed[col]).any():
                df_processed[col] = df_processed[col].replace([np.inf, -np.inf, np.nan], 0)
    
    print(f"✅ Enhanced processing completed: {len(df_processed)} rows")
    return df_processed, training_features

# ======================== FIXED PREDICTION FUNCTION ========================

def predict_with_training_compatible_scalers(model_lstm, model_gru, df, feature_columns, 
                                           ticker_scalers, ticker_encoder, market_encoder, seq_length=10):
    """
    🔧 ทำนายด้วย scalers ที่เข้ากันได้กับ training 100%
    แก้ไขปัญหาราคาที่ทำนายผิดปกติ และ data type issues
    """
    print(f"\n🔧 Fixed Prediction with Training-Compatible Scalers...")
    print(f"   ✅ Using exact same feature columns as training")
    print(f"   ✅ Using exact same column names as training")
    print(f"   ✅ Using exact same preprocessing as training")
    print(f"   🔧 Added data type validation and cleaning")
    
    future_predictions = []
    tickers = df['StockSymbol'].unique()
    
    def clean_and_validate_data(data, columns, ticker_name):
        """Clean and validate data before scaling"""
        cleaned_data = data[columns].copy()
        
        # Check data types and convert to numeric
        for col in columns:
            if col in cleaned_data.columns:
                try:
                    # Check if column contains non-numeric data
                    if cleaned_data[col].dtype == 'object':
                        print(f"      🔧 Converting {col} from object to numeric")
                        cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
                    
                    # Handle string-like values that might be in numeric columns
                    if cleaned_data[col].dtype == 'object' or str(cleaned_data[col].dtype).startswith('string'):
                        print(f"      ⚠️ Found string values in {col}, converting...")
                        cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
                    
                    # Replace inf and -inf with NaN
                    cleaned_data[col] = cleaned_data[col].replace([np.inf, -np.inf], np.nan)
                    
                    # Fill NaN with column mean or 0
                    if cleaned_data[col].isna().any():
                        col_mean = cleaned_data[col].mean()
                        if pd.isna(col_mean):
                            cleaned_data[col] = cleaned_data[col].fillna(0)
                        else:
                            cleaned_data[col] = cleaned_data[col].fillna(col_mean)
                    
                    # Final validation: ensure all values are numeric
                    if not pd.api.types.is_numeric_dtype(cleaned_data[col]):
                        print(f"      ❌ {col} still not numeric after conversion, using default values")
                        cleaned_data[col] = 0.0
                    
                except Exception as e:
                    print(f"      ❌ Error processing column {col}: {e}, using default values")
                    cleaned_data[col] = 0.0
            else:
                print(f"      ⚠️ Column {col} not found, adding with default value 0")
                cleaned_data[col] = 0.0
        
        # Final data type validation
        for col in cleaned_data.columns:
            if not pd.api.types.is_numeric_dtype(cleaned_data[col]):
                print(f"      🔧 Final conversion of {col} to float64")
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce').fillna(0).astype('float64')
        
        return cleaned_data
    
    for ticker in tickers:
        print(f"\n📊 Predicting {ticker} (training-compatible with validation)...")
        
        df_ticker = df[df['StockSymbol'] == ticker].sort_values('Date').reset_index(drop=True)
        
        if len(df_ticker) < seq_length:
            print(f"   ⚠️ Insufficient data: {len(df_ticker)} < {seq_length}")
            continue
        
        try:
            # Get latest data
            latest_data = df_ticker.iloc[-seq_length:].copy()
            ticker_id = latest_data['Ticker_ID'].iloc[-1]
            
            # Check scaler availability
            if ticker_id not in ticker_scalers:
                print(f"   ❌ No scaler for {ticker} (ID: {ticker_id})")
                continue
            
            # ✅ Use training-compatible scaler
            scaler_info = ticker_scalers[ticker_id]
            feature_scaler = scaler_info['feature_scaler']
            price_scaler = scaler_info['price_scaler']
            
            print(f"   🔧 Using training scaler for {ticker}:")
            print(f"      📊 Feature scaler: {type(feature_scaler).__name__}")
            print(f"      💰 Price scaler: {type(price_scaler).__name__}")
            
            # ✅ Clean and validate feature data
            print(f"      🔧 Cleaning and validating feature data...")
            features_data = clean_and_validate_data(latest_data, feature_columns, ticker)
            
            print(f"      📊 Cleaned feature shape: {features_data.shape}")
            print(f"      📊 Feature data types: {features_data.dtypes.nunique()} unique types")
            print(f"      📊 Feature range: {features_data.min().min():.3f} to {features_data.max().max():.3f}")
            
            # Validate that all features are numeric
            non_numeric_cols = []
            for col in features_data.columns:
                if not pd.api.types.is_numeric_dtype(features_data[col]):
                    non_numeric_cols.append(col)
            
            if non_numeric_cols:
                print(f"      ❌ Non-numeric columns found: {non_numeric_cols}")
                # Force convert to numeric
                for col in non_numeric_cols:
                    features_data[col] = pd.to_numeric(features_data[col], errors='coerce').fillna(0)
                print(f"      ✅ Forced conversion completed")
            
            # ✅ Scale features using training scaler with additional validation
            try:
                print(f"      🔧 Applying feature scaling...")
                
                # Ensure data is in the right format for scaler
                features_array = features_data.values.astype(np.float64)
                
                # Validate array shape and content
                if np.any(np.isnan(features_array)):
                    print(f"      ⚠️ Found NaN values, replacing with 0")
                    features_array = np.nan_to_num(features_array, nan=0.0)
                
                if np.any(np.isinf(features_array)):
                    print(f"      ⚠️ Found infinite values, replacing with finite values")
                    features_array = np.nan_to_num(features_array, posinf=1e6, neginf=-1e6)
                
                features_scaled = feature_scaler.transform(features_array)
                print(f"      ✅ Feature scaling successful")
                
            except Exception as e:
                print(f"      ❌ Feature scaling failed: {e}")
                print(f"      Debug - Expected features: {getattr(feature_scaler, 'n_features_in_', 'unknown')}")
                print(f"      Debug - Got features: {features_data.shape[1]}")
                print(f"      Debug - Feature data types: {features_data.dtypes.to_dict()}")
                continue
            
            # Prepare inputs with validation
            try:
                ticker_ids = latest_data["Ticker_ID"].values.astype(np.int64)
                market_ids = latest_data["Market_ID"].values.astype(np.int64)
                
                X_feat = features_scaled.reshape(1, seq_length, -1).astype(np.float32)
                X_ticker = ticker_ids.reshape(1, seq_length).astype(np.int64)
                X_market = market_ids.reshape(1, seq_length).astype(np.int64)
                
                print(f"      📊 Input shapes: Features{X_feat.shape}, Ticker{X_ticker.shape}, Market{X_market.shape}")
                
            except Exception as e:
                print(f"      ❌ Input preparation failed: {e}")
                continue
            
            # === LSTM PREDICTION ===
            print(f"   🔴 LSTM Prediction...")
            try:
                pred_output_lstm = model_lstm.predict([X_feat, X_ticker, X_market], verbose=0)
                pred_price_lstm_scaled = np.squeeze(pred_output_lstm[0])
                pred_direction_lstm = np.squeeze(pred_output_lstm[1])
                
                # ✅ Inverse transform with training scaler
                pred_price_lstm = price_scaler.inverse_transform(
                    pred_price_lstm_scaled.reshape(-1, 1)
                ).flatten()[0]
                
                print(f"      💰 LSTM Price: {pred_price_lstm:.2f}")
                print(f"      📈 LSTM Direction: {pred_direction_lstm:.4f}")
                
            except Exception as e:
                print(f"      ❌ LSTM prediction failed: {e}")
                continue
            
            # === GRU PREDICTION ===
            print(f"   🔵 GRU Prediction...")
            try:
                pred_output_gru = model_gru.predict([X_feat, X_ticker, X_market], verbose=0)
                pred_price_gru_scaled = np.squeeze(pred_output_gru[0])
                pred_direction_gru = np.squeeze(pred_output_gru[1])
                
                # ✅ Inverse transform with training scaler
                pred_price_gru = price_scaler.inverse_transform(
                    pred_price_gru_scaled.reshape(-1, 1)
                ).flatten()[0]
                
                print(f"      💰 GRU Price: {pred_price_gru:.2f}")
                print(f"      📈 GRU Direction: {pred_direction_gru:.4f}")
                
            except Exception as e:
                print(f"      ❌ GRU prediction failed: {e}")
                continue
            
            # === SANITY CHECK ===
            current_price = df_ticker.iloc[-1]['Close']
            
            # Validate current_price
            try:
                current_price = float(current_price)
                if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
                    print(f"   ⚠️ Invalid current price: {current_price}, using last valid price")
                    valid_prices = df_ticker['Close'][df_ticker['Close'] > 0].dropna()
                    if len(valid_prices) > 0:
                        current_price = float(valid_prices.iloc[-1])
                    else:
                        print(f"   ❌ No valid prices found for {ticker}")
                        continue
            except:
                print(f"   ❌ Cannot convert current price to float for {ticker}")
                continue
            
            # Check for unrealistic predictions
            lstm_change_pct = abs((pred_price_lstm - current_price) / current_price * 100)
            gru_change_pct = abs((pred_price_gru - current_price) / current_price * 100)
            
            if lstm_change_pct > 100:  # More than 100% change
                print(f"   ⚠️ LSTM prediction seems unrealistic: {lstm_change_pct:.1f}% change")
                pred_price_lstm = current_price * (1 + np.sign(pred_price_lstm - current_price) * 0.1)  # Cap at 10%
                
            if gru_change_pct > 100:   # More than 100% change
                print(f"   ⚠️ GRU prediction seems unrealistic: {gru_change_pct:.1f}% change")
                pred_price_gru = current_price * (1 + np.sign(pred_price_gru - current_price) * 0.1)   # Cap at 10%
            
            # === CONSISTENCY CHECK ===
            price_diff = abs(pred_price_lstm - pred_price_gru)
            price_diff_pct = (price_diff / current_price) * 100
            
            direction_lstm = 1 if pred_direction_lstm > 0.5 else 0
            direction_gru = 1 if pred_direction_gru > 0.5 else 0
            direction_agreement = direction_lstm == direction_gru
            
            print(f"   🤝 Sanity Check:")
            print(f"      💰 Current Price: {current_price:.2f}")
            print(f"      💰 Price difference: {price_diff:.2f} ({price_diff_pct:.2f}%)")
            print(f"      📊 Direction agreement: {direction_agreement}")
            print(f"      📈 LSTM change: {((pred_price_lstm - current_price) / current_price * 100):+.2f}%")
            print(f"      📈 GRU change: {((pred_price_gru - current_price) / current_price * 100):+.2f}%")
            
            # === ENSEMBLE PREDICTION ===
            ensemble_price = (pred_price_lstm + pred_price_gru) / 2
            ensemble_direction_prob = (pred_direction_lstm + pred_direction_gru) / 2
            ensemble_direction = 1 if ensemble_direction_prob > 0.5 else 0
            
            # Calculate confidence
            confidence = min(
                1.0 - (price_diff_pct / 100),  # Price agreement
                abs(ensemble_direction_prob - 0.5) * 2  # Direction confidence
            )
            
            print(f"   🎯 Ensemble Result:")
            print(f"      💰 Price: {ensemble_price:.2f}")
            print(f"      📊 Direction: {ensemble_direction} (prob: {ensemble_direction_prob:.4f})")
            print(f"      🎯 Confidence: {confidence:.3f}")
            print(f"      📈 Expected change: {((ensemble_price - current_price) / current_price * 100):+.2f}%")
            
            # Store results
            last_date = df_ticker['Date'].max()
            next_date = last_date + pd.Timedelta(days=1)
            
            prediction_result = {
                'StockSymbol': ticker,
                'Date': next_date,
                'Current_Price': current_price,
                'LSTM_Price': pred_price_lstm,
                'GRU_Price': pred_price_gru,
                'Ensemble_Price': ensemble_price,
                'LSTM_Direction': direction_lstm,
                'GRU_Direction': direction_gru,
                'Ensemble_Direction': ensemble_direction,
                'LSTM_Direction_Prob': pred_direction_lstm,
                'GRU_Direction_Prob': pred_direction_gru,
                'Ensemble_Direction_Prob': ensemble_direction_prob,
                'Price_Difference_Pct': price_diff_pct,
                'Direction_Agreement': direction_agreement,
                'Confidence': confidence,
                'Scaler_Used': f"Training_Ticker_{ticker_id}",
                'Training_Compatible': True,
                'Data_Validation_Passed': True,
                'LSTM_Change_Pct': ((pred_price_lstm - current_price) / current_price * 100),
                'GRU_Change_Pct': ((pred_price_gru - current_price) / current_price * 100),
                'Ensemble_Change_Pct': ((ensemble_price - current_price) / current_price * 100)
            }
            
            future_predictions.append(prediction_result)
            print(f"   ✅ {ticker} prediction completed successfully")
            
        except Exception as e:
            print(f"   ❌ Error predicting {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n🎉 Training-compatible predictions completed: {len(future_predictions)}/{len(tickers)} stocks")
    return pd.DataFrame(future_predictions)


# ======================== WALK-FORWARD VALIDATION FUNCTION ========================

def walk_forward_validation_multi_task_batch(
    model,
    df,
    feature_columns,
    ticker_scalers,   # Dict ของ Scaler per Ticker
    ticker_encoder,
    market_encoder,
    seq_length=10,
    retrain_frequency=5,
    chunk_size = 200
):

    all_predictions = []
    chunk_metrics = []
    tickers = df['StockSymbol'].unique()

    for ticker in tickers:
        print(f"\nProcessing Ticker: {ticker}")
        df_ticker = df[df['StockSymbol'] == ticker].sort_values('Date').reset_index(drop=True)
        
        total_days = len(df_ticker)
        print(f"   📊 Total data available: {total_days} days")
        
        if total_days < chunk_size + seq_length:
            print(f"   ⚠️ Not enough data (need at least {chunk_size + seq_length} days), skipping...")
            continue
        
        # คำนวณจำนวน chunks ที่สามารถสร้างได้
        num_chunks = total_days // chunk_size
        remaining_days = total_days % chunk_size
        
        # เพิ่ม partial chunk ถ้าข้อมูลเพียงพอ
        if remaining_days > seq_length:
            num_chunks += 1
            
        print(f"   📦 Number of chunks: {num_chunks} (chunk_size={chunk_size})")
        
        ticker_predictions = []
        
        # ประมวลผลแต่ละ chunk
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_days)
            
            # ตรวจสอบขนาด chunk
            if (end_idx - start_idx) < seq_length + 1:
                print(f"      ⚠️ Chunk {chunk_idx + 1} too small ({end_idx - start_idx} days), skipping...")
                continue
                
            current_chunk = df_ticker.iloc[start_idx:end_idx].reset_index(drop=True)
            
            print(f"\n      📦 Processing Chunk {chunk_idx + 1}/{num_chunks}")
            print(f"         📅 Date range: {current_chunk['Date'].min()} to {current_chunk['Date'].max()}")
            print(f"         📈 Days: {len(current_chunk)} ({start_idx}-{end_idx})")
            
            # === Walk-Forward Validation ภายใน Chunk ===
            chunk_predictions = []
            batch_features = []
            batch_tickers = []
            batch_market = []
            batch_price = []
            batch_dir = []
            predictions_count = 0

            for i in range(len(current_chunk) - seq_length):
                historical_data = current_chunk.iloc[i : i + seq_length]
                target_data = current_chunk.iloc[i + seq_length]

                t_id = historical_data['Ticker_ID'].iloc[-1]
                if t_id not in ticker_scalers:
                    print(f"         ⚠️ Ticker_ID {t_id} not found in scalers, skipping...")
                    continue

                scaler_f = ticker_scalers[t_id]['feature_scaler']
                scaler_p = ticker_scalers[t_id]['price_scaler']

                # เตรียม input features
                features = historical_data[feature_columns].values
                ticker_ids = historical_data['Ticker_ID'].values
                market_ids = historical_data['Market_ID'].values

                try:
                    features_scaled = scaler_f.transform(features)
                except Exception as e:
                    print(f"         ⚠️ Feature scaling error: {e}")
                    continue

                X_features = features_scaled.reshape(1, seq_length, len(feature_columns))
                X_ticker = ticker_ids.reshape(1, seq_length)
                X_market = market_ids.reshape(1, seq_length)

                # ทำนาย
                try:
                    pred_output = model.predict([X_features, X_ticker, X_market], verbose=0)
                    pred_price_scaled = pred_output[0]
                    pred_dir_prob = pred_output[1]

                    predicted_price = scaler_p.inverse_transform(pred_price_scaled)[0][0]
                    predicted_dir = 1 if pred_dir_prob[0][0] >= 0.5 else 0

                    # เก็บผลลัพธ์
                    actual_price = target_data['Close']
                    future_date = target_data['Date']
                    last_close = historical_data.iloc[-1]['Close']
                    actual_dir = 1 if (target_data['Close'] > last_close) else 0

                    chunk_predictions.append({
                        'Ticker': ticker,
                        'Date': future_date,
                        'Chunk_Index': chunk_idx + 1,
                        'Position_in_Chunk': i + 1,
                        'Predicted_Price': predicted_price,
                        'Actual_Price': actual_price,
                        'Predicted_Dir': predicted_dir,
                        'Actual_Dir': actual_dir,
                        'Last_Close': last_close,
                        'Price_Change_Actual': actual_price - last_close,
                        'Price_Change_Predicted': predicted_price - last_close
                    })

                    predictions_count += 1

                    # เตรียมข้อมูลสำหรับ mini-retrain
                    batch_features.append(X_features)
                    batch_tickers.append(X_ticker)
                    batch_market.append(X_market)

                    y_price_true_scaled = scaler_p.transform(np.array([[actual_price]], dtype=float))
                    batch_price.append(y_price_true_scaled)

                    y_dir_true = np.array([actual_dir], dtype=float)
                    batch_dir.append(y_dir_true)

                    # 🔄 Mini-retrain ทุกๆ retrain_frequency วัน
                    if (i+1) % retrain_frequency == 0 or (i == (len(current_chunk) - seq_length - 1)):
                        if len(batch_features) > 0:
                            try:
                                print(f"            🔄 Mini-retrain at position {i+1}...")
                                
                                # รวม batch data และแปลงเป็น float32 ทั้งหมด
                                bf = np.concatenate(batch_features, axis=0).astype(np.float32)
                                bt = np.concatenate(batch_tickers, axis=0).astype(np.int32)
                                bm = np.concatenate(batch_market, axis=0).astype(np.int32)
                                bp = np.concatenate(batch_price, axis=0).astype(np.float32)
                                bd = np.array([np.array(d, dtype=np.float32) for d in batch_dir], dtype=np.float32)
                                if len(bd.shape) > 1:
                                    bd = bd.flatten()
                                bd = bd.astype(np.float32)

                                # Mini-retrain
                                model.fit(
                                    [bf, bt, bm], 
                                    {'price_output': bp, 'direction_output': bd}, 
                                    epochs=1, 
                                    batch_size=min(len(bf), 8), 
                                    verbose=0
                                )
                                
                                print(f"            ✅ Mini-retrain completed with {len(bf)} samples")
                                
                            except Exception as e:
                                print(f"            ❌ Mini-retrain error: {e}")

                            # ล้าง batch
                            batch_features = []
                            batch_tickers = []
                            batch_market = []
                            batch_price = []
                            batch_dir = []
                            
                except Exception as e:
                    print(f"         ⚠️ Prediction error at position {i}: {e}")
                    continue

            # คำนวณ metrics สำหรับ chunk นี้
            if chunk_predictions:
                chunk_df = pd.DataFrame(chunk_predictions)
                
                actual_prices = chunk_df['Actual_Price'].values
                pred_prices = chunk_df['Predicted_Price'].values
                actual_dirs = chunk_df['Actual_Dir'].values
                pred_dirs = chunk_df['Predicted_Dir'].values
                
                # คำนวณ metrics
                mae_val = mean_absolute_error(actual_prices, pred_prices)
                mse_val = mean_squared_error(actual_prices, pred_prices)
                rmse_val = np.sqrt(mse_val)
                r2_val = r2_score(actual_prices, pred_prices)
                dir_acc = accuracy_score(actual_dirs, pred_dirs)
                dir_f1 = f1_score(actual_dirs, pred_dirs, zero_division=0)
                
                # คำนวณ MAPE และ SMAPE (safe calculation)
                try:
                    mape_val = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
                except:
                    mape_val = 0
                    
                try:
                    smape_val = 100/len(actual_prices) * np.sum(2 * np.abs(pred_prices - actual_prices) / (np.abs(actual_prices) + np.abs(pred_prices)))
                except:
                    smape_val = 0

                chunk_metric = {
                    'Ticker': ticker,
                    'Chunk_Index': chunk_idx + 1,
                    'Chunk_Start_Date': current_chunk['Date'].min(),
                    'Chunk_End_Date': current_chunk['Date'].max(),
                    'Chunk_Days': len(current_chunk),
                    'Predictions_Count': predictions_count,
                    'MAE': mae_val,
                    'MSE': mse_val,
                    'RMSE': rmse_val,
                    'MAPE': mape_val,
                    'SMAPE': smape_val,
                    'R2_Score': r2_val,
                    'Direction_Accuracy': dir_acc,
                    'Direction_F1': dir_f1
                }
                
                chunk_metrics.append(chunk_metric)
                ticker_predictions.extend(chunk_predictions)
                
                print(f"         📊 Chunk results: {predictions_count} predictions")
                print(f"         📈 Direction accuracy: {dir_acc:.3f}")
                print(f"         📈 Price MAE: {mae_val:.3f}")
            
            print(f"         ✅ Chunk {chunk_idx + 1} completed with mini-retrain")
        
        all_predictions.extend(ticker_predictions)
        print(f"   ✅ Completed {ticker}: {len(ticker_predictions)} total predictions from {num_chunks} chunks")

    # รวบรวมและบันทึกผลลัพธ์
    print(f"\n📊 Processing complete!")
    print(f"   Total predictions: {len(all_predictions)}")
    print(f"   Total chunks processed: {len(chunk_metrics)}")
    
    if len(all_predictions) == 0:
        print("❌ No predictions generated!")
        return pd.DataFrame(), {}

    predictions_df = pd.DataFrame(all_predictions)
    
    # บันทึก predictions
    predictions_df.to_csv('predictions_chunk_walkforward.csv', index=False)
    print("💾 Saved predictions to 'predictions_chunk_walkforward.csv'")
    
    # บันทึก chunk metrics
    if chunk_metrics:
        chunk_metrics_df = pd.DataFrame(chunk_metrics)
        chunk_metrics_df.to_csv('chunk_metrics.csv', index=False)
        print("💾 Saved chunk metrics to 'chunk_metrics.csv'")

    # คำนวณ Overall Metrics ต่อ Ticker
    print("\n📊 Calculating overall metrics...")
    overall_metrics = {}
    
    for ticker, group in predictions_df.groupby('Ticker'):
        actual_prices = group['Actual_Price'].values
        pred_prices = group['Predicted_Price'].values
        actual_dirs = group['Actual_Dir'].values
        pred_dirs = group['Predicted_Dir'].values

        # คำนวณ metrics
        mae_val = mean_absolute_error(actual_prices, pred_prices)
        mse_val = mean_squared_error(actual_prices, pred_prices)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(actual_prices, pred_prices)

        dir_acc = accuracy_score(actual_dirs, pred_dirs)
        dir_f1 = f1_score(actual_dirs, pred_dirs, zero_division=0)
        dir_precision = precision_score(actual_dirs, pred_dirs, zero_division=0)
        dir_recall = recall_score(actual_dirs, pred_dirs, zero_division=0)

        # Safe MAPE และ SMAPE calculation
        try:
            mape_val = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
        except:
            mape_val = 0
            
        try:
            smape_val = 100/len(actual_prices) * np.sum(2 * np.abs(pred_prices - actual_prices) / (np.abs(actual_prices) + np.abs(pred_prices)))
        except:
            smape_val = 0

        overall_metrics[ticker] = {
            'Total_Predictions': len(group),
            'Number_of_Chunks': len(group['Chunk_Index'].unique()),
            'MAE': mae_val,
            'MSE': mse_val,
            'RMSE': rmse_val,
            'MAPE': mape_val,
            'SMAPE': smape_val,
            'R2_Score': r2_val,
            'Direction_Accuracy': dir_acc,
            'Direction_F1_Score': dir_f1,
            'Direction_Precision': dir_precision,
            'Direction_Recall': dir_recall
        }

    # บันทึก overall metrics
    overall_metrics_df = pd.DataFrame.from_dict(overall_metrics, orient='index')
    overall_metrics_df.to_csv('overall_metrics_per_ticker.csv')
    print("💾 Saved overall metrics to 'overall_metrics_per_ticker.csv'")

    # สรุปผลลัพธ์
    print(f"\n🎯 Summary:")
    print(f"   📈 Tickers processed: {len(predictions_df['Ticker'].unique())}")
    print(f"   📈 Average predictions per ticker: {len(predictions_df)/len(predictions_df['Ticker'].unique()):.1f}")
    print(f"   📈 Average chunks per ticker: {len(chunk_metrics)/len(predictions_df['Ticker'].unique()):.1f}")
    
    if chunk_metrics:
        avg_chunk_acc = np.mean([c['Direction_Accuracy'] for c in chunk_metrics])
        avg_chunk_mae = np.mean([c['MAE'] for c in chunk_metrics])
        print(f"   📈 Average chunk direction accuracy: {avg_chunk_acc:.3f}")
        print(f"   📈 Average chunk MAE: {avg_chunk_mae:.3f}")

    print(f"\n📁 Files generated:")
    print(f"   📄 predictions_chunk_walkforward.csv - All predictions with chunk info")
    print(f"   📄 chunk_metrics.csv - Performance metrics per chunk")  
    print(f"   📄 overall_metrics_per_ticker.csv - Overall performance per ticker")

    return predictions_df, list(overall_metrics.values())

def create_unified_ticker_scalers(df, feature_columns, scaler_file_path="../LSTM_model/ticker_scalers.pkl"):
    """
    สร้าง ticker scalers ตามแนวทางของโค้ดเทรน + การจัดการข้อมูลที่ดีขึ้น
    """
    print("🔧 Creating unified per-ticker scalers...")
    
    # ======== STEP 1: Data Cleaning (จากระบบปัจจุบัน) ========
    df_clean = clean_data_for_unified_scaling(df, feature_columns)
    
    # ======== STEP 2: Create Per-Ticker Scalers (จากโค้ดเทรน) ========
    ticker_scalers = {}
    unique_tickers = df_clean['StockSymbol'].unique()
    
    # สร้าง mapping ระหว่าง Ticker_ID กับ StockSymbol
    ticker_id_to_name = {}
    name_to_ticker_id = {}
    
    print("📋 Creating ticker mappings...")
    for ticker_name in unique_tickers:
        ticker_rows = df_clean[df_clean['StockSymbol'] == ticker_name]
        if len(ticker_rows) > 0:
            ticker_id = ticker_rows['Ticker_ID'].iloc[0]
            ticker_id_to_name[ticker_id] = ticker_name
            name_to_ticker_id[ticker_name] = ticker_id
            print(f"   Mapping: Ticker_ID {ticker_id} = {ticker_name}")
    
    # ตรวจสอบการโหลด pre-trained scalers
    pre_trained_scalers = {}
    try:
        if os.path.exists(scaler_file_path):
            pre_trained_scalers = joblib.load(scaler_file_path)
            print(f"✅ Loaded pre-trained scalers for {len(pre_trained_scalers)} tickers")
    except Exception as e:
        print(f"⚠️ Could not load pre-trained scalers: {e}")
    
    # สร้าง scalers สำหรับแต่ละ ticker
    for ticker_name in unique_tickers:
        ticker_data = df_clean[df_clean['StockSymbol'] == ticker_name].copy()
        
        if len(ticker_data) < 30:
            print(f"   ⚠️ {ticker_name}: Not enough data ({len(ticker_data)} days), skipping...")
            continue
        
        ticker_id = name_to_ticker_id[ticker_name]
        
        # ตรวจสอบ pre-trained scaler
        if ticker_id in pre_trained_scalers:
            scaler_info = pre_trained_scalers[ticker_id]
            
            # ตรวจสอบ structure ของ scaler
            required_keys = ['feature_scaler', 'price_scaler']
            if all(key in scaler_info for key in required_keys):
                try:
                    # ทดสอบ scaler
                    test_features = ticker_data[feature_columns].iloc[:5]
                    test_price = ticker_data[['Close']].iloc[:5]
                    
                    _ = scaler_info['feature_scaler'].transform(test_features.fillna(0))
                    _ = scaler_info['price_scaler'].transform(test_price)
                    
                    # เพิ่ม metadata ที่จำเป็น
                    scaler_info.update({
                        'ticker_symbol': ticker_name,
                        'ticker': ticker_name,  # สำหรับ compatibility
                        'data_points': len(ticker_data)
                    })
                    
                    ticker_scalers[ticker_id] = scaler_info
                    print(f"   ✅ {ticker_name} (ID: {ticker_id}): Using pre-trained scaler")
                    continue
                    
                except Exception as e:
                    print(f"   ⚠️ {ticker_name}: Pre-trained scaler failed ({e}), creating new one")
        
        # สร้าง scaler ใหม่
        try:
            print(f"   🔧 {ticker_name}: Creating new scaler...")
            
            # เตรียม feature data
            features = ticker_data[feature_columns].copy()
            
            # จัดการ inf และ NaN ตามแนวทางโค้ดเทรน
            features = handle_infinite_values(features)
            features = features.fillna(features.mean()).fillna(0)
            
            # เตรียม price data
            price_data = ticker_data[['Close']].copy()
            price_data = handle_infinite_values(price_data)
            price_data = price_data.fillna(price_data.mean())
            
            # สร้าง scalers
            feature_scaler = RobustScaler()
            price_scaler = RobustScaler()
            
            feature_scaler.fit(features)
            price_scaler.fit(price_data)
            
            # บันทึก scaler พร้อม metadata (ตามโครงสร้างโค้ดเทรน)
            ticker_scalers[ticker_id] = {
                'feature_scaler': feature_scaler,
                'price_scaler': price_scaler,
                'ticker': ticker_name,  # สำหรับ compatibility กับโค้ดเทรน
                'ticker_symbol': ticker_name,  # สำหรับระบบปัจจุบัน
                'data_points': len(ticker_data)
            }
            
            print(f"   ✅ {ticker_name} (ID: {ticker_id}): Created new scaler with {len(ticker_data)} data points")
            
        except Exception as e:
            print(f"   ❌ {ticker_name}: Error creating scaler - {e}")
            continue
    
    # บันทึก scalers
    try:
        os.makedirs(os.path.dirname(scaler_file_path), exist_ok=True)
        joblib.dump(ticker_scalers, scaler_file_path)
        print(f"💾 Saved unified scalers to {scaler_file_path}")
        
        # แสดงสรุป
        print(f"\n📊 Unified Ticker Scalers Summary:")
        for t_id, scaler_info in ticker_scalers.items():
            ticker_name = scaler_info.get('ticker', 'Unknown')
            data_points = scaler_info.get('data_points', 'Unknown')
            print(f"   Ticker_ID {t_id}: {ticker_name} ({data_points} data points)")
            
    except Exception as e:
        print(f"❌ Error saving scalers: {e}")
    
    print(f"✅ Created unified scalers for {len(ticker_scalers)} tickers")
    return ticker_scalers

def clean_data_for_unified_scaling(df, feature_columns):
    """ทำความสะอาดข้อมูลก่อนสร้าง unified scalers"""
    print("🧹 Cleaning data for unified scaling...")
    
    df_clean = df.copy()
    
    # Map column names จาก database format เป็น training format
    column_mapping = {
        'Change_Percent': 'Change (%)',
        'P_BV_Ratio': 'P/BV Ratio',
        'TotalRevenue': 'Total Revenue',
        'QoQGrowth': 'QoQ Growth (%)',
        'EPS': 'Earnings Per Share (EPS)',
        'ROE': 'ROE (%)',
        'NetProfitMargin': 'Net Profit Margin (%)',
        'DebtToEquity': 'Debt to Equity',
        'PERatio': 'P/E Ratio',
        'Dividend_Yield': 'Dividend Yield (%)',
    }
    
    # Rename columns ถ้าจำเป็น
    for old_name, new_name in column_mapping.items():
        if old_name in df_clean.columns and new_name not in df_clean.columns:
            df_clean[new_name] = df_clean[old_name]
            print(f"   🔄 Mapped {old_name} → {new_name}")
    
    # ทำความสะอาดข้อมูลตาม feature columns
    for col in feature_columns:
        if col in df_clean.columns:
            try:
                # แปลงเป็น numeric
                if not pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # จัดการ infinite values (ตามแนวทางโค้ดเทรน)
                df_clean[col] = handle_infinite_values_column(df_clean[col])
                
                print(f"   ✅ Cleaned {col}: range {df_clean[col].min():.3f} - {df_clean[col].max():.3f}")
                
            except Exception as e:
                print(f"   ❌ Error cleaning {col}: {e}")
                df_clean[col] = 0.0
    
    return df_clean

def handle_infinite_values(data):
    """จัดการ infinite values ตามแนวทางโค้ดเทรน"""
    data_clean = data.copy()
    
    for col in data_clean.columns:
        col_data = data_clean[col]
        
        # แทนที่ +inf ด้วยค่าสูงสุดที่ไม่ใช่ inf
        pos_inf_mask = col_data == np.inf
        if pos_inf_mask.any():
            max_val = col_data[col_data != np.inf].max()
            if pd.notna(max_val):
                data_clean.loc[pos_inf_mask, col] = max_val
            else:
                data_clean.loc[pos_inf_mask, col] = 0
        
        # แทนที่ -inf ด้วยค่าต่ำสุดที่ไม่ใช่ -inf
        neg_inf_mask = col_data == -np.inf
        if neg_inf_mask.any():
            min_val = col_data[col_data != -np.inf].min()
            if pd.notna(min_val):
                data_clean.loc[neg_inf_mask, col] = min_val
            else:
                data_clean.loc[neg_inf_mask, col] = 0
    
    return data_clean

def handle_infinite_values_column(series):
    """จัดการ infinite values สำหรับ column เดียว"""
    series_clean = series.copy()
    
    # แทนที่ +inf
    pos_inf_mask = series_clean == np.inf
    if pos_inf_mask.any():
        max_val = series_clean[series_clean != np.inf].max()
        if pd.notna(max_val):
            series_clean[pos_inf_mask] = max_val
        else:
            series_clean[pos_inf_mask] = 0
    
    # แทนที่ -inf
    neg_inf_mask = series_clean == -np.inf
    if neg_inf_mask.any():
        min_val = series_clean[series_clean != -np.inf].min()
        if pd.notna(min_val):
            series_clean[neg_inf_mask] = min_val
        else:
            series_clean[neg_inf_mask] = 0
    
    return series_clean

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
                MIN_WEIGHT = 0.1
                MAX_WEIGHT = 0.9
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


def load_training_scalers(scaler_path="../LSTM_model/ticker_scalers.pkl"):
    """
    โหลด ticker_scalers จากตอนเทรน
    ✅ ใช้ scalers เดียวกันกับตอนเทรน
    ✅ รองรับทั้ง LSTM และ GRU
    """
    print(f"🔧 กำลังโหลด training scalers จาก {scaler_path}...")
    
    if not os.path.exists(scaler_path):
        print(f"❌ ไม่พบไฟล์ {scaler_path}")
        return None, False
    
    try:
        ticker_scalers = joblib.load(scaler_path)
        print(f"✅ โหลด training scalers สำเร็จ!")
        print(f"   📊 จำนวน tickers: {len(ticker_scalers)}")
        
        # ตรวจสอบ structure
        sample_ticker_id = list(ticker_scalers.keys())[0]
        sample_scaler = ticker_scalers[sample_ticker_id]
        
        required_keys = ['feature_scaler', 'price_scaler']
        if all(key in sample_scaler for key in required_keys):
            print(f"   ✅ Structure ถูกต้อง: {list(sample_scaler.keys())}")
            
            # แสดงข้อมูล scalers
            print(f"   📋 Ticker scalers:")
            for i, (ticker_id, scaler_info) in enumerate(list(ticker_scalers.items())[:5]):
                ticker_name = scaler_info.get('ticker', f'ID_{ticker_id}')
                print(f"      {ticker_name} (ID: {ticker_id})")
            
            if len(ticker_scalers) > 5:
                print(f"      ... และอีก {len(ticker_scalers) - 5} tickers")
            
            return ticker_scalers, True
        else:
            print(f"   ❌ Structure ไม่ถูกต้อง: ขาด {[k for k in required_keys if k not in sample_scaler]}")
            return None, False
            
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการโหลด: {e}")
        return None, False

def validate_ticker_scalers(ticker_scalers, df, feature_columns):
    """
    ตรวจสอบว่า ticker_scalers ใช้งานได้กับข้อมูลปัจจุบัน
    """
    print(f"🔍 กำลังตรวจสอบความเข้ากันได้ของ scalers...")
    
    valid_scalers = {}
    validation_results = []
    
    available_tickers = df['StockSymbol'].unique()
    
    for ticker in available_tickers:
        ticker_data = df[df['StockSymbol'] == ticker]
        if len(ticker_data) == 0:
            continue
            
        # หา ticker_id
        ticker_id = ticker_data['Ticker_ID'].iloc[0]
        
        if ticker_id not in ticker_scalers:
            print(f"   ⚠️ {ticker}: ไม่พบ scaler สำหรับ Ticker_ID {ticker_id}")
            validation_results.append({'ticker': ticker, 'status': 'missing_scaler'})
            continue
        
        try:
            # ทดสอบ feature scaler
            test_features = ticker_data[feature_columns].iloc[:3].fillna(0)
            scaler_info = ticker_scalers[ticker_id]
            
            transformed_features = scaler_info['feature_scaler'].transform(test_features)
            
            # ทดสอบ price scaler
            test_prices = ticker_data[['Close']].iloc[:3]
            transformed_prices = scaler_info['price_scaler'].transform(test_prices)
            
            valid_scalers[ticker_id] = scaler_info
            validation_results.append({'ticker': ticker, 'status': 'valid'})
            print(f"   ✅ {ticker}: Scaler ใช้งานได้")
            
        except Exception as e:
            print(f"   ❌ {ticker}: Scaler ใช้งานไม่ได้ - {e}")
            validation_results.append({'ticker': ticker, 'status': 'invalid', 'error': str(e)})
    
    # สรุปผล
    valid_count = len([r for r in validation_results if r['status'] == 'valid'])
    total_count = len(validation_results)
    
    print(f"\n📊 ผลการตรวจสอบ:")
    print(f"   ✅ ใช้งานได้: {valid_count}/{total_count} tickers")
    print(f"   ❌ ใช้งานไม่ได้: {total_count - valid_count}/{total_count} tickers")
    
    return valid_scalers, validation_results

def save_predictions_simple(predictions_df):
    """
    บันทึกผลลัพธ์การพยากรณ์แบบเรียบง่าย
    เก็บ: วันที่, หุ้น, ราคาทำนาย (LSTM, GRU, Ensemble), ทิศทางทำนาย (LSTM, GRU, Ensemble)
    """
    if predictions_df.empty:
        print("❌ ไม่มีข้อมูลพยากรณ์ที่จะบันทึก")
        return False

    try:
        DB_CONNECTION = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
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
                            'lstm_price': float(row.get('LSTM_Price', row.get('Predicted_Price', 0))),
                            'lstm_trend': int(row.get('LSTM_Direction', row.get('Predicted_Direction', 0))),
                            'gru_price': float(row.get('GRU_Price', row.get('Predicted_Price', 0))),
                            'gru_trend': int(row.get('GRU_Direction', row.get('Predicted_Direction', 0))),
                            'ensemble_price': float(row.get('Ensemble_Price', row.get('Predicted_Price', 0))),
                            'ensemble_trend': int(row.get('Ensemble_Direction', row.get('Predicted_Direction', 0))),
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
                            'lstm_price': float(row.get('LSTM_Price', row.get('Predicted_Price', 0))),
                            'lstm_trend': int(row.get('LSTM_Direction', row.get('Predicted_Direction', 0))),
                            'gru_price': float(row.get('GRU_Price', row.get('Predicted_Price', 0))),
                            'gru_trend': int(row.get('GRU_Direction', row.get('Predicted_Direction', 0))),
                            'ensemble_price': float(row.get('Ensemble_Price', row.get('Predicted_Price', 0))),
                            'ensemble_trend': int(row.get('Ensemble_Direction', row.get('Predicted_Direction', 0)))
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

# ======================== WalkForwardMiniRetrainManager Class ========================
class WalkForwardMiniRetrainManager:
    """
    Walk-Forward Validation + Retrain System พร้อม XGBoost Ensemble
    - แบ่งข้อมูลเป็น chunks
    - retrain ทุก N วัน
    - Continuous learning แบบ incremental
    - เสถียรและมีประสิทธิภาพ
    """
    def __init__(self, 
                 lstm_model_path="../LSTM_model/best_v6_plus_minimal_tuning_v2_final_model.keras",
                 gru_model_path="../GRU_Model/best_v6_plus_minimal_tuning_v2_final_model.keras",
                 retrain_frequency=5,  
                 chunk_size=200,       
                 seq_length=10):
        
        self.lstm_model_path = lstm_model_path
        self.gru_model_path = gru_model_path
        self.retrain_frequency = retrain_frequency
        self.chunk_size = chunk_size
        self.seq_length = seq_length
        
        # โมเดล
        self.lstm_model = None
        self.gru_model = None
        
        # Performance tracking
        self.all_predictions = []
        self.chunk_metrics = []
        
    def load_models_for_prediction(self, model_path=None, compile_model=False):
        """โหลดโมเดลสำหรับการทำนาย สามารถโหลดโมเดลเดี่ยวหรือทั้ง LSTM และ GRU"""
        custom_objects = {
            "quantile_loss": quantile_loss,
            "focal_weighted_binary_crossentropy": focal_weighted_binary_crossentropy
        }
        try:
            if model_path:  # กรณีโหลดโมเดลเดี่ยว
                print(f"🔄 กำลังโหลดโมเดลจาก {model_path}...")
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects=custom_objects,
                    safe_mode=False,
                    compile=False  # โหลดโดยไม่ compile ก่อน
                )
                
                # Compile โมเดลถ้าจำเป็นสำหรับ mini-retrain
                if compile_model:
                    print(f"🔧 Compiling model for training...")
                    
                    # สร้าง loss functions
                    price_loss = quantile_loss
                    direction_loss = focal_weighted_binary_crossentropy(class_weights_dict)
                    
                    model.compile(
                        optimizer=Adam(learning_rate=0.0001),
                        loss={
                            'price_output': price_loss,
                            'direction_output': direction_loss
                        },
                        metrics={
                            'price_output': ['mae'],
                            'direction_output': ['accuracy']
                        }
                    )
                    print(f"✅ Model compiled successfully")
                
                print("✅ โหลดโมเดลเดี่ยวสำเร็จ")
                return model
            else:  # กรณีโหลดทั้ง LSTM และ GRU
                print("🔄 กำลังโหลดโมเดลสำหรับการทำนาย...")
                
                # โหลด LSTM
                self.lstm_model = tf.keras.models.load_model(
                    self.lstm_model_path,
                    custom_objects=custom_objects,
                    safe_mode=False,
                    compile=False
                )
                
                # โหลด GRU
                self.gru_model = tf.keras.models.load_model(
                    self.gru_model_path,
                    custom_objects=custom_objects,
                    safe_mode=False,
                    compile=False
                )
                
                # Compile โมเดลถ้าจำเป็นสำหรับ mini-retrain
                if compile_model:
                    print(f"🔧 Compiling both models for training...")
                    
                    # สร้าง loss functions
                    price_loss = quantile_loss
                    direction_loss = focal_weighted_binary_crossentropy(class_weights_dict)
                    
                    compile_config = {
                        'optimizer': Adam(learning_rate=0.0001),
                        'loss': {
                            'price_output': price_loss,
                            'direction_output': direction_loss
                        },
                        'metrics': {
                            'price_output': ['mae'],
                            'direction_output': ['accuracy']
                        }
                    }
                    
                    self.lstm_model.compile(**compile_config)
                    self.gru_model.compile(**compile_config)
                    print(f"✅ Both models compiled successfully")
                
                print("✅ โหลดโมเดลสำหรับการทำนายสำเร็จ")
                return True
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
            import traceback
            traceback.print_exc()
            return None if model_path else False

# Fix 2: Enhanced data cleaning function
def clean_data_for_scalers_enhanced(df, feature_columns):
    """ทำความสะอาดข้อมูลก่อนสร้าง scalers - Enhanced version"""
    print("🧹 กำลังทำความสะอาดข้อมูลสำหรับ scalers (Enhanced)...")
    
    df_clean = df.copy()
    
    # Enhanced data cleaning for each column
    for col in feature_columns:
        if col in df_clean.columns:
            try:
                print(f"   🔧 Processing {col}...")
                
                # Step 1: Handle object/string types
                if df_clean[col].dtype == 'object':
                    print(f"      📝 Converting {col} from object type")
                    
                    # Check for mixed numeric/string values
                    sample_values = df_clean[col].dropna().astype(str).head(10)
                    print(f"      Sample values: {sample_values.tolist()}")
                    
                    # Try different conversion strategies
                    def smart_numeric_conversion(series):
                        """Smart conversion that handles various string formats"""
                        converted_series = series.copy()
                        
                        # Strategy 1: Direct numeric conversion
                        numeric_converted = pd.to_numeric(converted_series, errors='coerce')
                        
                        # Strategy 2: Handle concatenated numbers (e.g., "49.0349.03")
                        if numeric_converted.isna().sum() > len(converted_series) * 0.5:
                            print(f"        🔧 Trying advanced string parsing...")
                            def extract_first_number(x):
                                if pd.isna(x):
                                    return np.nan
                                x_str = str(x).strip()
                                
                                # Handle empty strings
                                if x_str == '' or x_str.lower() in ['nan', 'none', 'null']:
                                    return np.nan
                                
                                # Extract first valid number using regex
                                import re
                                match = re.search(r'^-?\d*\.?\d+', x_str)
                                if match:
                                    try:
                                        return float(match.group())
                                    except:
                                        return np.nan
                                return np.nan
                            
                            converted_series = converted_series.apply(extract_first_number)
                        else:
                            converted_series = numeric_converted
                        
                        return converted_series
                    
                    df_clean[col] = smart_numeric_conversion(df_clean[col])
                
                # Step 2: Handle infinite values
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    inf_count = np.isinf(df_clean[col]).sum()
                    if inf_count > 0:
                        print(f"      ♾️ Replacing {inf_count} infinite values in {col}")
                        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                
                # Step 3: Handle NaN values intelligently
                nan_count = df_clean[col].isna().sum()
                if nan_count > 0:
                    print(f"      🔧 Handling {nan_count} NaN values in {col}")
                    
                    if col in ['Volume']:
                        # For Volume, use median
                        fill_value = df_clean[col].median()
                        if pd.isna(fill_value):
                            fill_value = 0
                    elif col in ['Close', 'Open', 'High', 'Low']:
                        # For price columns, use forward fill then backward fill
                        df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
                        fill_value = df_clean[col].mean()
                        if pd.isna(fill_value):
                            fill_value = 1.0  # Default price
                    else:
                        # For other columns, use mean or 0
                        fill_value = df_clean[col].mean()
                        if pd.isna(fill_value):
                            fill_value = 0.0
                    
                    df_clean[col] = df_clean[col].fillna(fill_value)
                
                # Step 4: Final data type validation
                if not pd.api.types.is_numeric_dtype(df_clean[col]):
                    print(f"      ❌ {col} still not numeric, forcing conversion")
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype('float64')
                else:
                    df_clean[col] = df_clean[col].astype('float64')
                
                # Step 5: Sanity check for extreme values
                if col in ['Close', 'Open', 'High', 'Low'] and (df_clean[col] <= 0).any():
                    positive_mean = df_clean[col][df_clean[col] > 0].mean()
                    if not pd.isna(positive_mean):
                        df_clean.loc[df_clean[col] <= 0, col] = positive_mean
                        print(f"      🔧 Replaced {(df_clean[col] <= 0).sum()} non-positive values in {col}")
                
                print(f"   ✅ Cleaned {col}: dtype={df_clean[col].dtype}, "
                      f"range=[{df_clean[col].min():.3f}, {df_clean[col].max():.3f}], "
                      f"NaN={df_clean[col].isna().sum()}")
                
            except Exception as e:
                print(f"   ❌ Failed to clean {col}: {e}")
                # Use safe default values
                if col in ['Close', 'Open', 'High', 'Low']:
                    df_clean[col] = 1.0  # Default price
                elif col in ['Volume']:
                    df_clean[col] = 1000.0  # Default volume
                else:
                    df_clean[col] = 0.0  # Default for indicators
                
                df_clean[col] = df_clean[col].astype('float64')
                print(f"   🔧 Used default value for {col}")
    
    # Final validation
    print("🔍 Final data validation...")
    for col in feature_columns:
        if col in df_clean.columns:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                print(f"❌ Column {col} is still not numeric: {df_clean[col].dtype}")
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype('float64')
            
            if df_clean[col].isna().any():
                print(f"❌ Column {col} still has NaN values")
                df_clean[col] = df_clean[col].fillna(0)
            
            if np.isinf(df_clean[col]).any():
                print(f"❌ Column {col} still has infinite values")
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], 0)
    
    print(f"✅ Enhanced data cleaning completed. Shape: {df_clean.shape}")
    return df_clean

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup logging
import joblib
from sklearn.preprocessing import StandardScaler

class XGBoostEnsembleMetaLearner:
    """
    XGBoost Ensemble Meta Learner - ใช้ XGBoost รวม LSTM และ GRU predictions
    
    Features:
    - ใช้ XGBoost แทน rule-based weighting
    - Direction accuracy 72%+ 
    - Model consistency 93%+
    - Automatic feature engineering
    """
    
    def __init__(self, model_path='../Ensemble_Model/fixed_unified_trading_model.pkl'):
        self.model_path = model_path
        self.trading_system = None
        self.performance_history = {}
        
        # 📊 Historical Performance Data จาก metrics ที่ให้มา
        self.stock_performance = {
            # Stock: {metric: (LSTM_value, GRU_value)}
            'AAPL': {
                'MAE': (7.1761, 4.7221),
                'R2': (0.7203, 0.8783),
                'Direction_Acc': (0.6455, 0.6499)
            },
            'ADVANC': {
                'MAE': (6.5163, 5.6464),
                'R2': (0.8814, 0.9046),
                'Direction_Acc': (0.6888, 0.7056)
            },
            'AMD': {
                'MAE': (4.3655, 3.6774),
                'R2': (0.9354, 0.9613),
                'Direction_Acc': (0.7695, 0.7533)
            },
            'AMZN': {
                'MAE': (6.1828, 7.5087),
                'R2': (0.8641, 0.8300),
                'Direction_Acc': (0.6686, 0.6472)
            },
            'AVGO': {
                'MAE': (14.2846, 13.0835),
                'R2': (0.7351, 0.8006),
                'Direction_Acc': (0.6542, 0.6366)
            },
            'DIF': {
                'MAE': (0.2062, 0.1458),
                'R2': (0.6259, 0.7279),
                'Direction_Acc': (0.7147, 0.6844)
            },
            'DITTO': {
                'MAE': (1.3103, 0.6702),
                'R2': (0.7695, 0.9302),
                'Direction_Acc': (0.6945, 0.6844)
            },
            'GOOGL': {
                'MAE': (5.1876, 5.3210),
                'R2': (0.7843, 0.7746),
                'Direction_Acc': (0.7320, 0.7215)
            },
            'HUMAN': {
                'MAE': (0.2718, 0.1781),
                'R2': (0.9480, 0.9737),
                'Direction_Acc': (0.7630, 0.7686)
            },
            'INET': {
                'MAE': (0.1599, 0.1269),
                'R2': (0.8131, 0.8977),
                'Direction_Acc': (0.7378, 0.7162)
            },
            'INSET': {
                'MAE': (0.0824, 0.0539),
                'R2': (0.9629, 0.9832),
                'Direction_Acc': (0.7262, 0.7427)
            },
            'JAS': {
                'MAE': (0.0678, 0.0705),
                'R2': (0.9710, 0.9726),
                'Direction_Acc': (0.7378, 0.7507)
            },
            'JMART': {
                'MAE': (0.8154, 0.5712),
                'R2': (0.8850, 0.9508),
                'Direction_Acc': (0.7378, 0.7666)
            },
            'META': {
                'MAE': (22.3500, 28.5451),
                'R2': (0.8665, 0.7128),
                'Direction_Acc': (0.6340, 0.6658)
            },
            'MSFT': {
                'MAE': (16.2797, 8.9404),
                'R2': (0.6759, 0.9007),
                'Direction_Acc': (0.6282, 0.6101)
            },
            'NVDA': {
                'MAE': (12.2969, 9.9044),
                'R2': (0.1414, 0.4617),
                'Direction_Acc': (0.5303, 0.5199)
            },
            'TRUE': {
                'MAE': (0.3843, 0.2568),
                'R2': (0.7830, 0.8893),
                'Direction_Acc': (0.7176, 0.6844)
            },
            'TSLA': {
                'MAE': (19.3031, 7.7488),
                'R2': (0.8700, 0.9774),
                'Direction_Acc': (0.6916, 0.6790)
            },
        # XGBoost model will handle all the ensemble logic
    
    def load_model(self):
        """Load XGBoost ensemble model"""
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Ensemble_Model'))
            from XGBoost import FixedUnifiedTradingSystem
            
            if os.path.exists(self.model_path):
                self.trading_system = FixedUnifiedTradingSystem()
                self.trading_system.load_model(self.model_path)
                print(f"✅ Loaded XGBoost ensemble model from {self.model_path}")
            else:
                print(f"⚠️ XGBoost model not found at {self.model_path}, will use fallback method")
                self.trading_system = None
        except Exception as e:
            print(f"⚠️ Failed to load XGBoost model: {e}, using fallback method")
            self.trading_system = None
            
        print("🎆 XGBoost Ensemble Meta Learner initialized")
        print(f"🎨 Model path: {self.model_path}")
        print(f"🎯 Expected: 72% direction accuracy, 93% consistency")
        print(f"🔥 **XGBoost Ensemble Ready** 🔥")
    
    def predict_meta(self, df):
        
        perf = self.stock_performance[ticker]
        lstm_mae, gru_mae = perf['MAE']
        lstm_r2, gru_r2 = perf['R2']
        lstm_dir, gru_dir = perf['Direction_Acc']
        
        # คำนวณ performance advantage ของ GRU
        mae_advantage = (lstm_mae - gru_mae) / lstm_mae if lstm_mae > 0 else 0  # GRU better if positive
        r2_advantage = (gru_r2 - lstm_r2) / lstm_r2 if lstm_r2 > 0 else 0      # GRU better if positive
        dir_advantage = (gru_dir - lstm_dir) / lstm_dir if lstm_dir > 0 else 0  # GRU better if positive
        
        # รวม advantages (weighted)
        total_advantage = (0.5 * mae_advantage + 0.3 * r2_advantage + 0.2 * dir_advantage)
        
        # แปลงเป็น bonus multiplier (1.0 - 2.0) - increased max bonus
        adaptive_bonus = 1.0 + max(0, min(1.0, total_advantage * 2))  # Doubled the multiplier
        
        logger.debug(f"{ticker}: Adaptive GRU bonus = {adaptive_bonus:.3f} "
                    f"[MAE:{mae_advantage:.2f}, R²:{r2_advantage:.2f}, Dir:{dir_advantage:.2f}]")
        
        return adaptive_bonus
    
    def calculate_stock_specific_weights(self, ticker):
        """
        🎯 คำนวณน้ำหนักเฉพาะหุ้นจาก historical performance พร้อม GRU Bias
        """
        
        if ticker not in self.stock_performance:
            logger.warning(f"No historical performance data for {ticker}, using GRU-biased equal weights")
            # แม้ไม่มีข้อมูล ก็ให้ GRU น้ำหนักมากกว่า
            biased_gru_weight = 0.5 * self.gru_bias_factor
            total = 0.5 + biased_gru_weight
            return 0.5 / total, biased_gru_weight / total
        
        perf = self.stock_performance[ticker]
        
        # Extract metrics (LSTM, GRU)
        lstm_mae, gru_mae = perf['MAE']
        lstm_r2, gru_r2 = perf['R2']
        lstm_dir, gru_dir = perf['Direction_Acc']
        
        # 1. MAE Score (inverse - lower is better)
        mae_lstm_score = 1 / (lstm_mae + 1e-8)
        mae_gru_score = 1 / (gru_mae + 1e-8)
        mae_total = mae_lstm_score + mae_gru_score
        mae_lstm_weight = mae_lstm_score / mae_total
        mae_gru_weight = mae_gru_score / mae_total
        
        # 2. R² Score (direct - higher is better)
        r2_total = lstm_r2 + gru_r2
        r2_lstm_weight = lstm_r2 / r2_total if r2_total > 0 else 0.5
        r2_gru_weight = gru_r2 / r2_total if r2_total > 0 else 0.5
        
        # 3. Direction Accuracy (direct - higher is better)
        dir_total = lstm_dir + gru_dir
        dir_lstm_weight = lstm_dir / dir_total if dir_total > 0 else 0.5
        dir_gru_weight = gru_dir / dir_total if dir_total > 0 else 0.5
        
        # 🏆 Combined weighted score - ปรับให้เน้น MAE มากขึ้น (GRU มักดีกว่าใน MAE)
        # MAE มีน้ำหนัก 50%, R² มีน้ำหนัก 35%, Direction Accuracy มีน้ำหนัก 15%
        lstm_base_weight = (0.5 * mae_lstm_weight + 
                           0.35 * r2_lstm_weight + 
                           0.15 * dir_lstm_weight)
        
        gru_base_weight = (0.5 * mae_gru_weight + 
                          0.35 * r2_gru_weight + 
                          0.15 * dir_gru_weight)
        
        # 🎯 Apply GRU Bias Factor
        adaptive_bonus = self.calculate_adaptive_gru_bonus(ticker)
        total_gru_bias = self.gru_bias_factor * adaptive_bonus
        
        gru_biased_weight = gru_base_weight * total_gru_bias
        
        # Normalize to ensure sum = 1
        total_weight = lstm_base_weight + gru_biased_weight
        if total_weight > 0:
            lstm_final_weight = lstm_base_weight / total_weight
            gru_final_weight = gru_biased_weight / total_weight
        else:
            # Fallback with GRU bias
            lstm_final_weight = 0.4
            gru_final_weight = 0.6
        
        logger.debug(f"{ticker}: LSTM={lstm_final_weight:.3f}, GRU={gru_final_weight:.3f} "
                    f"[Base: {lstm_base_weight:.2f}/{gru_base_weight:.2f}, "
                    f"GRU_Bias: {total_gru_bias:.2f}]")
        
        return lstm_final_weight, gru_final_weight
    
    def calculate_dynamic_weights(self, ticker, recent_performance):
        """
        🎯 Dynamic Weighting Algorithm - ผสมระหว่าง historical และ recent performance พร้อม GRU Bias
        """
        
        # Get stock-specific historical weights (already GRU-biased)
        hist_lstm_weight, hist_gru_weight = self.calculate_stock_specific_weights(ticker)
        
        # If insufficient recent data, use historical weights
        if len(recent_performance) < 3:
            logger.debug(f"{ticker}: Using GRU-biased historical weights (insufficient recent data)")
            return hist_lstm_weight, hist_gru_weight
        
        try:
            # Calculate recent performance weights
            actual_prices = recent_performance['Actual_Price'].values
            lstm_predictions = recent_performance['Predicted_Price_LSTM'].values
            gru_predictions = recent_performance['Predicted_Price_GRU'].values
            
            # Recent MAE calculation
            lstm_recent_mae = mean_absolute_error(actual_prices, lstm_predictions)
            gru_recent_mae = mean_absolute_error(actual_prices, gru_predictions)
            
            # Recent weights (inverse MAE) with GRU bias
            lstm_recent_inv = 1 / (lstm_recent_mae + 1e-8)
            gru_recent_inv = (1 / (gru_recent_mae + 1e-8)) * self.gru_bias_factor  # 🎯 Apply GRU bias to recent performance too
            
            total_recent_inv = lstm_recent_inv + gru_recent_inv
            
            recent_lstm_weight = lstm_recent_inv / total_recent_inv
            recent_gru_weight = gru_recent_inv / total_recent_inv
            
            # 🏆 Adaptive blending: ใช้ historical 60% + recent 40% (เพิ่ม recent weight เพื่อให้ GRU bias มีผลมากขึ้น)
            alpha = 0.4  # recent performance weight (increased from 0.3)
            final_lstm_weight = (1 - alpha) * hist_lstm_weight + alpha * recent_lstm_weight
            final_gru_weight = (1 - alpha) * hist_gru_weight + alpha * recent_gru_weight
            
            logger.debug(f"{ticker}: GRU-Biased weights - LSTM={final_lstm_weight:.3f}, GRU={final_gru_weight:.3f} "
                        f"[Hist: {hist_lstm_weight:.2f}/{hist_gru_weight:.2f}, "
                        f"Recent: {recent_lstm_weight:.2f}/{recent_gru_weight:.2f}]")
            
            return final_lstm_weight, final_gru_weight
            
        except Exception as e:
            logger.warning(f"Error calculating dynamic weights for {ticker}: {e}")
            return hist_lstm_weight, hist_gru_weight
    
    def get_performance_summary(self, ticker):
        """📊 Get performance summary for a stock with GRU bias indication"""
        if ticker not in self.stock_performance:
            return "No historical data"
        
        perf = self.stock_performance[ticker]
        lstm_mae, gru_mae = perf['MAE']
        lstm_r2, gru_r2 = perf['R2']
        lstm_dir, gru_dir = perf['Direction_Acc']
        
        # Determine better model for each metric
        mae_winner = "LSTM" if lstm_mae < gru_mae else "GRU⭐"
        r2_winner = "LSTM" if lstm_r2 > gru_r2 else "GRU⭐"
        dir_winner = "LSTM" if lstm_dir > gru_dir else "GRU⭐"
        
        # Count GRU wins for bias indication
        gru_wins = sum([lstm_mae >= gru_mae, lstm_r2 <= gru_r2, lstm_dir <= gru_dir])
        bias_indicator = f"GRU_Advantage:{gru_wins}/3" if gru_wins >= 2 else f"Mixed_Performance:{gru_wins}/3"
        
        return f"MAE:{mae_winner}, R²:{r2_winner}, Dir:{dir_winner} [{bias_indicator}]"
    
    def prepare_data_for_model(self, df):
        """เตรียมข้อมูลสำหรับ ensemble"""
        
        prepared_df = df.copy()
        
        # Map column names
        column_mapping = {
            'StockSymbol': 'Ticker',
            'Close': 'Current_Price',
            'PredictionClose_LSTM': 'Predicted_Price_LSTM',
            'PredictionClose_GRU': 'Predicted_Price_GRU',
            'PredictionTrend_LSTM': 'Predicted_Dir_LSTM',
            'PredictionTrend_GRU': 'Predicted_Dir_GRU'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in prepared_df.columns:
                prepared_df[new_name] = prepared_df[old_name]
        
        # Add Date if missing
        if 'Date' not in prepared_df.columns:
            prepared_df['Date'] = datetime.now().strftime('%Y-%m-%d')
        prepared_df['Date'] = pd.to_datetime(prepared_df['Date'])
        
        # Filter valid data
        prediction_mask = (
            prepared_df['Predicted_Price_LSTM'].notna() & 
            prepared_df['Predicted_Price_GRU'].notna() &
            prepared_df['Predicted_Dir_LSTM'].notna() & 
            prepared_df['Predicted_Dir_GRU'].notna() &
            (prepared_df['Current_Price'] > 0)
        )
        
        prepared_df = prepared_df[prediction_mask].copy()
        
        if len(prepared_df) == 0:
            logger.error("No valid data for prediction")
            return None
        
        logger.info(f"✅ Data prepared: {len(prepared_df)} rows")
        return prepared_df
    
    def predict_single_stock(self, stock_data, ticker):
        """🎯 Enhanced Stock-Specific Ensemble Prediction with GRU Bias"""
        
        current_price = stock_data['Current_Price']
        lstm_price = stock_data['Predicted_Price_LSTM']
        gru_price = stock_data['Predicted_Price_GRU']
        lstm_dir = stock_data['Predicted_Dir_LSTM']
        gru_dir = stock_data['Predicted_Dir_GRU']
        
        # Get recent performance for dynamic weighting
        recent_performance = self.performance_history.get(ticker, pd.DataFrame())
        
        # Calculate stock-specific dynamic weights (with GRU bias)
        lstm_weight, gru_weight = self.calculate_dynamic_weights(ticker, recent_performance)
        
        # 🏆 GRU-Biased Weighted Prediction
        ensemble_price = lstm_weight * lstm_price + gru_weight * gru_price
        
        # Direction prediction (weighted voting) - GRU already has higher weight
        lstm_dir_weighted = lstm_weight * lstm_dir
        gru_dir_weighted = gru_weight * gru_dir
        ensemble_dir_prob = lstm_dir_weighted + gru_dir_weighted
        ensemble_direction = 1 if ensemble_dir_prob >= 0.5 else 0
        
        # Price change analysis
        price_change_pct = ((ensemble_price - current_price) / current_price) * 100
        price_implied_direction = 1 if price_change_pct > 0.5 else 0
        
        # 🧠 Smart Consistency Check - consider GRU dominance and magnitude
        basic_consistency = (ensemble_direction == price_implied_direction)
        
        if self.smart_consistency_check:
            # If GRU dominates and price change is significant, trust GRU more
            if gru_weight > 0.6 and abs(price_change_pct) > 2.0:
                smart_consistency = True  # Trust GRU-dominant prediction
                consistency_note = "GRU_DOMINANT_TRUSTED"
            # If models agree on direction probability trend
            elif (lstm_dir > 0.5 and gru_dir > 0.5) or (lstm_dir < 0.5 and gru_dir < 0.5):
                smart_consistency = True
                consistency_note = "DIRECTIONAL_AGREEMENT"
            else:
                smart_consistency = basic_consistency
                consistency_note = "BASIC_CHECK"
        else:
            smart_consistency = basic_consistency
            consistency_note = "BASIC_CHECK"
        
        is_consistent = smart_consistency
        
        # Enhanced confidence calculation with ultra-aggressive GRU bias boost
        direction_confidence = abs(ensemble_dir_prob - 0.5) * 2.5  # Increased multiplier from 2.2
        price_confidence = min(abs(price_change_pct) / 5, 1.0)  # Reduced divisor for even higher confidence
        
        # Add historical performance boost to confidence (with ultra-aggressive GRU weighting)
        if ticker in self.stock_performance:
            perf = self.stock_performance[ticker]
            # Give ultra-dominant weight to GRU performance in confidence calculation
            weighted_r2 = (0.05 * perf['R2'][0] + 0.95 * perf['R2'][1])  # 95% GRU
            weighted_dir_acc = (0.05 * perf['Direction_Acc'][0] + 0.95 * perf['Direction_Acc'][1])  # 95% GRU
            historical_confidence = (weighted_r2 + weighted_dir_acc) / 2
            
            # Blend current confidence with historical performance (more historical weight for stable base)
            overall_confidence = 0.4 * ((direction_confidence + price_confidence) / 2) + 0.6 * historical_confidence
        else:
            overall_confidence = (direction_confidence + price_confidence) / 2
        
        # Ultra-Aggressive GRU Dominance Bonus - massive boost when GRU dominates
        if gru_weight > lstm_weight:
            dominance_ratio = gru_weight / (lstm_weight + 0.01)  # Avoid division by zero
            gru_dominance_bonus = min(dominance_ratio * 0.35 * self.confidence_boost, 0.5)  # Up to 50% bonus
            overall_confidence = min(overall_confidence + gru_dominance_bonus, 0.95)
        
        # Performance-based confidence boost for high-performing stocks
        if ticker in self.stock_performance:
            perf = self.stock_performance[ticker]
            gru_r2, gru_dir = perf['R2'][1], perf['Direction_Acc'][1]
            if gru_r2 > 0.9 and gru_dir > 0.7:  # High-performance GRU
                performance_boost = 0.15
                overall_confidence = min(overall_confidence + performance_boost, 0.95)
        
        # Smart Consistency bonus/penalty
        if is_consistent:
            if consistency_note == "GRU_DOMINANT_TRUSTED":
                overall_confidence = min(overall_confidence * 1.35, 0.95)  # 35% bonus for trusted GRU
            elif consistency_note == "DIRECTIONAL_AGREEMENT":
                overall_confidence = min(overall_confidence * 1.3, 0.95)   # 30% bonus for directional agreement
            else:
                overall_confidence = min(overall_confidence * 1.25, 0.95)  # 25% bonus for normal consistency
        else:
            # Very reduced penalty when GRU is dominant (trust GRU heavily)
            if gru_weight > 0.7:
                overall_confidence = max(overall_confidence * 0.95, 0.2)  # Only 5% penalty for strong GRU
            elif gru_weight > 0.6:
                overall_confidence = max(overall_confidence * 0.9, 0.15)   # 10% penalty for moderate GRU
            else:
                overall_confidence = max(overall_confidence * 0.8, 0.1)    # 20% penalty for weak GRU
        
        # Enhanced minimum confidence floor based on GRU dominance and performance
        if gru_weight > 0.7:
            overall_confidence = max(overall_confidence, 0.35)  # Strong GRU minimum 35%
        elif gru_weight > 0.6:
            overall_confidence = max(overall_confidence, 0.3)   # Moderate GRU minimum 30%
        elif gru_weight > 0.5:
            overall_confidence = max(overall_confidence, 0.25)  # Weak GRU minimum 25%
        
        # Ultra-lenient risk assessment for GRU-biased predictions
        perf_summary = self.get_performance_summary(ticker)
        gru_dominance = gru_weight / (lstm_weight + gru_weight) if (lstm_weight + gru_weight) > 0 else 0.5
        
        # Ultra-lenient thresholds that favor GRU-dominant predictions
        if not is_consistent and overall_confidence < 0.2 and gru_weight < 0.5:  # Only penalize weak non-GRU cases
            risk_level = "🔴 HIGH_RISK"
            warning = f"INCONSISTENT_LOW_CONFIDENCE - {perf_summary} [GRU:{gru_dominance:.1%}]"
            action = "EXERCISE_EXTREME_CAUTION"
        elif abs(price_change_pct) > 30:  # Very high volatility threshold
            risk_level = "🟡 MEDIUM_RISK"
            warning = f"EXTREME_VOLATILITY_PREDICTION - {perf_summary} [GRU_Biased:{gru_dominance:.1%}]"
            action = "HIGH_RISK"
        elif overall_confidence >= 0.45:  # Significantly lowered threshold
            risk_level = "🟢 LOW_RISK"
            warning = f"HIGH_CONFIDENCE - {perf_summary} [GRU_Preferred:{gru_dominance:.1%}] [{consistency_note}]"
            action = "CONSIDER"
        elif overall_confidence >= 0.25:  # Very generous medium risk threshold
            risk_level = "🟡 MEDIUM_RISK"
            warning = f"MODERATE_CONFIDENCE - {perf_summary} [GRU_Weighted:{gru_dominance:.1%}] [{consistency_note}]"
            action = "CAUTION"
        else:
            risk_level = "🔴 HIGH_RISK"
            warning = f"LOW_CONFIDENCE - {perf_summary} [GRU_Bias:{gru_dominance:.1%}] [{consistency_note}]"
            action = "AVOID"
        
        return {
            'predicted_price': ensemble_price,
            'predicted_direction': ensemble_direction,
            'direction_probability': ensemble_dir_prob,
            'confidence': overall_confidence,
            'price_change_pct': price_change_pct,
            'is_consistent': is_consistent,
            'consistency_note': consistency_note,
            'lstm_weight': lstm_weight,
            'gru_weight': gru_weight,
            'gru_dominance': gru_dominance,
            'risk_level': risk_level,
            'warning': warning,
            'action': action,
            'performance_summary': perf_summary
        }
    
    def predict_meta(self, df):
        """Main Prediction Method using XGBoost Ensemble"""
        
        print("🚀 Starting XGBoost Ensemble Prediction...")
        
        if self.trading_system is None:
            print("⚠️ XGBoost model not available, using fallback method")
            return self._fallback_prediction(df)
        
        # Prepare data for XGBoost
        prepared_df = self.prepare_data_for_xgboost(df)
        if prepared_df is None or len(prepared_df) == 0:
            print("❌ No data to process")
            return df
        
        print(f"📊 Processing {len(prepared_df)} stocks with XGBoost")
        
        try:
            # Get XGBoost predictions
            xgb_results = self.trading_system.predict_signals(prepared_df)
            
            if len(xgb_results) == 0:
                print("❌ No XGBoost predictions generated")
                return df
                
            # Calculate summary statistics
            avg_confidence = xgb_results['Confidence'].mean()
            consistency_rate = (1 - xgb_results['Is_Inconsistent']).mean()
            direction_dist = xgb_results['Predicted_Direction'].value_counts()
            
            print(f"✅ Results: {len(xgb_results)} stocks processed")
            print(f"📊 Avg Confidence: {avg_confidence:.3f}")
            print(f"📊 Consistency Rate: {consistency_rate:.1%}")
            print(f"📊 Direction Distribution: UP={direction_dist.get(1, 0)}, DOWN={direction_dist.get(0, 0)}")
            
            # Map results back to dataframe
            ticker_col = 'StockSymbol' if 'StockSymbol' in df.columns else 'Ticker'
            
            for idx, result in xgb_results.iterrows():
                ticker = result['Ticker']
                mask = df[ticker_col] == ticker
                
                if mask.any():
                    df_idx = df[mask].index[0]
                    
                    # Core XGBoost predictions
                    df.loc[df_idx, 'XGB_Predicted_Direction'] = result['Predicted_Direction']
                    df.loc[df_idx, 'XGB_Predicted_Price'] = result['Predicted_Price']
                    df.loc[df_idx, 'XGB_Confidence'] = result['Confidence']
                    df.loc[df_idx, 'XGB_Predicted_Direction_Proba'] = result['Direction_Probability']
                    
                    # Additional info
                    df.loc[df_idx, 'Price_Change_Percent'] = result['Predicted_Return_Pct']
                    df.loc[df_idx, 'Is_Consistent'] = not result['Is_Inconsistent']
                    df.loc[df_idx, 'Model_Consistency'] = result['Model_Consistency']
                    df.loc[df_idx, 'Ensemble_Method'] = "XGBoost_Ensemble_v1.0"
                    
                    # Set reliability info based on confidence
                    if result['Confidence'] >= 0.8:
                        df.loc[df_idx, 'Risk_Level'] = 'LOW'
                        df.loc[df_idx, 'Suggested_Action'] = 'STRONG_BUY' if result['Predicted_Direction'] == 1 else 'STRONG_SELL'
                        df.loc[df_idx, 'Reliability_Warning'] = 'High confidence prediction'
                    elif result['Confidence'] >= 0.6:
                        df.loc[df_idx, 'Risk_Level'] = 'MEDIUM'
                        df.loc[df_idx, 'Suggested_Action'] = 'BUY' if result['Predicted_Direction'] == 1 else 'SELL'
                        df.loc[df_idx, 'Reliability_Warning'] = 'Moderate confidence'
                    else:
                        df.loc[df_idx, 'Risk_Level'] = 'HIGH'
                        df.loc[df_idx, 'Suggested_Action'] = 'HOLD'
                        df.loc[df_idx, 'Reliability_Warning'] = 'Low confidence - consider holding'
                    
                    df.loc[df_idx, 'Reliability_Score'] = result['Confidence']
            
            print("✅ XGBoost Ensemble prediction completed")
            return df
            
        except Exception as e:
            print(f"❌ XGBoost prediction failed: {e}")
            return self._fallback_prediction(df)
    
    def prepare_data_for_xgboost(self, df):
        """Prepare data in format expected by XGBoost model"""
        try:
            # Check if we have LSTM and GRU predictions
            required_cols = ['Predicted_Price_LSTM', 'Predicted_Price_GRU', 
                           'Predicted_Dir_LSTM', 'Predicted_Dir_GRU', 'Current_Price']
            
            ticker_col = 'StockSymbol' if 'StockSymbol' in df.columns else 'Ticker'
            
            # Use current price if available, otherwise use last close price
            if 'Current_Price' not in df.columns:
                if 'ClosePrice' in df.columns:
                    df = df.copy()
                    df['Current_Price'] = df['ClosePrice']
                elif 'Close' in df.columns:
                    df = df.copy()
                    df['Current_Price'] = df['Close']
                else:
                    print("❌ No price data available")
                    return None
            
            # Check for required prediction columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"❌ Missing required columns for XGBoost: {missing_cols}")
                return None
            
            # Prepare data in XGBoost format
            xgb_data = pd.DataFrame({
                'Ticker': df[ticker_col],
                'Date': df.get('Date', pd.Timestamp.now().strftime('%Y-%m-%d')),
                'Actual_Price': df['Current_Price'],  # XGBoost expects this as current price
                'Predicted_Price_LSTM': df['Predicted_Price_LSTM'],
                'Predicted_Price_GRU': df['Predicted_Price_GRU'],
                'Predicted_Dir_LSTM': df['Predicted_Dir_LSTM'],
                'Predicted_Dir_GRU': df['Predicted_Dir_GRU']
            })
            
            # Remove rows with missing data
            xgb_data = xgb_data.dropna()
            
            print(f"📊 Prepared {len(xgb_data)} stocks for XGBoost prediction")
            return xgb_data
            
        except Exception as e:
            print(f"❌ Error preparing data for XGBoost: {e}")
            return None
    
    def _fallback_prediction(self, df):
        """Fallback prediction method when XGBoost is not available"""
        print("🔄 Using simple ensemble fallback method")
        
        ticker_col = 'StockSymbol' if 'StockSymbol' in df.columns else 'Ticker'
        
        if 'Predicted_Price_LSTM' in df.columns and 'Predicted_Price_GRU' in df.columns:
            # Simple average ensemble
            df['XGB_Predicted_Price'] = (df['Predicted_Price_LSTM'] + df['Predicted_Price_GRU']) / 2
            df['XGB_Predicted_Direction'] = ((df.get('Predicted_Dir_LSTM', 0.5) + df.get('Predicted_Dir_GRU', 0.5)) / 2 > 0.5).astype(int)
            df['XGB_Confidence'] = 0.5  # Medium confidence for fallback
            df['XGB_Predicted_Direction_Proba'] = (df.get('Predicted_Dir_LSTM', 0.5) + df.get('Predicted_Dir_GRU', 0.5)) / 2
            df['Ensemble_Method'] = "Simple_Average_Fallback"
            df['Risk_Level'] = 'MEDIUM'
            df['Suggested_Action'] = 'HOLD'
            df['Reliability_Warning'] = 'Fallback method - XGBoost not available'
            df['Reliability_Score'] = 0.5
        
        return df
    
    def update_performance_history(self, ticker, actual_price, lstm_pred, gru_pred, date):
        """Update performance history for dynamic weighting"""
        
        if ticker not in self.performance_history:
            self.performance_history[ticker] = pd.DataFrame()
        
        new_record = pd.DataFrame({
            'Date': [date],
            'Actual_Price': [actual_price],
            'Predicted_Price_LSTM': [lstm_pred],
            'Predicted_Price_GRU': [gru_pred]
        })
        
        self.performance_history[ticker] = pd.concat([
            self.performance_history[ticker], 
            new_record
        ], ignore_index=True)
        
        # Keep only recent records
        max_history = self.window_size * 3
        if len(self.performance_history[ticker]) > max_history:
            self.performance_history[ticker] = self.performance_history[ticker].tail(max_history)
    
    def get_stock_recommendations(self):
        """📈 Get stock recommendations based on GRU-biased performance"""
        recommendations = {}
        
        # Since XGBoost handles ensemble automatically, provide general recommendations
        general_performance = {
            'direction_accuracy': 0.72,
            'consistency': 0.93,
            'confidence_threshold': 0.6
        }
        
        # Placeholder recommendations - in practice, this would be based on recent XGBoost results
        default_tickers = ['AAPL', 'ADVANC', 'AMD', 'AMZN', 'AVGO', 'DIF', 'DITTO', 'GOOGL', 
                          'HUMAN', 'INET', 'INSET', 'JAS', 'JMART', 'META', 'MSFT', 'NVDA', 'TRUE', 'TSLA', 'TSM']
        
        for ticker in default_tickers:
            recommendations[ticker] = {
                'score': general_performance['direction_accuracy'],
                'recommendation': "🟢 HIGH_CONFIDENCE_XGB_ENSEMBLE",
                'expected_accuracy': general_performance['direction_accuracy'],
                'expected_consistency': general_performance['consistency']
            }
        
        return recommendations
    
    # Compatibility methods
    def is_model_available(self):
        return self.trading_system is not None
    
    def get_model_status(self):
        if self.trading_system is not None:
            return "XGBoost_Ensemble_v1.0_READY"
        else:
            return "XGBoost_Ensemble_v1.0_FALLBACK"
    
    def should_retrain_meta(self):
        return False  # XGBoost model handles its own retraining
    
    def validate_predictions(self, df):
        return 'XGB_Predicted_Price' in df.columns and df['XGB_Predicted_Price'].notna().any()

# Backward compatibility - use XGBoost as default
DynamicEnsembleMetaLearner = XGBoostEnsembleMetaLearner
XGBoostMetaLearner = XGBoostEnsembleMetaLearner  
UpdatedXGBoostMetaLearner = XGBoostEnsembleMetaLearner
EnhancedDynamicEnsembleMetaLearner = XGBoostEnsembleMetaLearner
EnhancedGRUBiasedEnsembleMetaLearner = XGBoostEnsembleMetaLearner

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
meta_learner = UpdatedXGBoostMetaLearner()

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
            WHERE Stock.Market in ('America','Thailand')
            AND StockDetail.Date >= CURDATE() - INTERVAL 350 DAY
            ORDER BY StockDetail.StockSymbol, StockDetail.Date ASC;
        """

        df = pd.read_sql(query, engine)
        engine.dispose()
        
        print(f"📊 ข้อมูลดิบจาก DB:")
        print(f"   📅 วันที่: {df['Date'].min()} ถึง {df['Date'].max()}")
        print(f"   🏷️ TRUE data:")
        true_data = df[df['StockSymbol'] == 'TRUE']
        if not true_data.empty:
            print(f"      📅 TRUE วันที่: {true_data['Date'].min()} ถึง {true_data['Date'].max()}")
            print(f"      📋 จำนวน: {len(true_data)} วัน")
            print(f"      💰 วันที่ล่าสุด: {true_data.iloc[-1]['Date']} = {true_data.iloc[-1]['Close']}")
        

        if df.empty:
            print("❌ ไม่มีข้อมูลหุ้นสำหรับตลาดที่กำลังเปิดอยู่")
            return df

        # Data processing
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 🔧 Debug หลัง convert datetime
        print(f"\n📊 หลัง convert datetime:")
        true_data = df[df['StockSymbol'] == 'TRUE']
        if not true_data.empty:
            print(f"   🏷️ TRUE data: {true_data['Date'].min()} ถึง {true_data['Date'].max()} ({len(true_data)} วัน)")
        
        # Fill missing dates for each stock
        grouped = df.groupby('StockSymbol')
        filled_dfs = []
        
        for name, group in grouped:
            if name == 'TRUE':  # 🔧 Debug เฉพาะ TRUE
                print(f"\n🔍 Processing TRUE:")
                print(f"   📅 ก่อน fill: {group['Date'].min()} ถึง {group['Date'].max()} ({len(group)} วัน)")
                print(f"   💰 ข้อมูลจริง: {group['Close'].notna().sum()} วัน")
                print(f"   🔍 วันที่ล่าสุด: {group.iloc[-1]['Date']} (Close: {group.iloc[-1]['Close']})")
            
            # Create complete date range for this stock
            all_dates = pd.date_range(start=group['Date'].min(), end=group['Date'].max(), freq='D')
            temp_df = pd.DataFrame({'Date': all_dates})
            temp_df['StockSymbol'] = name
            # Merge with original data
            merged = pd.merge(temp_df, group, on=['StockSymbol', 'Date'], how='left')
            
            if name == 'TRUE':  # 🔧 Debug หลัง merge
                print(f"   📅 หลัง merge: {merged['Date'].min()} ถึง {merged['Date'].max()} ({len(merged)} วัน)")
                print(f"   💰 ข้อมูลจริงก่อน fill: {merged['Close'].notna().sum()} วัน")
                print(f"   🔍 วันที่ล่าสุด: {merged.iloc[-1]['Date']} (Close: {merged.iloc[-1]['Close']})")
            
            # Forward fill missing values
            financial_cols = [
                'TotalRevenue', 'QoQGrowth', 'EPS', 'ROE',
                'NetProfitMargin', 'DebtToEquity', 'PERatio', 'Dividend_Yield'
            ]
            merged[financial_cols] = merged[financial_cols].fillna(0)
            merged = merged.ffill()
            
            if name == 'TRUE':  # 🔧 Debug หลัง ffill
                print(f"   📅 หลัง ffill: {merged['Date'].min()} ถึง {merged['Date'].max()} ({len(merged)} วัน)")
                print(f"   💰 วันที่ล่าสุด: {merged.iloc[-1]['Date']} = {merged.iloc[-1]['Close']}")
                # ตรวจสอบ critical columns
                critical_check = merged.iloc[-1][['Open', 'High', 'Low', 'Close']].isna()
                print(f"   🔍 Critical columns สำหรับวันล่าสุด: {critical_check.to_dict()}")
            
            filled_dfs.append(merged)
        
        df = pd.concat(filled_dfs, ignore_index=True)
        
        # 🔧 Debug หลัง concat
        print(f"\n📊 หลัง fill missing dates:")
        true_data = df[df['StockSymbol'] == 'TRUE']
        if not true_data.empty:
            print(f"   🏷️ TRUE data: {true_data['Date'].min()} ถึง {true_data['Date'].max()} ({len(true_data)} วัน)")
            print(f"   💰 วันที่ล่าสุด: {true_data.iloc[-1]['Date']} = {true_data.iloc[-1]['Close']}")
        
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
        
        # 🔧 Debug หลัง calculate indicators
        print(f"\n📊 หลัง calculate indicators:")
        true_data = df[df['StockSymbol'] == 'TRUE']
        if not true_data.empty:
            print(f"   🏷️ TRUE data: {true_data['Date'].min()} ถึง {true_data['Date'].max()} ({len(true_data)} วัน)")
            print(f"   💰 วันที่ล่าสุด: {true_data.iloc[-1]['Date']} = {true_data.iloc[-1]['Close']}")
            # ตรวจสอบ critical columns หลัง indicators
            critical_check = true_data.iloc[-1][['Open', 'High', 'Low', 'Close']].isna()
            print(f"   🔍 Critical columns: {critical_check.to_dict()}")
        
        # Handle missing values
        critical_columns = ['Open', 'High', 'Low', 'Close']
        before_drop = len(df)
        before_drop_true = len(df[df['StockSymbol'] == 'TRUE'])
        
        df = df.dropna(subset=critical_columns)
        
        after_drop = len(df)
        after_drop_true = len(df[df['StockSymbol'] == 'TRUE'])
        
        # 🔧 Debug หลัง dropna
        print(f"\n📊 หลัง dropna critical columns:")
        print(f"   📋 ทั้งหมด: ลบออก {before_drop - after_drop} แถว ({before_drop} → {after_drop})")
        print(f"   🏷️ TRUE: ลบออก {before_drop_true - after_drop_true} แถว ({before_drop_true} → {after_drop_true})")
        
        true_data = df[df['StockSymbol'] == 'TRUE']
        if not true_data.empty:
            print(f"   🏷️ TRUE data: {true_data['Date'].min()} ถึง {true_data['Date'].max()} ({len(true_data)} วัน)")
            print(f"   💰 วันที่ล่าสุด: {true_data.iloc[-1]['Date']} = {true_data.iloc[-1]['Close']}")
        else:
            print(f"   ❌ TRUE data หายไปหมด!")
        
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
        
        # 🔧 Final debug
        print(f"\n📊 ข้อมูลสุดท้าย:")
        true_data = df[df['StockSymbol'] == 'TRUE']
        if not true_data.empty:
            print(f"   🏷️ TRUE data: {true_data['Date'].min()} ถึง {true_data['Date'].max()} ({len(true_data)} วัน)")
            print(f"   💰 วันที่ล่าสุด: {true_data.iloc[-1]['Date']} = {true_data.iloc[-1]['Close']}")
        
        print(f"\n✅ ข้อมูลพร้อมใช้งาน: {len(df)} แถว, {len(df['StockSymbol'].unique())} หุ้น")
        print(f"📊 Technical indicators ที่คำนวณได้: {[col for col in technical_columns if col in df.columns]}")
        
        return df
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def verify_model_paths():
    """ตรวจสอบ path ของโมเดลก่อนใช้งาน"""
    import os
    
    lstm_path = "../LSTM_model/best_v6_plus_minimal_tuning_v2_final_model.keras"
    gru_path = "../GRU_Model/best_v6_plus_minimal_tuning_v2_final_model.keras"
    scaler_path = "../LSTM_model/ticker_scalers.pkl"
    
    print("🔍 Verifying model paths...")
    
    if os.path.exists(lstm_path):
        size = os.path.getsize(lstm_path)
        print(f"✅ LSTM model found: {size:,} bytes")
    else:
        print(f"❌ LSTM model not found: {lstm_path}")
        return False
    
    if os.path.exists(gru_path):
        size = os.path.getsize(gru_path)
        print(f"✅ GRU model found: {size:,} bytes")
    else:
        print(f"❌ GRU model not found: {gru_path}")
        return False
    
    if os.path.exists(scaler_path):
        print(f"✅ Scaler file found")
    else:
        print(f"❌ Scaler file not found: {scaler_path}")
        return False
    
    return True

# เรียกใช้ตรวจสอบก่อนโหลดโมเดล
if not verify_model_paths():
    print("❌ Path verification failed!")
    sys.exit(1)


if __name__ == "__main__":
    print("\n🚀 เริ่มต้นระบบทำนายหุ้นแบบ Walk-Forward Validation Only")
    print("🔧 Using Walk-Forward Validation with Mini-Retrain")
    print("⚡ ระบบจะ retrain ทุกๆ 5 วันระหว่างการทำนาย")

    # โหลดโมเดล LSTM และ GRU
    print("\n🤖 กำลังโหลดโมเดล LSTM และ GRU...")

    MODEL_LSTM_PATH = "../LSTM_model/best_v6_plus_minimal_tuning_v2_final_model.keras"
    MODEL_GRU_PATH = "../GRU_Model/best_v6_plus_minimal_tuning_v2_final_model.keras"

    if not verify_model_paths():
        print("❌ Model path verification failed!")
        sys.exit(1)

    try:
        # สร้าง instance ของ WalkForwardMiniRetrainManager
        manager = WalkForwardMiniRetrainManager(
            lstm_model_path=MODEL_LSTM_PATH,
            gru_model_path=MODEL_GRU_PATH,
            retrain_frequency=RETRAIN_FREQUENCY
        )
        
        # โหลดโมเดลสำหรับ Walk-Forward Validation + Mini-Retrain (compile เพื่อการ retrain)
        print(f"\n✅ โหลดโมเดลสำหรับ Walk-Forward + Mini-Retrain...")
        model_lstm_retrain = manager.load_models_for_prediction(model_path=MODEL_LSTM_PATH, compile_model=True)
        model_gru_retrain = manager.load_models_for_prediction(model_path=MODEL_GRU_PATH, compile_model=True)
        
        if model_lstm_retrain is None or model_gru_retrain is None:
            print("❌ ไม่สามารถโหลดโมเดลได้")
            sys.exit()
            
        print("✅ โหลดโมเดลทั้งหมดสำเร็จ!")
        print(f"🔧 LSTM retrain model: {'Compiled' if hasattr(model_lstm_retrain, 'optimizer') and model_lstm_retrain.optimizer else 'Not compiled'}")
        print(f"🔧 GRU retrain model: {'Compiled' if hasattr(model_gru_retrain, 'optimizer') and model_gru_retrain.optimizer else 'Not compiled'}")
        
        # ======== 📅 บันทึกวันที่เริ่มต้น Retrain Process ========
        try:
            start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open('last_retrain_start.txt', 'w', encoding='utf-8') as f:
                f.write(f"Retrain Process Started: {start_time}\n")
                f.write(f"LSTM Model Loaded: {'✅ Success' if model_lstm_retrain else '❌ Failed'}\n")
                f.write(f"GRU Model Loaded: {'✅ Success' if model_gru_retrain else '❌ Failed'}\n")
            print(f"📅 บันทึกเวลาเริ่มต้น retrain: {start_time}")
        except Exception as log_e:
            print(f"⚠️ ไม่สามารถบันทึกเวลาเริ่มต้น retrain: {log_e}")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        import traceback
        traceback.print_exc()
        sys.exit()

    # ดึงและเตรียมข้อมูล
    print("\n📥 กำลังดึงข้อมูลจากฐานข้อมูล...")
    raw_df = fetch_latest_data()

    if raw_df.empty:
        print("❌ ไม่มีข้อมูลสำหรับประมวลผล")
        sys.exit()

    print(f"📊 ได้รับข้อมูลดิบ: {len(raw_df)} แถว จาก {len(raw_df['StockSymbol'].unique())} หุ้น")

    # ======== TRAINING-COMPATIBLE DATA PROCESSING ========
    print(f"\n🔧 เริ่มต้นระบบ Training-Compatible Data Processing...")
    df_processed, training_features = process_data_training_compatible_enhanced(raw_df)
    
    print(f"✅ Data processed for training compatibility:")
    print(f"   📊 Processed rows: {len(df_processed)}")
    print(f"   🔧 Training-compatible features: {len(training_features)}")

    # ======== TRAINING-COMPATIBLE SCALERS ========
    print("\n🔧 เริ่มต้นระบบ Training-Compatible Scalers...")
    ticker_scalers, scalers_loaded = load_training_scalers("../LSTM_model/ticker_scalers.pkl")
    
    if not scalers_loaded:
        print("💡 สร้าง scalers ใหม่...")
        ticker_encoder_temp = LabelEncoder()
        df_processed["Ticker_ID"] = ticker_encoder_temp.fit_transform(df_processed["StockSymbol"])
        
        us_stock = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'TSM', 'AMD']
        thai_stock = ['ADVANC', 'TRUE', 'DITTO', 'DIF', 'INSET', 'JMART', 'INET', 'JAS', 'HUMAN']
        df_processed['Market_ID'] = df_processed['StockSymbol'].apply(
            lambda x: 0 if x in us_stock else 1 if x in thai_stock else 2
        )
        
        ticker_scalers = create_unified_ticker_scalers(df_processed, training_features)
        
        if len(ticker_scalers) == 0:
            print("❌ ไม่สามารถสร้าง scalers ได้")
            sys.exit()

    # เตรียม prediction dataframe
    prediction_df = df_processed.copy()
    
    if 'Ticker_ID' not in prediction_df.columns:
        ticker_encoder = LabelEncoder()
        prediction_df["Ticker_ID"] = ticker_encoder.fit_transform(prediction_df["StockSymbol"])
    else:
        ticker_encoder = LabelEncoder()
        ticker_encoder.fit(prediction_df["StockSymbol"])
    
    if 'Market_ID' not in prediction_df.columns:
        us_stock = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'TSM', 'AMD']
        thai_stock = ['ADVANC', 'TRUE', 'DITTO', 'DIF', 'INSET', 'JMART', 'INET', 'JAS', 'HUMAN']
        prediction_df['Market_ID'] = prediction_df['StockSymbol'].apply(
            lambda x: 0 if x in us_stock else 1 if x in thai_stock else 2
        )
        market_encoder = LabelEncoder()
        market_encoder.fit(['US', 'TH', 'OTHER'])
    else:
        market_encoder = LabelEncoder()
        market_encoder.fit(prediction_df['Market_ID'].astype(str).unique())

    # ตรวจสอบ scalers
    valid_scalers, validation_results = validate_ticker_scalers(
        ticker_scalers, prediction_df, training_features
    )
    
    if len(valid_scalers) == 0:
        print("❌ ไม่มี scaler ที่ใช้งานได้")
        sys.exit()

    print(f"✅ Scalers พร้อมใช้งาน: {len(valid_scalers)} tickers")

    # ======== 🔄 WALK-FORWARD VALIDATION WITH MINI-RETRAIN ONLY ========
    print(f"\n🚶 เริ่มต้น Walk-Forward Validation พร้อม Mini-Retrain...")
    print(f"🔄 ระบบจะ retrain ทุกๆ {RETRAIN_FREQUENCY} วันระหว่างการทำนาย")
    
    try:
        # เทรนโมเดลหลักทั้งคู่พร้อมกัน (LSTM และ GRU) พร้อม chunk_size ที่แตกต่างกัน
        print(f"🔧 กำลังรัน Walk-Forward Validation พร้อม Mini-Retrain สำหรับทั้ง LSTM และ GRU")
        print(f"   📦 LSTM chunk_size: 90, GRU chunk_size: 200")
        
        # เก็บผลลัพธ์จากทั้งสองโมเดล
        all_walk_predictions = []
        all_walk_metrics = []
        
        models_to_train = [
            ("LSTM", model_lstm_retrain, 90),   # LSTM ใช้ chunk_size = 90 และโมเดลที่ compile แล้ว
            ("GRU", model_gru_retrain, 200)     # GRU ใช้ chunk_size = 200 และโมเดลที่ compile แล้ว
        ]
        
        for model_name, model, chunk_size in models_to_train:
            print(f"\n🔧 กำลังประมวลผล {model_name} Model (chunk_size={chunk_size})...")
            
            # เรียกใช้ Walk-Forward Validation พร้อม Mini-Retrain สำหรับแต่ละโมเดล
            walk_predictions_df, walk_metrics = walk_forward_validation_multi_task_batch(
                model=model,
                df=prediction_df,
                feature_columns=training_features,
                ticker_scalers=valid_scalers,
                ticker_encoder=ticker_encoder,
                market_encoder=market_encoder,
                seq_length=10,
                retrain_frequency=RETRAIN_FREQUENCY,  # Mini-retrain ทุกๆ 5 วัน
                chunk_size=chunk_size  # ใช้ chunk_size ที่แตกต่างกันตามโมเดล
            )
            
            if not walk_predictions_df.empty:
                # เพิ่ม model name และ chunk_size ใน dataframe
                walk_predictions_df['Model_Type'] = model_name
                walk_predictions_df['Chunk_Size'] = chunk_size
                all_walk_predictions.append(walk_predictions_df)
                
                # ✅ แก้ไข: ตรวจสอบประเภทของ metrics ก่อน assign
                processed_metrics = []
                for metric in walk_metrics:
                    if isinstance(metric, dict):
                        metric['Model_Type'] = model_name
                        metric['Chunk_Size'] = chunk_size
                        processed_metrics.append(metric)
                    elif isinstance(metric, str):
                        # สร้าง dict ใหม่สำหรับ string metrics
                        print(f"⚠️ Found unexpected metric type in {model_name}: {type(metric)}")
                        processed_metrics.append({
                            'unexpected_metric': str(metric),
                            'Model_Type': model_name,
                            'Chunk_Size': chunk_size,
                            'Direction_Accuracy': 0,
                            'MAE': 0
                        })
                
                all_walk_metrics.extend(processed_metrics)
                
                print(f"✅ {model_name} Walk-Forward Validation เสร็จสิ้น:")
                print(f"   📊 Predictions: {len(walk_predictions_df)}")
                print(f"   📈 Metrics: {len(processed_metrics)}")
            else:
                print(f"⚠️ {model_name}: ไม่มีข้อมูลผลลัพธ์")
        
        # รวมผลลัพธ์จากทั้งสองโมเดล
        if all_walk_predictions:
            combined_walk_df = pd.concat(all_walk_predictions, ignore_index=True)
            combined_walk_df.to_csv('combined_walk_forward_predictions.csv', index=False)
            print(f"💾 บันทึกผลลัพธ์รวม: {len(combined_walk_df)} predictions")
            
            # สรุปผลตามโมเดล
            print(f"\n📊 สรุปผลลัพธ์ Walk-Forward Validation:")
            for model_name in ['LSTM', 'GRU']:
                model_data = combined_walk_df[combined_walk_df['Model_Type'] == model_name]
                if not model_data.empty:
                    avg_acc = model_data.groupby('Ticker')['Actual_Dir'].count().mean()
                    print(f"   {model_name}: {len(model_data)} predictions, avg per stock: {avg_acc:.1f}")
        
        # บันทึก metrics รวม
        if all_walk_metrics:
            metrics_df = pd.DataFrame(all_walk_metrics)
            metrics_df.to_csv('combined_walk_forward_metrics.csv', index=False)
            print(f"💾 บันทึก metrics รวม: {len(metrics_df)} metrics")
        
        # ======== 📅 บันทึกวันที่หลังจาก Walk-Forward Validation เสร็จ ========
        try:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open('last_retrain_walkforward.txt', 'w', encoding='utf-8') as f:
                f.write(f"Last Walk-Forward Validation: {current_time}\n")
                f.write(f"LSTM Model: {'✅ Success' if model_lstm_retrain else '❌ Failed'}\n")
                f.write(f"GRU Model: {'✅ Success' if model_gru_retrain else '❌ Failed'}\n")
                f.write(f"Total Predictions: {len(combined_walk_df) if 'combined_walk_df' in locals() else 0}\n")
                f.write(f"Total Metrics: {len(metrics_df) if 'metrics_df' in locals() else 0}\n")
            print(f"📅 บันทึกวันที่ Walk-Forward Validation: {current_time}")
        except Exception as log_e:
            print(f"⚠️ ไม่สามารถบันทึกวันที่ Walk-Forward Validation: {log_e}")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดใน Walk-Forward Validation: {e}")
        import traceback
        traceback.print_exc()

    # ======== 🎯 ENSEMBLE PREDICTION FOR LATEST DATA ========
    print(f"\n🎯 เริ่มต้นการทำนายล่าสุดด้วย Enhanced Ensemble...")
    
    try:
        # โหลดโมเดลสำหรับ prediction (ไม่ต้อง compile)
        model_lstm_pred = manager.load_models_for_prediction(model_path=MODEL_LSTM_PATH, compile_model=False)
        model_gru_pred = manager.load_models_for_prediction(model_path=MODEL_GRU_PATH, compile_model=False)
        
        if model_lstm_pred is None or model_gru_pred is None:
            print("❌ ไม่สามารถโหลดโมเดลสำหรับ prediction ได้")
        else:
            print("✅ โหลดโมเดลสำหรับ prediction สำเร็จ")
            
            # ทำนายด้วย training-compatible scalers
            print("🔧 ทำนายด้วย Training-Compatible Scalers...")
            future_predictions = predict_with_training_compatible_scalers(
                model_lstm=model_lstm_pred,
                model_gru=model_gru_pred, 
                df=prediction_df,
                feature_columns=training_features,
                ticker_scalers=valid_scalers,
                ticker_encoder=ticker_encoder,
                market_encoder=market_encoder,
                seq_length=10
            )
            
            if not future_predictions.empty:
                print(f"✅ ได้ผลการทำนาย: {len(future_predictions)} หุ้น")
                
                # บันทึกผลการทำนาย
                future_predictions.to_csv('latest_ensemble_predictions.csv', index=False)
                print("💾 บันทึกผลการทำนายล่าสุด: latest_ensemble_predictions.csv")
                
                # แสดงสรุปผลการทำนาย
                print(f"\n📊 สรุปผลการทำนายล่าสุด:")
                print(f"   📈 หุ้นที่ทำนายขึ้น: {len(future_predictions[future_predictions['Ensemble_Direction'] == 1])}")
                print(f"   📉 หุ้นที่ทำนายลง: {len(future_predictions[future_predictions['Ensemble_Direction'] == 0])}")
                print(f"   🎯 ความเชื่อมั่นเฉลี่ย: {future_predictions['Confidence'].mean():.3f}")
                
                # แสดงหุ้นที่มีความเชื่อมั่นสูง
                high_confidence = future_predictions[future_predictions['Confidence'] > 0.7]
                if not high_confidence.empty:
                    print(f"\n🎯 หุ้นที่มีความเชื่อมั่นสูง (>70%):")
                    for _, stock in high_confidence.iterrows():
                        direction = "📈 UP" if stock['Ensemble_Direction'] == 1 else "📉 DOWN"
                        print(f"   {stock['StockSymbol']}: {direction} "
                              f"(Price: {stock['Current_Price']:.2f} → {stock['Ensemble_Price']:.2f}, "
                              f"Change: {stock['Ensemble_Change_Pct']:+.2f}%, "
                              f"Confidence: {stock['Confidence']:.3f})")
                
                # บันทึกลงฐานข้อมูล
                print(f"\n💾 บันทึกผลการทำนายลงฐานข้อมูล...")
                save_success = save_predictions_simple(future_predictions)
                
                if save_success:
                    print("✅ บันทึกลงฐานข้อมูลสำเร็จ")
                else:
                    print("⚠️ บันทึกลงฐานข้อมูลไม่สำเร็จ")
                
                # ======== 🧠 XGBoost META-LEARNER ENSEMBLE ========
                print(f"\n🧠 เริ่มต้น Enhanced XGBoost Meta-Learner...")
                
                # เตรียมข้อมูลสำหรับ Meta-Learner
                latest_data_with_predictions = raw_df.copy()
                
                # Map predictions กลับไปยัง raw data
                prediction_map = {}
                for _, pred in future_predictions.iterrows():
                    prediction_map[pred['StockSymbol']] = {
                        'PredictionClose_LSTM': pred['LSTM_Price'],
                        'PredictionTrend_LSTM': pred['LSTM_Direction'],
                        'PredictionClose_GRU': pred['GRU_Price'],
                        'PredictionTrend_GRU': pred['GRU_Direction']
                    }
                
                # เพิ่ม predictions เข้าไปใน dataframe
                for col in ['PredictionClose_LSTM', 'PredictionTrend_LSTM', 
                           'PredictionClose_GRU', 'PredictionTrend_GRU']:
                    latest_data_with_predictions[col] = latest_data_with_predictions['StockSymbol'].map(
                        lambda x: prediction_map.get(x, {}).get(col, np.nan)
                    )
                
                # กรองเฉพาะข้อมูลล่าสุดที่มี predictions
                latest_date = latest_data_with_predictions['Date'].max()
                latest_data_filtered = latest_data_with_predictions[
                    (latest_data_with_predictions['Date'] == latest_date) &
                    (latest_data_with_predictions['PredictionClose_LSTM'].notna()) &
                    (latest_data_with_predictions['PredictionClose_GRU'].notna())
                ].copy()
                
                if not latest_data_filtered.empty:
                    print(f"🔧 ข้อมูลสำหรับ Meta-Learner: {len(latest_data_filtered)} หุ้น")
                    
                    # รัน Enhanced XGBoost Meta-Learner
                    enhanced_results = meta_learner.predict_meta(latest_data_filtered)
                    
                    if meta_learner.validate_predictions(enhanced_results):
                        print("✅ Enhanced XGBoost Meta-Learner เสร็จสิ้น!")
                        
                        # บันทึกผลลัพธ์ Enhanced
                        enhanced_results.to_csv('enhanced_meta_predictions.csv', index=False)
                        print("💾 บันทึกผลลัพธ์ Enhanced: enhanced_meta_predictions.csv")
                        
                        # แสดงสรุปผล Enhanced
                        xgb_results = enhanced_results[enhanced_results['XGB_Predicted_Price'].notna()]
                        if not xgb_results.empty:
                            print(f"\n🧠 สรุปผล Enhanced Meta-Learner:")
                            print(f"   📊 หุ้นที่ประมวลผล: {len(xgb_results)}")
                            print(f"   📈 ทิศทางขึ้น: {len(xgb_results[xgb_results['XGB_Predicted_Direction'] == 1])}")
                            print(f"   📉 ทิศทางลง: {len(xgb_results[xgb_results['XGB_Predicted_Direction'] == 0])}")
                            print(f"   🎯 ความเชื่อมั่นเฉลี่ย: {xgb_results['XGB_Confidence'].mean():.3f}")
                            print(f"   ⚖️ GRU Dominance เฉลี่ย: {xgb_results['GRU_Dominance'].mean():.1%}")
                            
                            # แสดงแนะนำการลงทุน
                            high_confidence_meta = xgb_results[xgb_results['XGB_Confidence'] > 0.6]
                            if not high_confidence_meta.empty:
                                print(f"\n🎯 แนะนำการลงทุนจาก Enhanced Meta-Learner (Confidence > 60%):")
                                for _, stock in high_confidence_meta.iterrows():
                                    action_icon = {"CONSIDER": "✅", "CAUTION": "⚠️", "AVOID": "❌", "HIGH_RISK": "🔴"}.get(
                                        stock.get('Suggested_Action', 'UNKNOWN'), "❓"
                                    )
                                    direction = "📈 UP" if stock['XGB_Predicted_Direction'] == 1 else "📉 DOWN"
                                    print(f"   {stock['StockSymbol']}: {direction} "
                                          f"(Price: {stock['Close']:.2f} → {stock['XGB_Predicted_Price']:.2f}, "
                                          f"Change: {stock.get('Price_Change_Percent', 0):+.2f}%, "
                                          f"Confidence: {stock['XGB_Confidence']:.3f}) "
                                          f"{action_icon} {stock.get('Suggested_Action', 'UNKNOWN')}")
                        else:
                            print("⚠️ ไม่มีผลลัพธ์จาก Enhanced Meta-Learner")
                    else:
                        print("❌ Enhanced Meta-Learner validation failed")
                else:
                    print("⚠️ ไม่มีข้อมูลล่าสุดสำหรับ Meta-Learner")
            else:
                print("❌ ไม่ได้ผลการทำนาย")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการทำนายล่าสุด: {e}")
        import traceback
        traceback.print_exc()

    # ======== 📊 FINAL SUMMARY & CLEANUP ========
    print(f"\n" + "="*80)
    print(f"🎉 สรุปผลการดำเนินงานทั้งหมด")
    print(f"="*80)
    
    print(f"✅ ระบบ Walk-Forward Validation พร้อม Mini-Retrain")
    print(f"   🔄 Retrain ทุกๆ {RETRAIN_FREQUENCY} วัน")
    print(f"   📦 Chunk sizes: LSTM=90, GRU=200")
    
    print(f"✅ Training-Compatible Scalers")
    print(f"   📊 Scalers ที่ใช้งานได้: {len(valid_scalers)} tickers")
    
    print(f"✅ Enhanced Ensemble Prediction")
    print(f"   🧠 Meta-Learner: {meta_learner.get_model_status()}")
    
    # ตรวจสอบไฟล์ที่สร้าง
    output_files = [
        'predictions_chunk_walkforward.csv',
        'chunk_metrics.csv', 
        'overall_metrics_per_ticker.csv',
        'combined_walk_forward_predictions.csv',
        'combined_walk_forward_metrics.csv',
        'latest_ensemble_predictions.csv',
        'enhanced_meta_predictions.csv'
    ]
    
    print(f"\n📁 ไฟล์ผลลัพธ์ที่สร้าง:")
    for file in output_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"   ✅ {file} ({size:,} bytes)")
        else:
            print(f"   ❌ {file} (ไม่พบไฟล์)")
    
    print(f"\n🚀 ระบบทำงานเสร็จสิ้นสมบูรณ์!")
    print(f"💡 สามารถตรวจสอบผลลัพธ์ได้จากไฟล์ CSV ที่สร้างขึ้น")
    print(f"📊 สำหรับการวิเคราะห์เพิ่มเติม ให้ดูที่ enhanced_meta_predictions.csv")
    
    # Cleanup memory
    try:
        import gc
        del model_lstm_retrain, model_gru_retrain
        if 'model_lstm_pred' in locals():
            del model_lstm_pred, model_gru_pred
        gc.collect()
        print(f"🧹 ทำความสะอาด memory เสร็จสิ้น")
    except:
        pass
    
    # ======== 📅 บันทึกวันที่ Retrain ล่าสุด ========
    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # บันทึกวันที่ retrain ล่าสุด
        with open('last_retrain_model.txt', 'w', encoding='utf-8') as f:
            f.write(f"Last Retrain: {current_time}\n")
            f.write(f"LSTM Retrain Frequency: {RETRAIN_FREQUENCY} days\n")
            f.write(f"GRU Retrain Frequency: {RETRAIN_FREQUENCY} days\n")
            f.write(f"Valid Scalers: {len(valid_scalers)} tickers\n")
            f.write(f"Meta-Learner Status: {meta_learner.get_model_status()}\n")
        
        print(f"📅 บันทึกวันที่ retrain ล่าสุด: {current_time}")
        print(f"💾 ไฟล์: last_retrain_model.txt")
        
        # บันทึกรายละเอียดการ retrain ลงไฟล์ log
        log_file = f"retrain_logs/retrain_log_model_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        os.makedirs('retrain_logs', exist_ok=True)
        
        retrain_log = pd.DataFrame({
            'Retrain_Date': [current_time],
            'LSTM_Retrain_Frequency': [RETRAIN_FREQUENCY],
            'GRU_Retrain_Frequency': [RETRAIN_FREQUENCY], 
            'Valid_Scalers_Count': [len(valid_scalers)],
            'Meta_Learner_Status': [meta_learner.get_model_status()],
            'Output_Files_Generated': [len([f for f in output_files if os.path.exists(f)])],
            'Total_Output_Files': [len(output_files)]
        })
        
        retrain_log.to_csv(log_file, index=False)
        print(f"📊 บันทึก retrain log: {log_file}")
        
    except Exception as e:
        print(f"⚠️ ไม่สามารถบันทึกวันที่ retrain ได้: {e}")
    
    print(f"🎯 ระบบพร้อมสำหรับการใช้งานครั้งต่อไป!")