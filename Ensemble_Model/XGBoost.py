import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
import joblib
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

## ----------------------------------
## 1. Transparent Data Preparation with Warning System
## ----------------------------------

def add_market_features_transparent(df):
    """เพิ่ม Market Features แบบโปร่งใส"""
    
    thai_tickers = ['ADVANC', 'DIF', 'DITTO', 'HUMAN', 'INET', 'INSET', 'JAS', 'JMART', 'TRUE']
    us_tickers = ['AAPL', 'AMD', 'AMZN', 'AVGO', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA', 'TSM']
    
    df['Market_ID'] = df['Ticker'].apply(lambda x: 0 if x in thai_tickers else 1)
    df['Market_Name'] = df['Market_ID'].map({0: 'Thai', 1: 'US'})
    
    # Price categories
    def get_price_category(row):
        if row['Market_ID'] == 0:  # Thai
            if row['Actual_Price'] < 10: return 0  # Low
            elif row['Actual_Price'] < 50: return 1  # Medium  
            else: return 2  # High
        else:  # US
            if row['Actual_Price'] < 100: return 0  # Low
            elif row['Actual_Price'] < 300: return 1  # Medium
            else: return 2  # High
    
    df['Price_Category'] = df.apply(get_price_category, axis=1)
    
    # Market-aware normalization (แต่เก็บข้อมูลดิบไว้)
    df['Price_Raw'] = df['Actual_Price']  # เก็บราคาดิบ
    df['Price_Market_Norm'] = df.groupby('Market_ID')['Actual_Price'].transform(
        lambda x: (x - x.median()).abs().median()  # ใช้ median และ MAD แทน mean/std
    )
    
    return df

def create_prediction_quality_metrics(df):
    """สร้างเมตริกประเมินคุณภาพการพยากรณ์"""
    
    # คำนวณ prediction errors
    df['LSTM_Error_Pct'] = abs((df['Predicted_Price_LSTM'] - df['Actual_Price']) / df['Actual_Price']) * 100
    df['GRU_Error_Pct'] = abs((df['Predicted_Price_GRU'] - df['Actual_Price']) / df['Actual_Price']) * 100
    
    # Model agreement strength
    df['Price_Agreement'] = 1 - abs(df['Predicted_Price_LSTM'] - df['Predicted_Price_GRU']) / df['Actual_Price']
    df['Price_Agreement'] = np.clip(df['Price_Agreement'], 0, 1)
    
    # Prediction stability (เอาไว้เตือนเมื่อการพยากรณ์ไม่เสถียร)
    df['Price_Volatility_Window'] = df.groupby('Ticker')['Actual_Price'].transform(
        lambda x: x.rolling(window=min(5, len(x)), min_periods=1).std()
    )
    
    return df

def detect_prediction_anomalies(df):
    """ตรวจจับการพยากรณ์ที่ผิดปกติ"""
    
    anomaly_flags = []
    
    for idx, row in df.iterrows():
        flags = []
        
        # 1. ราคาติดลบ
        if row['Predicted_Price_LSTM'] <= 0 or row['Predicted_Price_GRU'] <= 0:
            flags.append('NEGATIVE_PRICE')
        
        # 2. การเปลี่ยนแปลงสุดโต่ง
        lstm_change = abs((row['Predicted_Price_LSTM'] - row['Actual_Price']) / row['Actual_Price'])
        gru_change = abs((row['Predicted_Price_GRU'] - row['Actual_Price']) / row['Actual_Price'])
        
        threshold = 0.5 if row['Market_ID'] == 0 else 0.3  # Thai stocks มี volatility สูงกว่า
        
        if lstm_change > threshold:
            flags.append('LSTM_EXTREME')
        if gru_change > threshold:
            flags.append('GRU_EXTREME')
        
        # 3. Model disagreement สูง
        if row['Price_Agreement'] < 0.5:
            flags.append('HIGH_DISAGREEMENT')
        
        # 4. ราคาสูงกว่าหรือต่ำกว่าประวัติการณ์
        ticker_data = df[df['Ticker'] == row['Ticker']]
        price_range = ticker_data['Actual_Price'].quantile([0.01, 0.99])
        
        if (row['Predicted_Price_LSTM'] < price_range.iloc[0] * 0.5 or 
            row['Predicted_Price_LSTM'] > price_range.iloc[1] * 2):
            flags.append('OUT_OF_RANGE')
        
        anomaly_flags.append('|'.join(flags) if flags else 'NORMAL')
    
    df['Anomaly_Flags'] = anomaly_flags
    
    return df

## ----------------------------------
## 2. Improved Model Architecture
## ----------------------------------

def create_robust_features(df):
    """สร้าง features ที่แข็งแกร่งกว่า"""
    
    # เพิ่ม High และ Low
    df['High'] = df['Actual_Price'] * 1.005  # ลด noise
    df['Low'] = df['Actual_Price'] * 0.995
    
    # Robust price differences
    df['Price_Diff'] = df['Predicted_Price_LSTM'] - df['Predicted_Price_GRU']
    df['Price_Diff_Normalized'] = df['Price_Diff'] / (df['Actual_Price'] + 1e-8)
    
    # Direction features
    df['Dir_Agreement'] = (df['Predicted_Dir_LSTM'] == df['Predicted_Dir_GRU']).astype(int)
    df['Dir_Confidence'] = abs(df['Predicted_Dir_LSTM'] - 0.5) + abs(df['Predicted_Dir_GRU'] - 0.5)
    
    # Rolling statistics (แบบ robust)
    def calculate_robust_indicators(group):
        if len(group) < 14:
            return group
        
        price_series = group['Actual_Price']
        
        # RSI
        try:
            group['RSI'] = RSIIndicator(close=price_series, window=14).rsi()
        except:
            group['RSI'] = 50  # default neutral RSI
        
        # Robust trend indicators
        if len(group) >= 20:
            group['SMA_20'] = SMAIndicator(close=price_series, window=20).sma_indicator()
            
            # Bollinger Bands
            bb = BollingerBands(close=price_series, window=20, window_dev=2)
            group['BB_Upper'] = bb.bollinger_hband()
            group['BB_Lower'] = bb.bollinger_lband()
            group['BB_Position'] = (price_series - group['BB_Lower']) / (group['BB_Upper'] - group['BB_Lower'] + 1e-8)
        
        if len(group) >= 26:
            group['MACD'] = MACD(close=price_series, window_slow=26, window_fast=12).macd()
        
        # ATR
        if len(group) >= 14:
            group['ATR'] = AverageTrueRange(
                high=group['High'], 
                low=group['Low'], 
                close=price_series, 
                window=14
            ).average_true_range()
        
        return group
    
    # Apply indicators by ticker
    df = df.groupby('Ticker', include_groups=False).apply(calculate_robust_indicators).reset_index(drop=True)
    
    # Market-aware feature normalization
    market_features = ['RSI', 'BB_Position', 'Price_Diff_Normalized']
    for feature in market_features:
        if feature in df.columns:
            df[f'{feature}_Market_Norm'] = df.groupby('Market_ID')[feature].transform(
                lambda x: (x - x.median()).abs().median()
            )
    
    return df

## ----------------------------------
## 3. Transparent Training Pipeline
## ----------------------------------

try:
    # Load data
    lstm_df = pd.read_csv("../LSTM_model/all_predictions_per_day_multi_task.csv")
    gru_df = pd.read_csv("../GRU_Model/all_predictions_per_day_multi_task.csv")

    logger.info(f"Loaded LSTM: {len(lstm_df)}, GRU: {len(gru_df)} records")

    # Combine data
    df = pd.DataFrame({
        "Ticker": lstm_df["Ticker"],
        "Date": pd.to_datetime(lstm_df["Date"]),
        "Actual_Price": lstm_df["Actual_Price"],
        "Predicted_Price_LSTM": lstm_df["Predicted_Price"],
        "Predicted_Price_GRU": gru_df["Predicted_Price"],
        "Actual_Direction": lstm_df["Actual_Dir"],
        "Predicted_Dir_LSTM": lstm_df["Predicted_Dir"],
        "Predicted_Dir_GRU": gru_df["Predicted_Dir"]
    })

    # Apply transparent preprocessing
    df = add_market_features_transparent(df)
    df = create_prediction_quality_metrics(df)
    df = detect_prediction_anomalies(df)
    df = create_robust_features(df)

    # Log anomaly statistics
    anomaly_stats = df['Anomaly_Flags'].value_counts()
    logger.info("Anomaly Detection Results:")
    for anomaly, count in anomaly_stats.items():
        logger.info(f"  {anomaly}: {count} ({count/len(df)*100:.1f}%)")

    # Clean data but keep transparency
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # Remove insufficient data tickers
    ticker_counts = df.groupby('Ticker').size()
    insufficient_tickers = ticker_counts[ticker_counts < 26].index
    if len(insufficient_tickers) > 0:
        logger.warning(f"Removing tickers with <26 days: {list(insufficient_tickers)}")
        df = df[~df['Ticker'].isin(insufficient_tickers)]

    # Imputation strategy
    numeric_cols = ['RSI', 'SMA_20', 'MACD', 'BB_Upper', 'BB_Lower', 'BB_Position', 'ATR',
                   'Price_Diff_Normalized', 'RSI_Market_Norm', 'BB_Position_Market_Norm']
    
    # Forward/backward fill by ticker
    df[numeric_cols] = df.groupby('Ticker')[numeric_cols].ffill().bfill()
    
    # Market-aware imputation for remaining NaN
    for market_id in df['Market_ID'].unique():
        market_mask = df['Market_ID'] == market_id
        imputer = SimpleImputer(strategy='median')
        
        for col in numeric_cols:
            if df.loc[market_mask, col].isna().any():
                df.loc[market_mask, col] = imputer.fit_transform(
                    df.loc[market_mask, [col]]
                ).flatten()

    # Final cleanup
    df.dropna(subset=['Actual_Price', 'Actual_Direction'], inplace=True)
    
    # Outlier detection (but don't remove - just flag)
    for market_id in df['Market_ID'].unique():
        market_data = df[df['Market_ID'] == market_id]
        q1, q3 = market_data['Actual_Price'].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        
        outlier_mask = (market_data['Actual_Price'] < lower_bound) | (market_data['Actual_Price'] > upper_bound)
        df.loc[df['Market_ID'] == market_id, 'Is_Outlier'] = outlier_mask

    logger.info(f"Final dataset size: {len(df)}")
    logger.info(f"Outliers detected: {df['Is_Outlier'].sum()} ({df['Is_Outlier'].sum()/len(df)*100:.1f}%)")

    # Feature sets
    base_features = ['Predicted_Dir_LSTM', 'Predicted_Dir_GRU', 'Dir_Agreement', 'Dir_Confidence']
    market_features = ['Market_ID', 'Price_Category']
    technical_features = ['RSI', 'SMA_20', 'MACD', 'BB_Position', 'ATR']
    quality_features = ['Price_Agreement', 'LSTM_Error_Pct', 'GRU_Error_Pct']
    
    dir_features = base_features + market_features + technical_features + quality_features
    
    price_base = ['Predicted_Price_LSTM', 'Predicted_Price_GRU', 'Price_Diff_Normalized']
    price_features = price_base + market_features + technical_features + quality_features + ['Price_Market_Norm']

    X_dir = df[dir_features]
    y_dir = df['Actual_Direction']
    X_price = df[price_features]
    y_price = df['Actual_Price']

    # Use RobustScaler instead of StandardScaler
    scaler_dir = RobustScaler()  # Less sensitive to outliers
    scaler_price = RobustScaler()
    
    X_dir_scaled = scaler_dir.fit_transform(X_dir)
    X_price_scaled = scaler_price.fit_transform(X_price)

    # Save scalers and feature lists
    joblib.dump(scaler_dir, 'robust_scaler_dir.pkl')
    joblib.dump(scaler_price, 'robust_scaler_price.pkl')
    joblib.dump(dir_features, 'transparent_dir_features.pkl')
    joblib.dump(price_features, 'transparent_price_features.pkl')

    # Time-based split
    split_date = df['Date'].quantile(0.8)
    train_mask = df['Date'] < split_date
    test_mask = df['Date'] >= split_date

    X_train_dir, X_test_dir = X_dir_scaled[train_mask], X_dir_scaled[test_mask]
    y_train_dir, y_test_dir = y_dir[train_mask], y_dir[test_mask]
    X_train_price, X_test_price = X_price_scaled[train_mask], X_price_scaled[test_mask]
    y_train_price, y_test_price = y_price[train_mask], y_price[test_mask]
    test_indices = df.index[test_mask]

    logger.info(f"Training: {len(X_train_dir)}, Testing: {len(X_test_dir)}")

except Exception as e:
    logger.error(f"Error in transparent data preparation: {str(e)}")
    raise

## ----------------------------------
## 4. Robust Model Training
## ----------------------------------

try:
    logger.info("Training Transparent XGBoost Models")
    
    # Direction Model
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1
    )

    # Simpler grid search for stability
    param_grid_clf = {
        'n_estimators': [150, 200, 250],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.05, 0.1, 0.15]
    }
    
    grid_clf = GridSearchCV(xgb_clf, param_grid_clf, cv=3, scoring='accuracy', n_jobs=-1)
    grid_clf.fit(X_train_dir, y_train_dir)
    
    xgb_clf = grid_clf.best_estimator_
    logger.info(f"Best Direction Model params: {grid_clf.best_params_}")

    # Price Model  
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1
    )

    param_grid_reg = {
        'n_estimators': [150, 200, 250],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.05, 0.1, 0.15]
    }
    
    grid_reg = GridSearchCV(xgb_reg, param_grid_reg, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_reg.fit(X_train_price, y_train_price)
    
    xgb_reg = grid_reg.best_estimator_
    logger.info(f"Best Price Model params: {grid_reg.best_params_}")

    # Predictions
    y_pred_dir = xgb_clf.predict(X_test_dir)
    y_pred_dir_proba = xgb_clf.predict_proba(X_test_dir)[:, 1]
    y_pred_price = xgb_reg.predict(X_test_price)

    # NO CLIPPING - Keep raw predictions for transparency
    
    # Model evaluation
    dir_accuracy = accuracy_score(y_test_dir, y_pred_dir)
    price_rmse = np.sqrt(mean_squared_error(y_test_price, y_pred_price))
    price_r2 = r2_score(y_test_price, y_pred_price)

    logger.info(f"Direction Accuracy: {dir_accuracy:.4f}")
    logger.info(f"Price RMSE: {price_rmse:.4f}")
    logger.info(f"Price R²: {price_r2:.4f}")

    # Feature importance
    dir_importance = pd.DataFrame({
        'feature': dir_features,
        'importance': xgb_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    price_importance = pd.DataFrame({
        'feature': price_features,
        'importance': xgb_reg.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info("Top Direction Features:")
    logger.info(dir_importance.head(8).to_string(index=False))
    
    logger.info("Top Price Features:")
    logger.info(price_importance.head(8).to_string(index=False))

    # Save models
    joblib.dump(xgb_clf, 'transparent_xgb_classifier.pkl')
    joblib.dump(xgb_reg, 'transparent_xgb_regressor.pkl')

except Exception as e:
    logger.error(f"Error in model training: {str(e)}")
    raise

## ----------------------------------
## 5. Transparent Prediction Function
## ----------------------------------

def predict_with_transparency(new_data, return_diagnostics=True):
    """
    ทำนายแบบโปร่งใส - แสดงทั้งผลดิบและคำเตือน
    """
    try:
        # Load models and scalers
        scaler_dir = joblib.load('robust_scaler_dir.pkl')
        scaler_price = joblib.load('robust_scaler_price.pkl')
        xgb_clf = joblib.load('transparent_xgb_classifier.pkl')
        xgb_reg = joblib.load('transparent_xgb_regressor.pkl')
        dir_features = joblib.load('transparent_dir_features.pkl')
        price_features = joblib.load('transparent_price_features.pkl')
        
        # Preprocess new data (same as training)
        processed_data = new_data.copy()
        processed_data = add_market_features_transparent(processed_data)
        processed_data = create_prediction_quality_metrics(processed_data)
        processed_data = detect_prediction_anomalies(processed_data)
        processed_data = create_robust_features(processed_data)
        
        # Handle missing values
        numeric_cols = ['RSI', 'SMA_20', 'MACD', 'BB_Upper', 'BB_Lower', 'BB_Position', 'ATR']
        for col in numeric_cols:
            if col in processed_data.columns:
                processed_data[col] = processed_data.groupby('Ticker')[col].ffill().bfill()
                if processed_data[col].isna().any():
                    processed_data[col] = processed_data[col].fillna(processed_data[col].median())
        
        # Make predictions
        X_dir = scaler_dir.transform(processed_data[dir_features])
        X_price = scaler_price.transform(processed_data[price_features])
        
        pred_dir = xgb_clf.predict(X_dir)
        pred_dir_proba = xgb_clf.predict_proba(X_dir)[:, 1]
        pred_price_raw = xgb_reg.predict(X_price)  # RAW predictions
        
        # Calculate prediction quality metrics
        price_change_pct = ((pred_price_raw - processed_data['Actual_Price']) / processed_data['Actual_Price']) * 100
        
        # Create transparency metrics
        def get_reliability_score(row, pred_price, price_change):
            score = 1.0
            warnings = []
            
            # Check for extreme predictions
            if abs(price_change) > 50:
                score *= 0.1
                warnings.append("EXTREME_CHANGE")
            elif abs(price_change) > 20:
                score *= 0.3
                warnings.append("HIGH_CHANGE")
            
            # Check for negative prices
            if pred_price <= 0:
                score *= 0.0
                warnings.append("NEGATIVE_PRICE")
            
            # Check model agreement
            if row.get('Price_Agreement', 1) < 0.7:
                score *= 0.7
                warnings.append("LOW_AGREEMENT")
            
            # Check for anomalies
            if row.get('Anomaly_Flags', 'NORMAL') != 'NORMAL':
                score *= 0.5
                warnings.append("ANOMALY_DETECTED")
            
            return score, '|'.join(warnings) if warnings else 'RELIABLE'
        
        reliability_scores = []
        reliability_warnings = []
        
        for idx, row in processed_data.iterrows():
            score, warning = get_reliability_score(row, pred_price_raw[idx], price_change_pct.iloc[idx])
            reliability_scores.append(score)
            reliability_warnings.append(warning)
        
        # Create comprehensive results
        results = pd.DataFrame({
            'Ticker': processed_data['Ticker'],
            'Date': processed_data['Date'],
            'Market_Name': processed_data['Market_Name'],
            'Last_Close': processed_data['Actual_Price'],
            
            # RAW PREDICTIONS (no clipping)
            'Predicted_Price_Raw': pred_price_raw,
            'Price_Change_Percent_Raw': price_change_pct,
            'Predicted_Direction': pred_dir,
            'Direction_Confidence': pred_dir_proba,
            
            # QUALITY METRICS
            'Reliability_Score': reliability_scores,
            'Reliability_Warning': reliability_warnings,
            'Model_Agreement': processed_data.get('Price_Agreement', 1.0),
            'Anomaly_Flags': processed_data.get('Anomaly_Flags', 'NORMAL'),
            
            # SUGGESTED ACTIONS (based on reliability)
            'Suggested_Action': ['INVEST' if (rel > 0.7 and abs(change) < 20) 
                               else 'CAUTION' if rel > 0.3 
                               else 'AVOID' 
                               for rel, change in zip(reliability_scores, price_change_pct)]
        })
        
        # Add conservative estimates for high-risk predictions
        def get_conservative_estimate(row):
            if row['Reliability_Score'] < 0.3:
                # Very unreliable - use minimal change
                return row['Last_Close'] * (1 + np.sign(row['Price_Change_Percent_Raw']) * 0.02)
            elif row['Reliability_Score'] < 0.7:
                # Somewhat unreliable - reduce the change
                conservative_change = row['Price_Change_Percent_Raw'] * 0.3
                return row['Last_Close'] * (1 + conservative_change/100)
            else:
                # Reliable - use raw prediction
                return row['Predicted_Price_Raw']
        
        results['Predicted_Price_Conservative'] = results.apply(get_conservative_estimate, axis=1)
        results['Price_Change_Percent_Conservative'] = ((results['Predicted_Price_Conservative'] - results['Last_Close']) / results['Last_Close']) * 100
        
        if return_diagnostics:
            # Print diagnostic summary
            print("\n🔍 PREDICTION TRANSPARENCY REPORT")
            print("=" * 50)
            
            reliability_dist = pd.Series(reliability_warnings).value_counts()
            print("\nReliability Distribution:")
            for warning, count in reliability_dist.items():
                print(f"  {warning}: {count} predictions")
            
            extreme_predictions = results[abs(results['Price_Change_Percent_Raw']) > 30]
            if len(extreme_predictions) > 0:
                print(f"\n⚠️  EXTREME PREDICTIONS (>30% change): {len(extreme_predictions)}")
                for _, row in extreme_predictions.iterrows():
                    print(f"     {row['Ticker']}: {row['Price_Change_Percent_Raw']:.1f}% (Warning: {row['Reliability_Warning']})")
            
            negative_predictions = results[results['Predicted_Price_Raw'] <= 0]
            if len(negative_predictions) > 0:
                print(f"\n❌ NEGATIVE PRICE PREDICTIONS: {len(negative_predictions)}")
                for _, row in negative_predictions.iterrows():
                    print(f"     {row['Ticker']}: ${row['Predicted_Price_Raw']:.2f}")
            
            print(f"\n📊 SUMMARY:")
            print(f"   Reliable predictions (>0.7): {len(results[results['Reliability_Score'] > 0.7])}")
            print(f"   Questionable predictions (0.3-0.7): {len(results[(results['Reliability_Score'] > 0.3) & (results['Reliability_Score'] <= 0.7)])}")
            print(f"   Unreliable predictions (<0.3): {len(results[results['Reliability_Score'] <= 0.3])}")
            
        return results
        
    except Exception as e:
        logger.error(f"Error in transparent prediction: {str(e)}")
        raise

# Example usage function
def demonstrate_transparency():
    """แสดงตัวอย่างการใช้งานแบบโปร่งใส"""
    
    print("\n🎓 HOW TO USE TRANSPARENT PREDICTIONS:")
    print("=" * 50)
    print("1. Check 'Reliability_Score' (0-1, higher = better)")
    print("2. Read 'Reliability_Warning' for specific issues")
    print("3. Use 'Predicted_Price_Raw' for full transparency")
    print("4. Use 'Predicted_Price_Conservative' for safer estimates")
    print("5. Follow 'Suggested_Action' (INVEST/CAUTION/AVOID)")
    print("\n💡 INTERPRETATION GUIDE:")
    print("   Reliability > 0.7: High confidence, use raw prediction")
    print("   Reliability 0.3-0.7: Medium confidence, use conservative estimate")
    print("   Reliability < 0.3: Low confidence, avoid or investigate further")

if __name__ == "__main__":
    logger.info("Transparent ensemble model training completed!")
    demonstrate_transparency()