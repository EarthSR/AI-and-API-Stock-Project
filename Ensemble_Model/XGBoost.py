import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
import joblib
import logging
import warnings

# ปิด warning ที่ไม่จำเป็น
warnings.filterwarnings("ignore", category=UserWarning)

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

## ----------------------------------
## 1. การเตรียมข้อมูล (Data Preparation)
## ----------------------------------
try:
    # โหลดข้อมูล
    lstm_df = pd.read_csv("../LSTM_model/all_predictions_per_day_multi_task.csv")
    gru_df = pd.read_csv("../GRU_Model/all_predictions_per_day_multi_task.csv")

    logger.info(f"LSTM data: {len(lstm_df)} records")
    logger.info(f"GRU data: {len(gru_df)} records")

    # ตรวจสอบความสอดคล้องของข้อมูล
    if len(lstm_df) != len(gru_df):
        raise ValueError("LSTM and GRU data have different lengths")

    # รวมข้อมูลจากสองโมเดลหลัก
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

    # เพิ่ม High และ Low สำหรับ ATR (สมมติว่าไม่มีใน CSV เดิม)
    df['High'] = df['Actual_Price'] * 1.01
    df['Low'] = df['Actual_Price'] * 0.99

    # เรียงข้อมูลตาม Ticker และ Date
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # ตรวจสอบจำนวนวันต่อ Ticker
    ticker_counts = df.groupby('Ticker').size()
    logger.info(f"Number of days per Ticker:\n{ticker_counts.to_string()}")
    insufficient_data_tickers = ticker_counts[ticker_counts < 26].index
    if len(insufficient_data_tickers) > 0:
        logger.warning(f"Tickers with insufficient data (<26 days): {list(insufficient_data_tickers)}")
        df = df[~df['Ticker'].isin(insufficient_data_tickers)]

    # Feature Engineering
    df['Price_Diff'] = df['Predicted_Price_LSTM'] - df['Predicted_Price_GRU']
    df['Dir_Agreement'] = (df['Predicted_Dir_LSTM'] == df['Predicted_Dir_GRU']).astype(int)
    df['RSI'] = df.groupby('Ticker')['Actual_Price'].transform(lambda x: RSIIndicator(close=x, window=14).rsi() if len(x) >= 14 else pd.Series(np.nan, index=x.index))
    df['SMA_20'] = df.groupby('Ticker')['Actual_Price'].transform(lambda x: SMAIndicator(close=x, window=20).sma_indicator() if len(x) >= 20 else pd.Series(np.nan, index=x.index))
    df['MACD'] = df.groupby('Ticker')['Actual_Price'].transform(lambda x: MACD(close=x).macd() if len(x) >= 26 else pd.Series(np.nan, index=x.index))
    df['BB_High'] = df.groupby('Ticker')['Actual_Price'].transform(lambda x: BollingerBands(close=x, window=20).bollinger_hband() if len(x) >= 20 else pd.Series(np.nan, index=x.index))
    df['BB_Low'] = df.groupby('Ticker')['Actual_Price'].transform(lambda x: BollingerBands(close=x, window=20).bollinger_lband() if len(x) >= 20 else pd.Series(np.nan, index=x.index))
    df['ATR'] = df.groupby('Ticker').apply(lambda g: AverageTrueRange(high=g['High'], low=g['Low'], close=g['Actual_Price'], window=14).average_true_range() if len(g) >= 14 else pd.Series(np.nan, index=g.index)).reset_index(level=0, drop=True)

    # Impute missing values ด้วย ffill และ bfill
    df[['RSI', 'SMA_20', 'MACD', 'BB_High', 'BB_Low', 'ATR']] = df.groupby('Ticker')[['RSI', 'SMA_20', 'MACD', 'BB_High', 'BB_Low', 'ATR']].ffill().bfill()
    imputer = SimpleImputer(strategy='mean')
    df[['Price_Diff', 'RSI', 'SMA_20', 'MACD', 'BB_High', 'BB_Low', 'ATR']] = imputer.fit_transform(df[['Price_Diff', 'RSI', 'SMA_20', 'MACD', 'BB_High', 'BB_Low', 'ATR']])
    df.dropna(subset=['Actual_Price', 'Actual_Direction'], inplace=True)

    # Normalize Actual_Price ตาม Ticker
    df['Actual_Price_Normalized'] = df.groupby('Ticker')['Actual_Price'].transform(lambda x: (x - x.mean()) / x.std())

    # ตรวจสอบ technical indicators
    logger.info(f"RSI statistics: min={df['RSI'].min():.2f}, max={df['RSI'].max():.2f}, mean={df['RSI'].mean():.2f}")
    logger.info(f"SMA_20 statistics: min={df['SMA_20'].min():.2f}, max={df['SMA_20'].max():.2f}, mean={df['SMA_20'].mean():.2f}")
    logger.info(f"MACD statistics: min={df['MACD'].min():.2f}, max={df['MACD'].max():.2f}, mean={df['MACD'].mean():.2f}")
    logger.info(f"ATR statistics: min={df['ATR'].min():.2f}, max={df['ATR'].max():.2f}, mean={df['ATR'].mean():.2f}")

    # ตรวจสอบหน่วยข้อมูล
    logger.info(f"Actual_Price stats: min={df['Actual_Price'].min():.2f}, max={df['Actual_Price'].max():.2f}, mean={df['Actual_Price'].mean():.2f}")
    logger.info(f"Actual_Price Normalized stats: min={df['Actual_Price_Normalized'].min():.2f}, max={df['Actual_Price_Normalized'].max():.2f}, mean={df['Actual_Price_Normalized'].mean():.2f}")

    # กรอง outlier ใน Actual_Price
    q_low, q_high = df['Actual_Price'].quantile([0.01, 0.99])
    df = df[df['Actual_Price'].between(q_low, q_high)]
    logger.info(f"Data size after outlier filtering: {len(df)}")

    # Reset index
    df = df.reset_index(drop=True)

    # Feature และ target
    X_dir = df[['Predicted_Dir_LSTM', 'Predicted_Dir_GRU', 'Dir_Agreement', 'RSI', 'SMA_20', 'MACD', 'BB_High', 'BB_Low', 'ATR']]
    y_dir = df['Actual_Direction']
    X_price = df[['Predicted_Price_LSTM', 'Predicted_Price_GRU', 'Price_Diff', 'RSI', 'SMA_20', 'MACD', 'BB_High', 'BB_Low', 'ATR', 'Actual_Price_Normalized']]
    y_price = df['Actual_Price']

    # Normalize ข้อมูล
    scaler_dir = StandardScaler()
    scaler_price = StandardScaler()
    X_dir_scaled = scaler_dir.fit_transform(X_dir)
    X_price_scaled = scaler_price.fit_transform(X_price)

    # บันทึก scaler
    joblib.dump(scaler_dir, 'scaler_dir.pkl')
    joblib.dump(scaler_price, 'scaler_price.pkl')

    # แบ่งข้อมูลตามวันที่ (Time-based Split)
    split_date = df['Date'].quantile(0.8)
    train_mask = df['Date'] < split_date
    test_mask = df['Date'] >= split_date

    X_train_dir, X_test_dir = X_dir_scaled[train_mask], X_dir_scaled[test_mask]
    y_train_dir, y_test_dir = y_dir[train_mask], y_dir[test_mask]
    X_train_price, X_test_price = X_price_scaled[train_mask], X_price_scaled[test_mask]
    y_train_price, y_test_price = y_price[train_mask], y_price[test_mask]
    test_indices = df.index[test_mask]

    logger.info(f"Training set size: {len(X_train_dir)}")
    logger.info(f"Test set size: {len(X_test_dir)}")
    logger.info(f"Split date: {split_date}")
    logger.info(f"Training Actual_Direction distribution:\n{y_train_dir.value_counts().to_string()}")
    logger.info(f"Test Actual_Direction distribution:\n{y_test_dir.value_counts().to_string()}")

except Exception as e:
    logger.error(f"Error in data preparation: {str(e)}")
    raise

## ------------------------------------------
## 2. สร้างโมเดลทำนายทิศทาง (XGBClassifier)
## ------------------------------------------
try:
    logger.info("Training Direction Prediction Model (XGBClassifier)")
    
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )

    # Grid Search สำหรับ hyperparameter tuning
    param_grid_clf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    }
    grid_clf = GridSearchCV(xgb_clf, param_grid_clf, cv=5, scoring='accuracy', n_jobs=-1)
    grid_clf.fit(X_train_dir, y_train_dir)

    logger.info(f"Best parameters for XGBClassifier: {grid_clf.best_params_}")
    xgb_clf = grid_clf.best_estimator_

    # ทำนายผลและ confidence score
    y_pred_dir = xgb_clf.predict(X_test_dir)
    y_pred_dir_proba = xgb_clf.predict_proba(X_test_dir)[:, 1]

    # ประเมินผล
    accuracy = accuracy_score(y_test_dir, y_pred_dir)
    logger.info(f"Direction Model Accuracy: {accuracy:.4f}")
    logger.info("Classification Report:")
    logger.info(classification_report(y_test_dir, y_pred_dir))

    # Log distribution ของ probability
    logger.info(f"XGB_Predicted_Direction_Proba stats: min={y_pred_dir_proba.min():.2f}, max={y_pred_dir_proba.max():.2f}, mean={y_pred_dir_proba.mean():.2f}")

    # บันทึกโมเดล
    joblib.dump(xgb_clf, 'xgb_classifier_model.pkl')
    logger.info("XGBClassifier model saved as xgb_classifier_model.pkl")

except Exception as e:
    logger.error(f"Error in direction model training: {str(e)}")
    raise

## -----------------------------------------
## 3. สร้างโมเดลทำนายราคา (XGBRegressor)
## -----------------------------------------
try:
    logger.info("Training Price Prediction Model (XGBRegressor)")
    
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )

    # Grid Search สำหรับ hyperparameter tuning
    param_grid_reg = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    }
    grid_reg = GridSearchCV(xgb_reg, param_grid_reg, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_reg.fit(X_train_price, y_train_price)

    logger.info(f"Best parameters for XGBRegressor: {grid_reg.best_params_}")
    xgb_reg = grid_reg.best_estimator_

    # ทำนายผล
    y_pred_price = xgb_reg.predict(X_test_price)

    # ประเมินผล
    rmse = np.sqrt(mean_squared_error(y_test_price, y_pred_price))
    r2 = r2_score(y_test_price, y_pred_price)
    logger.info(f"Price Model RMSE: {rmse:.4f}")
    logger.info(f"Price Model R-squared (R²): {r2:.4f}")

    # บันทึกโมเดล
    joblib.dump(xgb_reg, 'xgb_regressor_model.pkl')
    logger.info("XGBRegressor model saved as xgb_regressor_model.pkl")

except Exception as e:
    logger.error(f"Error in price model training: {str(e)}")
    raise

## -----------------------------------
## 4. Backtesting และเมตริกเพิ่มเติม
## -----------------------------------
try:
    logger.info("Performing backtesting with additional metrics")
    
    # สร้าง DataFrame สำหรับผลลัพธ์
    results_df = pd.DataFrame({
        'Ticker': df.loc[test_indices, 'Ticker'],
        'Date': df.loc[test_indices, 'Date'],
        'Actual_Price': y_test_price,
        'Predicted_Price_LSTM': df.loc[test_indices, 'Predicted_Price_LSTM'],
        'Predicted_Price_GRU': df.loc[test_indices, 'Predicted_Price_GRU'],
        'XGB_Predicted_Price': y_pred_price,
        'Actual_Direction': y_test_dir,
        'Predicted_Dir_LSTM': df.loc[test_indices, 'Predicted_Dir_LSTM'],
        'Predicted_Dir_GRU': df.loc[test_indices, 'Predicted_Dir_GRU'],
        'XGB_Predicted_Direction': y_pred_dir,
        'XGB_Predicted_Direction_Proba': y_pred_dir_proba
    }, index=test_indices)

    # เรียงข้อมูลตาม Ticker และ Date
    results_df = results_df.sort_values(['Ticker', 'Date'])

    # บันทึกผลลัพธ์ลงในไฟล์ CSV
    results_df.to_csv('ensemble_predictions.csv', index=False)
    logger.info("Ensemble predictions saved to ensemble_predictions.csv")

    # คำนวณ Returns โดยไม่มี Look-ahead Bias
    results_df['Returns'] = results_df.groupby('Ticker')['Actual_Price'].pct_change()
    results_df['Returns'] = results_df['Returns'].fillna(0)  # จัดการ NaN ในวันแรก

    # ปรับกลยุทธ์: ซื้อเมื่อ confidence score > 0.3
    results_df['Strategy_Returns'] = results_df['Returns'] * results_df['XGB_Predicted_Direction'].where(results_df['XGB_Predicted_Direction_Proba'] > 0.3, 0)
    results_df['Strategy_Returns'] = results_df['Strategy_Returns'].clip(lower=-0.05, upper=0.05)

    # คำนวณเมตริก
    total_return = results_df['Strategy_Returns'].sum()
    annualized_return = (1 + results_df['Strategy_Returns'].mean()) ** 252 - 1 if results_df['Strategy_Returns'].mean() is not np.nan else 0
    sharpe_ratio = results_df['Strategy_Returns'].mean() / results_df['Strategy_Returns'].std() * np.sqrt(252) if results_df['Strategy_Returns'].std() != 0 else 0
    cumulative_returns = (1 + results_df['Strategy_Returns']).cumprod()
    max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
    win_rate = len(results_df[results_df['Strategy_Returns'] > 0]) / len(results_df[results_df['Strategy_Returns'].notna()]) if len(results_df[results_df['Strategy_Returns'].notna()]) > 0 else 0

    logger.info(f"Total Strategy Return: {total_return:.4f}")
    logger.info(f"Annualized Return: {annualized_return:.4f}")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    logger.info(f"Maximum Drawdown: {max_drawdown:.4f}")
    logger.info(f"Win Rate: {win_rate:.4f}")
    logger.info("Final Predictions vs Actual Values (Top 10 rows):")
    logger.info(results_df.head(10).to_string())

except Exception as e:
    logger.error(f"Error in backtesting: {str(e)}")
    raise

## -----------------------------------
## 5. ฟังก์ชันสำหรับทำนายข้อมูลใหม่
## -----------------------------------
def predict_new_data(new_data, clf_model_path='xgb_classifier_model.pkl', reg_model_path='xgb_regressor_model.pkl', 
                    scaler_dir_path='scaler_dir.pkl', scaler_price_path='scaler_price.pkl'):
    try:
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_columns = ['Ticker', 'Date', 'Actual_Price', 'Predicted_Price_LSTM', 'Predicted_Price_GRU', 'Predicted_Dir_LSTM', 'Predicted_Dir_GRU']
        if not all(col in new_data.columns for col in required_columns):
            raise ValueError(f"Input data must contain columns: {required_columns}")

        # เพิ่ม High และ Low สำหรับ ATR
        new_data['High'] = new_data['Actual_Price'] * 1.01
        new_data['Low'] = new_data['Actual_Price'] * 0.99

        # โหลด scaler และโมเดล
        scaler_dir = joblib.load(scaler_dir_path)
        scaler_price = joblib.load(scaler_price_path)
        xgb_clf = joblib.load(clf_model_path)
        xgb_reg = joblib.load(reg_model_path)
        
        # เตรียมข้อมูลใหม่
        new_data = new_data.copy()
        new_data['Date'] = pd.to_datetime(new_data['Date'])
        new_data = new_data.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        # Feature Engineering
        new_data['Price_Diff'] = new_data['Predicted_Price_LSTM'] - new_data['Predicted_Price_GRU']
        new_data['Dir_Agreement'] = (new_data['Predicted_Dir_LSTM'] == new_data['Predicted_Dir_GRU']).astype(int)
        new_data['RSI'] = new_data.groupby('Ticker')['Actual_Price'].transform(lambda x: RSIIndicator(close=x, window=14).rsi() if len(x) >= 14 else pd.Series(np.nan, index=x.index))
        new_data['SMA_20'] = new_data.groupby('Ticker')['Actual_Price'].transform(lambda x: SMAIndicator(close=x, window=20).sma_indicator() if len(x) >= 20 else pd.Series(np.nan, index=x.index))
        new_data['MACD'] = new_data.groupby('Ticker')['Actual_Price'].transform(lambda x: MACD(close=x).macd() if len(x) >= 26 else pd.Series(np.nan, index=x.index))
        new_data['BB_High'] = new_data.groupby('Ticker')['Actual_Price'].transform(lambda x: BollingerBands(close=x, window=20).bollinger_hband() if len(x) >= 20 else pd.Series(np.nan, index=x.index))
        new_data['BB_Low'] = new_data.groupby('Ticker')['Actual_Price'].transform(lambda x: BollingerBands(close=x, window=20).bollinger_lband() if len(x) >= 20 else pd.Series(np.nan, index=x.index))
        new_data['ATR'] = new_data.groupby('Ticker').apply(lambda g: AverageTrueRange(high=g['High'], low=g['Low'], close=g['Actual_Price'], window=14).average_true_range() if len(g) >= 14 else pd.Series(np.nan, index=g.index)).reset_index(level=0, drop=True)
        new_data['Actual_Price_Normalized'] = new_data.groupby('Ticker')['Actual_Price'].transform(lambda x: (x - x.mean()) / x.std())
        
        # Impute missing values
        new_data[['RSI', 'SMA_20', 'MACD', 'BB_High', 'BB_Low', 'ATR']] = new_data.groupby('Ticker')[['RSI', 'SMA_20', 'MACD', 'BB_High', 'BB_Low', 'ATR']].ffill().bfill()
        imputer = SimpleImputer(strategy='mean')
        new_data[['Price_Diff', 'RSI', 'SMA_20', 'MACD', 'BB_High', 'BB_Low', 'ATR']] = imputer.fit_transform(new_data[['Price_Diff', 'RSI', 'SMA_20', 'MACD', 'BB_High', 'BB_Low', 'ATR']])
        
        X_new_dir = scaler_dir.transform(new_data[['Predicted_Dir_LSTM', 'Predicted_Dir_GRU', 'Dir_Agreement', 'RSI', 'SMA_20', 'MACD', 'BB_High', 'BB_Low', 'ATR']])
        X_new_price = scaler_price.transform(new_data[['Predicted_Price_LSTM', 'Predicted_Price_GRU', 'Price_Diff', 'RSI', 'SMA_20', 'MACD', 'BB_High', 'BB_Low', 'ATR', 'Actual_Price_Normalized']])
        
        # ทำนาย
        pred_dir = xgb_clf.predict(X_new_dir)
        pred_dir_proba = xgb_clf.predict_proba(X_new_dir)[:, 1]
        pred_price = xgb_reg.predict(X_new_price)
        
        # เพิ่ม Stop-loss/Take-profit
        predicted_returns = pred_dir * (pred_price - new_data['Actual_Price']) / new_data['Actual_Price']
        predicted_returns = predicted_returns.clip(lower=-0.05, upper=0.05)
        
        return pd.DataFrame({
            'Ticker': new_data['Ticker'],
            'Date': new_data['Date'],
            'Predicted_Direction': pred_dir,
            'Predicted_Direction_Proba': pred_dir_proba,
            'Predicted_Price': pred_price,
            'Predicted_Returns': predicted_returns
        })
    
    except Exception as e:
        logger.error(f"Error in predicting new data: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Model training and evaluation completed.")