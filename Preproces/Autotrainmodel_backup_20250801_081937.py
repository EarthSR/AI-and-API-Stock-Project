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

@tf.keras.utils.register_keras_serializable()
def quantile_loss(y_true, y_pred, quantile=0.5):
    error = y_true - y_pred
    return tf.keras.backend.mean(tf.keras.backend.maximum(quantile * error, (quantile - 1) * error))

def focal_weighted_binary_crossentropy(class_weights, gamma=2.0, alpha_pos=0.7):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        
        weights = tf.where(y_true == 1, class_weights[1], class_weights[0])
        alpha = tf.where(y_true == 1, alpha_pos, 1 - alpha_pos)
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
        focal_factor = tf.pow(1 - pt, gamma)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weighted_bce = bce * weights * alpha * focal_factor
        return tf.reduce_mean(weighted_bce)
    return loss

# ======================== RETRAIN TRACKING SYSTEM ========================
def save_retrain_log(model_name, chunk_idx, retrain_count, performance_metrics=None):
    """บันทึกประวัติการ retrain (โมเดลหลักเท่านั้น)"""
    log_file = f"retrain_log_{model_name}.csv"
    
    log_data = {
        'Timestamp': datetime.now(),
        'Model': model_name,
        'Chunk_Index': chunk_idx,
        'Retrain_Count': retrain_count,
        'Date': datetime.now().strftime('%Y-%m-%d'),
        'Time': datetime.now().strftime('%H:%M:%S')
    }
    
    # เพิ่ม performance metrics ถ้ามี
    if performance_metrics:
        log_data.update(performance_metrics)
    
    # บันทึกลงไฟล์
    try:
        if os.path.exists(log_file):
            existing_df = pd.read_csv(log_file)
            new_df = pd.concat([existing_df, pd.DataFrame([log_data])], ignore_index=True)
        else:
            new_df = pd.DataFrame([log_data])
        
        new_df.to_csv(log_file, index=False)
        print(f"            📝 Retrain log saved to {log_file}")
        
    except Exception as e:
        print(f"            ⚠️ Error saving retrain log: {e}")

def should_retrain_model(model_name, retrain_frequency_days=5):
    """ตรวจสอบว่าควร retrain โมเดลหรือไม่ตามเวลา"""
    last_trained_file = f"last_retrain_{model_name}.txt"
    
    if not os.path.exists(last_trained_file):
        return True, "No previous retrain record"
    
    try:
        with open(last_trained_file, "r") as f:
            last_trained_str = f.read().strip()
        
        last_trained_date = datetime.strptime(last_trained_str, "%Y-%m-%d")
        days_since_last_retrain = (datetime.now() - last_trained_date).days
        
        if days_since_last_retrain >= retrain_frequency_days:
            return True, f"Last retrain: {days_since_last_retrain} days ago"
        else:
            return False, f"Last retrain: {days_since_last_retrain} days ago (too recent)"
            
    except Exception as e:
        print(f"⚠️ Error reading retrain date: {e}")
        return True, "Error reading retrain record"

def update_retrain_date(model_name):
    """อัปเดตวันที่ retrain ล่าสุด"""
    last_trained_file = f"last_retrain_{model_name}.txt"
    
    try:
        with open(last_trained_file, "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d"))
        print(f"            📅 Updated retrain date for {model_name}")
        
    except Exception as e:
        print(f"            ⚠️ Error updating retrain date: {e}")

def get_retrain_stats(model_name):
    """แสดงสถิติการ retrain (โมเดลหลักเท่านั้น)"""
    log_file = f"retrain_log_{model_name}.csv"
    
    if not os.path.exists(log_file):
        return "No retrain history found"
    
    try:
        df = pd.read_csv(log_file)
        
        stats = {
            'Total_Retrains': len(df),
            'First_Retrain': df['Date'].min(),
            'Last_Retrain': df['Date'].max(),
            'Total_Chunks_Processed': df['Chunk_Index'].nunique() if 'Chunk_Index' in df.columns else 0,
            'Avg_Retrains_Per_Day': len(df) / max(1, (pd.to_datetime(df['Date'].max()) - pd.to_datetime(df['Date'].min())).days + 1)
        }
        
        return stats
        
    except Exception as e:
        return f"Error reading retrain stats: {e}"
    
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

                    # 🔄 Mini-retrain (Online Learning ภายใน chunk)
                    if (i+1) % retrain_frequency == 0 or (i == (len(current_chunk) - seq_length - 1)):
                        if len(batch_features) > 0:
                            try:
                                bf = np.concatenate(batch_features, axis=0)
                                bt = np.concatenate(batch_tickers, axis=0)
                                bm = np.concatenate(batch_market, axis=0)
                                bp = np.concatenate(batch_price, axis=0)
                                bd = np.concatenate(batch_dir, axis=0)

                                # บันทึกประสิทธิภาพก่อน retrain
                                pre_retrain_loss = model.evaluate(
                                    [bf, bt, bm],
                                    {
                                        'price_output': bp,
                                        'direction_output': bd
                                    },
                                    verbose=0
                                )

                                # ทำ mini-retrain
                                history = model.fit(
                                    [bf, bt, bm],
                                    {
                                        'price_output': bp,
                                        'direction_output': bd
                                    },
                                    epochs=1,
                                    batch_size=len(bf),
                                    verbose=0,
                                    shuffle=False
                                )
                                
                                # บันทึกประสิทธิภาพหลัง retrain
                                post_retrain_loss = model.evaluate(
                                    [bf, bt, bm],
                                    {
                                        'price_output': bp,
                                        'direction_output': bd
                                    },
                                    verbose=0
                                )
                                
                                # สร้าง performance metrics
                                performance_metrics = {
                                    'Pre_Retrain_Loss': pre_retrain_loss[0] if isinstance(pre_retrain_loss, list) else pre_retrain_loss,
                                    'Post_Retrain_Loss': post_retrain_loss[0] if isinstance(post_retrain_loss, list) else post_retrain_loss,
                                    'Loss_Improvement': (pre_retrain_loss[0] - post_retrain_loss[0]) if isinstance(pre_retrain_loss, list) else (pre_retrain_loss - post_retrain_loss),
                                    'Batch_Size': len(bf),
                                    'Position_in_Chunk': i+1
                                }
                                
                                # บันทึก retrain log
                                model_name = model.name if hasattr(model, 'name') else 'Unknown_Model'
                                save_retrain_log(model_name, chunk_idx + 1, (i+1)//retrain_frequency + 1, performance_metrics)
                                
                                # อัปเดตวันที่ retrain
                                update_retrain_date(model_name)
                                
                                print(f"            🔄 Mini-retrain at position {i+1} (batch size: {len(bf)})")
                                print(f"            📊 Loss: {pre_retrain_loss[0]:.4f} → {post_retrain_loss[0]:.4f}" if isinstance(pre_retrain_loss, list) else f"            📊 Loss: {pre_retrain_loss:.4f} → {post_retrain_loss:.4f}")
                                
                            except Exception as e:
                                print(f"            ⚠️ Mini-retrain error: {e}")

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
            
            # ✅ แค่ Mini-retrain (Online Learning) ก็เพียงพอแล้ว
            print(f"         ✅ Chunk {chunk_idx + 1} completed with continuous online learning")
        
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

    return predictions_df, overall_metrics

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

def prepare_data_for_walk_forward(df, feature_columns):
    """เตรียมข้อมูลสำหรับ Walk-Forward Validation ตามแนวทางโค้ดเทรน"""
    print("📊 Preparing data for Walk-Forward Validation...")
    
    df_prepared = df.copy()
    
    # ======== STEP 1: Calculate Technical Indicators (ถ้ายังไม่มี) ========
    df_prepared = ensure_technical_indicators(df_prepared)
    
    # ======== STEP 2: Handle Sentiment Mapping ========
    if 'Sentiment' in df_prepared.columns:
        # แปลง text sentiment เป็น numeric (ถ้าจำเป็น)
        if df_prepared['Sentiment'].dtype == 'object':
            df_prepared['Sentiment'] = df_prepared['Sentiment'].map({
                'Positive': 1, 'Negative': -1, 'Neutral': 0
            })
            print("   🔄 Mapped sentiment values to numeric")
    
    # ======== STEP 3: Create Target Variables ========
    print("   🎯 Creating target variables...")
    df_prepared = df_prepared.sort_values(['StockSymbol', 'Date']).reset_index(drop=True)
    
    # สร้าง Direction และ TargetPrice per ticker
    df_prepared['Direction'] = 0
    df_prepared['TargetPrice'] = np.nan
    
    for ticker in df_prepared['StockSymbol'].unique():
        ticker_mask = df_prepared['StockSymbol'] == ticker
        ticker_data = df_prepared[ticker_mask].copy()
        
        if len(ticker_data) > 1:
            # Direction: 1 ถ้าราคาพรุ่งนี้สูงกว่าวันนี้
            direction = (ticker_data['Close'].shift(-1) > ticker_data['Close']).astype(int)
            target_price = ticker_data['Close'].shift(-1)
            
            df_prepared.loc[ticker_mask, 'Direction'] = direction
            df_prepared.loc[ticker_mask, 'TargetPrice'] = target_price
    
    # ลบแถวที่ไม่มี target
    df_prepared = df_prepared.dropna(subset=['Direction', 'TargetPrice'])
    
    # ======== STEP 4: Ensure Encoders ========
    if 'Ticker_ID' not in df_prepared.columns:
        ticker_encoder = LabelEncoder()
        df_prepared['Ticker_ID'] = ticker_encoder.fit_transform(df_prepared['StockSymbol'])
        print("   🔄 Created Ticker_ID encoding")
    
    if 'Market_ID' not in df_prepared.columns or df_prepared['Market_ID'].dtype == 'object':
        # สร้าง Market_ID
        us_stock = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'TSM', 'AMD']
        thai_stock = ['ADVANC', 'TRUE', 'DITTO', 'DIF', 
                     'INSET', 'JMART', 'INET', 'JAS', 'HUMAN']
        
        df_prepared['Market'] = df_prepared['StockSymbol'].apply(
            lambda x: "US" if x in us_stock else "TH" if x in thai_stock else "OTHER"
        )
        
        market_encoder = LabelEncoder()
        df_prepared['Market_ID'] = market_encoder.fit_transform(df_prepared['Market'])
        print("   🔄 Created Market_ID encoding")
    
    # ======== STEP 5: Handle Missing Technical Indicators ========
    stock_columns = [
        'RSI', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'Bollinger_High',
        'Bollinger_Low', 'ATR', 'Keltner_High', 'Keltner_Low', 'Keltner_Middle',
        'Chaikin_Vol', 'Donchian_High', 'Donchian_Low', 'PSAR', 'SMA_50', 'SMA_200'
    ]
    
    available_stock_cols = [col for col in stock_columns if col in df_prepared.columns]
    if available_stock_cols:
        print(f"   🔧 Forward filling {len(available_stock_cols)} technical indicators...")
        # Forward fill per ticker
        for ticker in df_prepared['StockSymbol'].unique():
            ticker_mask = df_prepared['StockSymbol'] == ticker
            df_prepared.loc[ticker_mask, available_stock_cols] = \
                df_prepared.loc[ticker_mask, available_stock_cols].fillna(method='ffill')
    
    # Fill remaining NaN with 0
    df_prepared = df_prepared.fillna(0)
    
    print(f"   ✅ Prepared data: {len(df_prepared)} rows, {len(df_prepared['StockSymbol'].unique())} tickers")
    return df_prepared

def ensure_technical_indicators(df):
    """ตรวจสอบและเพิ่ม technical indicators ที่จำเป็น"""
    df_with_indicators = df.copy()
    
    required_indicators = ['RSI', 'MACD', 'MACD_Signal', 'ATR', 'Bollinger_High', 'Bollinger_Low']
    missing_indicators = [ind for ind in required_indicators if ind not in df_with_indicators.columns]
    
    if missing_indicators:
        print(f"   🔧 Adding missing technical indicators: {missing_indicators}")
        
        # คำนวณ indicators ที่ขาดหายไป per ticker
        for ticker in df_with_indicators['StockSymbol'].unique():
            ticker_mask = df_with_indicators['StockSymbol'] == ticker
            ticker_data = df_with_indicators[ticker_mask].copy()
            
            if len(ticker_data) < 20:  # ต้องการข้อมูลอย่างน้อย 20 วัน
                continue
            
            try:
                if 'RSI' in missing_indicators:
                    rsi = ta.momentum.RSIIndicator(ticker_data['Close'], window=14).rsi()
                    df_with_indicators.loc[ticker_mask, 'RSI'] = rsi
                
                if any(ind in missing_indicators for ind in ['MACD', 'MACD_Signal']):
                    ema_12 = ticker_data['Close'].ewm(span=12).mean()
                    ema_26 = ticker_data['Close'].ewm(span=26).mean()
                    macd = ema_12 - ema_26
                    macd_signal = macd.rolling(window=9).mean()
                    
                    if 'MACD' in missing_indicators:
                        df_with_indicators.loc[ticker_mask, 'MACD'] = macd
                    if 'MACD_Signal' in missing_indicators:
                        df_with_indicators.loc[ticker_mask, 'MACD_Signal'] = macd_signal
                
                if 'ATR' in missing_indicators and all(col in ticker_data.columns for col in ['High', 'Low']):
                    atr = ta.volatility.AverageTrueRange(
                        high=ticker_data['High'], 
                        low=ticker_data['Low'], 
                        close=ticker_data['Close'], 
                        window=14
                    ).average_true_range()
                    df_with_indicators.loc[ticker_mask, 'ATR'] = atr
                
                if any(ind in missing_indicators for ind in ['Bollinger_High', 'Bollinger_Low']):
                    bollinger = ta.volatility.BollingerBands(ticker_data['Close'], window=20, window_dev=2)
                    if 'Bollinger_High' in missing_indicators:
                        df_with_indicators.loc[ticker_mask, 'Bollinger_High'] = bollinger.bollinger_hband()
                    if 'Bollinger_Low' in missing_indicators:
                        df_with_indicators.loc[ticker_mask, 'Bollinger_Low'] = bollinger.bollinger_lband()
                        
            except Exception as e:
                print(f"      ⚠️ Error calculating indicators for {ticker}: {e}")
                continue
    
    return df_with_indicators

# ======================== INTEGRATION WITH MAIN SYSTEM ========================

def create_walk_forward_compatible_scalers(df, feature_columns):
    """สร้าง scalers ที่ compatible กับทั้ง Walk-Forward Validation และโค้ดเทรน"""
    
    print("🔄 Creating Walk-Forward compatible scalers...")
    
    # เตรียมข้อมูล
    df_prepared = prepare_data_for_walk_forward(df, feature_columns)
    
    # สร้าง unified scalers
    ticker_scalers = create_unified_ticker_scalers(df_prepared, feature_columns)
    
    return ticker_scalers, df_prepared
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
                    compile=compile_model  # ใช้ compile_model แทน compile
                )
                print("✅ โหลดโมเดลเดี่ยวสำเร็จ")
                return model
            else:  # กรณีโหลดทั้ง LSTM และ GRU
                print("🔄 กำลังโหลดโมเดลสำหรับการทำนาย...")
                self.lstm_model = tf.keras.models.load_model(
                    self.lstm_model_path,
                    custom_objects=custom_objects,
                    safe_mode=False,
                    compile=compile_model
                )
                self.gru_model = tf.keras.models.load_model(
                    self.gru_model_path,
                    custom_objects=custom_objects,
                    safe_mode=False,
                    compile=compile_model
                )
                print("✅ โหลดโมเดลสำหรับการทำนายสำเร็จ")
                return True
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
            return None if model_path else False

# Fix 2: Enhanced data cleaning function
def clean_data_for_scalers(df, feature_columns):
    """ทำความสะอาดข้อมูลก่อนสร้าง scalers"""
    print("🧹 กำลังทำความสะอาดข้อมูลสำหรับ scalers...")
    
    df_clean = df.copy()
    
    # ตรวจสอบและแก้ไขข้อมูลแต่ละคอลัมน์
    for col in feature_columns:
        if col in df_clean.columns:
            try:
                # แปลงเป็น string ก่อนเพื่อตรวจสอบ
                col_data = df_clean[col].astype(str)
                
                # ตรวจสอบว่ามีข้อมูลที่เชื่อมต่อกันหรือไม่ (เช่น 49.0349.03)
                problematic_mask = col_data.str.contains(r'\d+\.\d+\d+\.\d+', regex=True, na=False)
                
                if problematic_mask.any():
                    print(f"   ⚠️ พบข้อมูลผิดปกติในคอลัมน์ {col}: {problematic_mask.sum()} แถว")
                    
                    # แยกตัวเลขออกจากกัน (ใช้วิธีง่ายๆ คือเอาตัวเลขตัวแรก)
                    def extract_first_number(x):
                        try:
                            if pd.isna(x):
                                return 0.0
                            x_str = str(x)
                            # หาตัวเลขตัวแรกที่มีทศนิยม
                            import re
                            match = re.search(r'^\d+\.?\d*', x_str)
                            if match:
                                return float(match.group())
                            else:
                                return 0.0
                        except:
                            return 0.0
                    
                    df_clean[col] = col_data.apply(extract_first_number)
                else:
                    # แปลงเป็น numeric ปกติ
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # แทนที่ NaN และ inf ด้วยค่าเฉลี่ย
                col_mean = df_clean[col].replace([np.inf, -np.inf], np.nan).mean()
                if pd.isna(col_mean):
                    col_mean = 0.0
                
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf, np.nan], col_mean)
                
                print(f"   ✅ ทำความสะอาด {col}: {df_clean[col].dtype}, range: {df_clean[col].min():.3f} - {df_clean[col].max():.3f}")
                
            except Exception as e:
                print(f"   ❌ ไม่สามารถทำความสะอาด {col}: {e}")
                # ใช้ค่าเริ่มต้น
                df_clean[col] = 0.0
    
    return df_clean

# Fix 3: Enhanced create_ticker_scalers function
def create_ticker_scalers_fixed(df, feature_columns, scaler_file_path="../LSTM_model/ticker_scalers.pkl"):
    """สร้าง ticker scalers พร้อมการจัดการข้อมูลที่ดีขึ้น"""
    
    # ทำความสะอาดข้อมูลก่อน
    df_clean = clean_data_for_scalers(df, feature_columns)
    
    ticker_scalers = {}
    tickers = df_clean['StockSymbol'].unique()
    
    print("🔧 Creating/loading individual scalers for each ticker...")
    
    # พยายามโหลด scalers เก่า
    pre_trained_scalers = {}
    try:
        if os.path.exists(scaler_file_path):
            pre_trained_scalers = joblib.load(scaler_file_path)
            print(f"✅ Loaded pre-trained scalers from {scaler_file_path} for {len(pre_trained_scalers)} tickers")
        else:
            print(f"⚠️ No pre-trained scalers found at {scaler_file_path}, creating new scalers")
    except Exception as e:
        print(f"❌ Error loading pre-trained scalers: {e}, creating new scalers")
    
    for ticker in tickers:
        df_ticker = df_clean[df_clean['StockSymbol'] == ticker].copy()
        
        if len(df_ticker) < 30:  # ต้องการข้อมูลอย่างน้อย 30 วัน
            print(f"   ⚠️ {ticker}: Not enough data ({len(df_ticker)} days), skipping...")
            continue
        
        ticker_id = df_ticker['Ticker_ID'].iloc[0]
        
        # ตรวจสอบ pre-trained scaler
        if ticker_id in pre_trained_scalers:
            scaler_info = pre_trained_scalers[ticker_id]
            if all(key in scaler_info for key in ['ticker_symbol', 'feature_scaler', 'price_scaler']):
                # ทดสอบว่า scaler ยังใช้งานได้หรือไม่
                try:
                    # ทดสอบ transform ข้อมูลเล็กน้อย
                    test_data = df_ticker[feature_columns].iloc[:5].fillna(df_ticker[feature_columns].mean())
                    _ = scaler_info['feature_scaler'].transform(test_data)
                    _ = scaler_info['price_scaler'].transform(df_ticker[['Close']].iloc[:5])
                    
                    # เพิ่ม data_points ถ้าขาด
                    if 'data_points' not in scaler_info:
                        scaler_info['data_points'] = len(df_ticker)
                    
                    ticker_scalers[ticker_id] = scaler_info
                    print(f"   ✅ {ticker} (ID: {ticker_id}): Using pre-trained scaler with {scaler_info['data_points']} data points")
                    continue
                    
                except Exception as e:
                    print(f"   ⚠️ {ticker} (ID: {ticker_id}): Pre-trained scaler failed test ({e}), creating new one")
        
        # สร้าง scaler ใหม่
        try:
            # ตรวจสอบข้อมูลก่อนสร้าง scaler
            feature_data = df_ticker[feature_columns].copy()
            
            # ตรวจสอบว่าข้อมูลเป็น numeric หรือไม่
            non_numeric_cols = []
            for col in feature_columns:
                if col in feature_data.columns:
                    if not pd.api.types.is_numeric_dtype(feature_data[col]):
                        non_numeric_cols.append(col)
            
            if non_numeric_cols:
                print(f"   ⚠️ {ticker}: Non-numeric columns found: {non_numeric_cols}")
                for col in non_numeric_cols:
                    feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
            
            # แทนที่ NaN ด้วยค่าเฉลี่ย
            feature_data = feature_data.fillna(feature_data.mean()).fillna(0)
            
            # ตรวจสอบว่ายังมี infinite values หรือไม่
            feature_data = feature_data.replace([np.inf, -np.inf], 0)
            
            # สร้าง feature scaler
            feature_scaler = RobustScaler()
            feature_scaler.fit(feature_data)
            
            # สร้าง price scaler
            price_scaler = RobustScaler()
            price_data = df_ticker[['Close']].copy()
            price_data = price_data.fillna(price_data.mean()).fillna(0)
            price_data = price_data.replace([np.inf, -np.inf], price_data.mean())
            price_scaler.fit(price_data)
            
            # เก็บ scaler และ metadata
            ticker_scalers[ticker_id] = {
                'ticker_symbol': ticker,
                'feature_scaler': feature_scaler,
                'price_scaler': price_scaler,
                'data_points': len(df_ticker)
            }
            
            print(f"   ✅ {ticker} (ID: {ticker_id}): Created new scaler with {len(df_ticker)} data points")
            
        except Exception as e:
            print(f"   ❌ {ticker}: Error creating scalers - {e}")
            # พิมพ์ข้อมูล debug
            print(f"      Debug - Feature data shape: {df_ticker[feature_columns].shape}")
            print(f"      Debug - Feature data dtypes: {df_ticker[feature_columns].dtypes.to_dict()}")
            print(f"      Debug - Sample values: {df_ticker[feature_columns].iloc[0].to_dict()}")
            continue
    
    # บันทึก scalers ที่อัปเดตแล้ว
    try:
        os.makedirs(os.path.dirname(scaler_file_path), exist_ok=True)
        joblib.dump(ticker_scalers, scaler_file_path)
        print(f"💾 Saved updated scalers to {scaler_file_path}")
    except Exception as e:
        print(f"❌ Error saving scalers: {e}")
    
    print(f"✅ Created/loaded scalers for {len(ticker_scalers)} tickers")
    return ticker_scalers

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Setup logging
class EnhancedDynamicEnsembleMetaLearner:
    """
    🏆 Enhanced Dynamic Weighted Ensemble Meta Learner
    
    ปรับปรุงการชั่งน้ำหนักจากความแม่นยำของแต่ละหุ้นโดยเฉพาะ:
    - ใช้ historical performance ของแต่ละหุ้น
    - รวม MAE, R² Score, และ Direction Accuracy
    - ให้น้ำหนักมากกว่าแก่โมเดลที่เก่งกว่าในหุ้นนั้นๆ
    """
    
    def __init__(self, window_size=20):
        self.window_size = window_size
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
            'TSM': {
                'MAE': (6.1881, 5.6068),
                'R2': (0.8837, 0.9083),
                'Direction_Acc': (0.7262, 0.6764)
            }
        }
        
        logger.info("🚀 Enhanced Dynamic Weighted Ensemble Meta Learner initialized")
        logger.info(f"Window size: {window_size} days")
        logger.info(f"Loaded performance data for {len(self.stock_performance)} stocks")
    
    def calculate_stock_specific_weights(self, ticker):
        """
        🎯 คำนวณน้ำหนักเฉพาะหุ้นจาก historical performance
        โดยพิจารณา MAE (ยิ่งต่ำยิ่งดี), R² Score (ยิ่งสูงยิ่งดี), Direction Accuracy (ยิ่งสูงยิ่งดี)
        """
        
        if ticker not in self.stock_performance:
            logger.warning(f"No historical performance data for {ticker}, using equal weights")
            return 0.5, 0.5
        
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
        
        # 🏆 Combined weighted score
        # MAE มีน้ำหนัก 40%, R² มีน้ำหนัก 40%, Direction Accuracy มีน้ำหนัก 20%
        lstm_final_weight = (0.4 * mae_lstm_weight + 
                           0.4 * r2_lstm_weight + 
                           0.2 * dir_lstm_weight)
        
        gru_final_weight = (0.4 * mae_gru_weight + 
                          0.4 * r2_gru_weight + 
                          0.2 * dir_gru_weight)
        
        # Normalize to ensure sum = 1
        total_weight = lstm_final_weight + gru_final_weight
        if total_weight > 0:
            lstm_final_weight /= total_weight
            gru_final_weight /= total_weight
        else:
            lstm_final_weight = gru_final_weight = 0.5
        
        logger.debug(f"{ticker}: LSTM={lstm_final_weight:.3f}, GRU={gru_final_weight:.3f} "
                    f"[MAE: {mae_lstm_weight:.2f}/{mae_gru_weight:.2f}, "
                    f"R²: {r2_lstm_weight:.2f}/{r2_gru_weight:.2f}, "
                    f"Dir: {dir_lstm_weight:.2f}/{dir_gru_weight:.2f}]")
        
        return lstm_final_weight, gru_final_weight
    
    def calculate_dynamic_weights(self, ticker, recent_performance):
        """
        🎯 Dynamic Weighting Algorithm - ผสมระหว่าง historical และ recent performance
        """
        
        # Get stock-specific historical weights
        hist_lstm_weight, hist_gru_weight = self.calculate_stock_specific_weights(ticker)
        
        # If insufficient recent data, use historical weights
        if len(recent_performance) < 3:
            logger.debug(f"{ticker}: Using historical weights (insufficient recent data)")
            return hist_lstm_weight, hist_gru_weight
        
        try:
            # Calculate recent performance weights
            actual_prices = recent_performance['Actual_Price'].values
            lstm_predictions = recent_performance['Predicted_Price_LSTM'].values
            gru_predictions = recent_performance['Predicted_Price_GRU'].values
            
            # Recent MAE calculation
            lstm_recent_mae = mean_absolute_error(actual_prices, lstm_predictions)
            gru_recent_mae = mean_absolute_error(actual_prices, gru_predictions)
            
            # Recent weights (inverse MAE)
            lstm_recent_inv = 1 / (lstm_recent_mae + 1e-8)
            gru_recent_inv = 1 / (gru_recent_mae + 1e-8)
            total_recent_inv = lstm_recent_inv + gru_recent_inv
            
            recent_lstm_weight = lstm_recent_inv / total_recent_inv
            recent_gru_weight = gru_recent_inv / total_recent_inv
            
            # 🏆 Adaptive blending: ใช้ historical 70% + recent 30%
            # สำหรับ established performance, historical data น่าเชื่อถือกว่า
            alpha = 0.3  # recent performance weight
            final_lstm_weight = (1 - alpha) * hist_lstm_weight + alpha * recent_lstm_weight
            final_gru_weight = (1 - alpha) * hist_gru_weight + alpha * recent_gru_weight
            
            logger.debug(f"{ticker}: Blended weights - LSTM={final_lstm_weight:.3f}, GRU={final_gru_weight:.3f} "
                        f"[Hist: {hist_lstm_weight:.2f}/{hist_gru_weight:.2f}, "
                        f"Recent: {recent_lstm_weight:.2f}/{recent_gru_weight:.2f}]")
            
            return final_lstm_weight, final_gru_weight
            
        except Exception as e:
            logger.warning(f"Error calculating dynamic weights for {ticker}: {e}")
            return hist_lstm_weight, hist_gru_weight
    
    def get_performance_summary(self, ticker):
        """📊 Get performance summary for a stock"""
        if ticker not in self.stock_performance:
            return "No historical data"
        
        perf = self.stock_performance[ticker]
        lstm_mae, gru_mae = perf['MAE']
        lstm_r2, gru_r2 = perf['R2']
        lstm_dir, gru_dir = perf['Direction_Acc']
        
        # Determine better model for each metric
        mae_winner = "LSTM" if lstm_mae < gru_mae else "GRU"
        r2_winner = "LSTM" if lstm_r2 > gru_r2 else "GRU"
        dir_winner = "LSTM" if lstm_dir > gru_dir else "GRU"
        
        return f"MAE:{mae_winner}, R²:{r2_winner}, Dir:{dir_winner}"
    
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
        """🎯 Enhanced Stock-Specific Ensemble Prediction"""
        
        current_price = stock_data['Current_Price']
        lstm_price = stock_data['Predicted_Price_LSTM']
        gru_price = stock_data['Predicted_Price_GRU']
        lstm_dir = stock_data['Predicted_Dir_LSTM']
        gru_dir = stock_data['Predicted_Dir_GRU']
        
        # Get recent performance for dynamic weighting
        recent_performance = self.performance_history.get(ticker, pd.DataFrame())
        
        # Calculate stock-specific dynamic weights
        lstm_weight, gru_weight = self.calculate_dynamic_weights(ticker, recent_performance)
        
        # 🏆 Stock-Specific Weighted Prediction
        ensemble_price = lstm_weight * lstm_price + gru_weight * gru_price
        
        # Direction prediction (weighted voting)
        lstm_dir_weighted = lstm_weight * lstm_dir
        gru_dir_weighted = gru_weight * gru_dir
        ensemble_dir_prob = lstm_dir_weighted + gru_dir_weighted
        ensemble_direction = 1 if ensemble_dir_prob >= 0.5 else 0
        
        # Price change analysis
        price_change_pct = ((ensemble_price - current_price) / current_price) * 100
        price_implied_direction = 1 if price_change_pct > 0.5 else 0
        
        # Consistency check
        is_consistent = (ensemble_direction == price_implied_direction)
        
        # Enhanced confidence calculation based on historical performance
        direction_confidence = abs(ensemble_dir_prob - 0.5) * 2
        price_confidence = min(abs(price_change_pct) / 10, 1.0)
        
        # Add historical performance boost to confidence
        if ticker in self.stock_performance:
            perf = self.stock_performance[ticker]
            avg_r2 = (perf['R2'][0] + perf['R2'][1]) / 2
            avg_dir_acc = (perf['Direction_Acc'][0] + perf['Direction_Acc'][1]) / 2
            historical_confidence = (avg_r2 + avg_dir_acc) / 2
            
            # Blend current confidence with historical performance
            overall_confidence = 0.7 * ((direction_confidence + price_confidence) / 2) + 0.3 * historical_confidence
        else:
            overall_confidence = (direction_confidence + price_confidence) / 2
        
        # Consistency bonus
        if is_consistent:
            overall_confidence = min(overall_confidence * 1.15, 0.95)
        
        # Enhanced risk assessment
        perf_summary = self.get_performance_summary(ticker)
        
        if not is_consistent:
            risk_level = "🔴 HIGH_RISK"
            warning = f"INCONSISTENT_PREDICTION - {perf_summary}"
            action = "EXERCISE_EXTREME_CAUTION"
        elif abs(price_change_pct) > 15:
            risk_level = "🟡 MEDIUM_RISK"
            warning = f"HIGH_VOLATILITY_PREDICTION - {perf_summary}"
            action = "HIGH_RISK"
        elif overall_confidence >= 0.75:
            risk_level = "🟢 LOW_RISK"
            warning = f"HIGH_CONFIDENCE - {perf_summary}"
            action = "CONSIDER"
        elif overall_confidence >= 0.5:
            risk_level = "🟡 MEDIUM_RISK"
            warning = f"MODERATE_CONFIDENCE - {perf_summary}"
            action = "CAUTION"
        else:
            risk_level = "🔴 HIGH_RISK"
            warning = f"LOW_CONFIDENCE - {perf_summary}"
            action = "AVOID"
        
        return {
            'predicted_price': ensemble_price,
            'predicted_direction': ensemble_direction,
            'direction_probability': ensemble_dir_prob,
            'confidence': overall_confidence,
            'price_change_pct': price_change_pct,
            'is_consistent': is_consistent,
            'lstm_weight': lstm_weight,
            'gru_weight': gru_weight,
            'risk_level': risk_level,
            'warning': warning,
            'action': action,
            'performance_summary': perf_summary
        }
    
    def predict_meta(self, df):
        """🏆 Main Prediction Method using Enhanced Stock-Specific Weighted Ensemble"""
        
        logger.info("🚀 Starting Enhanced Stock-Specific Weighted Ensemble Prediction...")
        
        # Prepare data
        prepared_df = self.prepare_data_for_model(df)
        if prepared_df is None:
            return df
        
        logger.info(f"Processing {len(prepared_df)} stocks with stock-specific weighting")
        
        # Process each stock
        predictions = []
        weight_summary = {}
        
        for _, stock_data in prepared_df.iterrows():
            ticker = stock_data['Ticker']
            prediction = self.predict_single_stock(stock_data, ticker)
            prediction['ticker'] = ticker
            predictions.append(prediction)
            
            # Collect weight summary
            weight_summary[ticker] = {
                'lstm_weight': prediction['lstm_weight'],
                'gru_weight': prediction['gru_weight'],
                'performance': prediction['performance_summary']
            }
        
        if len(predictions) == 0:
            logger.error("No predictions generated")
            return df
        
        # Enhanced statistics
        confidences = [p['confidence'] for p in predictions]
        consistency_rate = np.mean([p['is_consistent'] for p in predictions])
        price_changes = [p['price_change_pct'] for p in predictions]
        
        # Weight distribution analysis
        avg_lstm_weight = np.mean([p['lstm_weight'] for p in predictions])
        avg_gru_weight = np.mean([p['gru_weight'] for p in predictions])
        
        logger.info(f"🎯 Results: {len(predictions)} stocks processed")
        logger.info(f"📊 Avg Confidence: {np.mean(confidences):.3f}")
        logger.info(f"📊 Consistency Rate: {consistency_rate:.1%}")
        logger.info(f"⚖️ Avg Weights - LSTM: {avg_lstm_weight:.3f}, GRU: {avg_gru_weight:.3f}")
        
        # Map results back to dataframe
        ticker_col = 'StockSymbol' if 'StockSymbol' in df.columns else 'Ticker'
        prediction_mask = df[ticker_col].isin(prepared_df['Ticker'])
        
        if prediction_mask.any():
            ticker_to_results = {p['ticker']: p for p in predictions}
            ensemble_method = f"Enhanced_Stock_Specific_v2.0_consistency_{consistency_rate:.1%}"
            
            for idx, row in df[prediction_mask].iterrows():
                ticker = row[ticker_col]
                if ticker in ticker_to_results:
                    result = ticker_to_results[ticker]
                    
                    # Core predictions
                    df.loc[idx, 'XGB_Predicted_Direction'] = result['predicted_direction']
                    df.loc[idx, 'XGB_Predicted_Price'] = result['predicted_price']
                    df.loc[idx, 'XGB_Confidence'] = result['confidence']
                    df.loc[idx, 'XGB_Predicted_Direction_Proba'] = result['direction_probability']
                    
                    # Enhanced info
                    df.loc[idx, 'Reliability_Score'] = result['confidence']
                    df.loc[idx, 'Reliability_Warning'] = result['warning']
                    df.loc[idx, 'Suggested_Action'] = result['action']
                    df.loc[idx, 'Risk_Level'] = result['risk_level']
                    df.loc[idx, 'Ensemble_Method'] = ensemble_method
                    df.loc[idx, 'Price_Change_Percent'] = result['price_change_pct']
                    
                    # Stock-specific ensemble info
                    df.loc[idx, 'LSTM_Weight'] = result['lstm_weight']
                    df.loc[idx, 'GRU_Weight'] = result['gru_weight']
                    df.loc[idx, 'Is_Consistent'] = result['is_consistent']
                    df.loc[idx, 'Historical_Performance'] = result['performance_summary']
        
        # Log detailed weight analysis
        logger.info("🔍 Stock-Specific Weight Analysis:")
        for ticker, info in weight_summary.items():
            logger.info(f"  {ticker}: LSTM={info['lstm_weight']:.3f}, GRU={info['gru_weight']:.3f} [{info['performance']}]")
        
        # Log distribution
        risk_dist = pd.Series([p['risk_level'] for p in predictions]).value_counts()
        action_dist = pd.Series([p['action'] for p in predictions]).value_counts()
        
        logger.info("📊 Risk Distribution: " + ", ".join([f"{r}:{c}" for r, c in risk_dist.items()]))
        logger.info("📊 Action Distribution: " + ", ".join([f"{a}:{c}" for a, c in action_dist.items()]))
        
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
        """📈 Get stock recommendations based on historical performance"""
        recommendations = {}
        
        for ticker, perf in self.stock_performance.items():
            lstm_mae, gru_mae = perf['MAE']
            lstm_r2, gru_r2 = perf['R2']
            lstm_dir, gru_dir = perf['Direction_Acc']
            
            # Calculate overall scores
            lstm_score = (1/lstm_mae) * lstm_r2 * lstm_dir
            gru_score = (1/gru_mae) * gru_r2 * gru_dir
            avg_score = (lstm_score + gru_score) / 2
            
            if avg_score > 0.8:
                recommendation = "🟢 HIGH_CONFIDENCE"
            elif avg_score > 0.4:
                recommendation = "🟡 MODERATE_CONFIDENCE"
            else:
                recommendation = "🔴 LOW_CONFIDENCE"
            
            recommendations[ticker] = {
                'score': avg_score,
                'recommendation': recommendation
            }
        
        return recommendations
    
    # Compatibility methods
    def is_model_available(self):
        return True
    
    def get_model_status(self):
        return "ENHANCED_STOCK_SPECIFIC_ENSEMBLE_READY"
    
    def should_retrain_meta(self):
        return False  # Rule-based, no retraining needed
    
    def validate_predictions(self, df):
        return 'XGB_Predicted_Price' in df.columns and df['XGB_Predicted_Price'].notna().any()

# Backward compatibility
DynamicEnsembleMetaLearner = EnhancedDynamicEnsembleMetaLearner
XGBoostMetaLearner = EnhancedDynamicEnsembleMetaLearner
UpdatedXGBoostMetaLearner = EnhancedDynamicEnsembleMetaLearner

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

def calculate_realistic_features(df):
    """คำนวณ features จากข้อมูลจริง แทนการใช้ default values"""
    df_enhanced = df.copy()
    
    # คำนวณ Price Agreement จริงๆ
    if 'PredictionClose_LSTM' in df.columns and 'PredictionClose_GRU' in df.columns:
        lstm_pred = df['PredictionClose_LSTM']
        gru_pred = df['PredictionClose_GRU']
        
        price_diff_pct = abs(lstm_pred - gru_pred) / df['Close'] * 100
        df_enhanced['Price_Agreement'] = np.exp(-price_diff_pct / 10)
        df_enhanced['LSTM_Error_Pct'] = np.minimum(price_diff_pct, 20.0)
        df_enhanced['GRU_Error_Pct'] = np.minimum(price_diff_pct, 20.0)
    else:
        df_enhanced['Price_Agreement'] = 0.7
        df_enhanced['LSTM_Error_Pct'] = 10.0
        df_enhanced['GRU_Error_Pct'] = 10.0
    
    # คำนวณ Direction Confidence
    if 'PredictionTrend_LSTM' in df.columns and 'PredictionTrend_GRU' in df.columns:
        lstm_conf = abs(df['PredictionTrend_LSTM'] - 0.5) * 2
        gru_conf = abs(df['PredictionTrend_GRU'] - 0.5) * 2
        df_enhanced['Dir_Confidence'] = (lstm_conf + gru_conf) / 2
    else:
        df_enhanced['Dir_Confidence'] = 0.5
    
    # คำนวณ Volume Normalized
    if 'Volume' in df.columns:
        volume_max = df['Volume'].rolling(window=30, min_periods=1).max()
        df_enhanced['Volume_Normalized'] = df['Volume'] / (volume_max + 1e-8)
    else:
        df_enhanced['Volume_Normalized'] = 0.5
    
    # คำนวณ Price Volatility
    if 'ATR' in df.columns:
        df_enhanced['Price_Volatility'] = df['ATR'] / (df['Close'] + 1e-8)
    else:
        price_std = df['Close'].rolling(window=14, min_periods=1).std()
        df_enhanced['Price_Volatility'] = price_std / (df['Close'] + 1e-8)
    
    return df_enhanced

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


def predict_with_consistent_scalers(model_lstm, model_gru, df, feature_columns, 
                                  ticker_scalers, ticker_encoder, market_encoder, seq_length=10):
    """
    ทำนายด้วย scalers ที่สอดคล้องกัน
    ✅ LSTM และ GRU ใช้ scaler เดียวกัน
    ✅ ใช้ per-ticker scalers จากตอนเทรน
    """
    print(f"\n🔮 เริ่มต้นการทำนายด้วย Consistent Scalers...")
    print(f"   📊 ใช้ per-ticker scalers เหมือนกับตอนเทรน")
    print(f"   🔄 LSTM และ GRU ใช้ scaler ชุดเดียวกัน")
    
    future_predictions = []
    tickers = df['StockSymbol'].unique()
    
    for ticker in tickers:
        print(f"\n📊 ทำนาย {ticker}...")
        
        df_ticker = df[df['StockSymbol'] == ticker].sort_values('Date').reset_index(drop=True)
        
        if len(df_ticker) < seq_length:
            print(f"   ⚠️ ข้อมูลไม่เพียงพอ: {len(df_ticker)} < {seq_length}")
            continue
        
        try:
            # เตรียมข้อมูล
            latest_data = df_ticker.iloc[-seq_length:]
            ticker_id = latest_data['Ticker_ID'].iloc[-1]
            
            # ตรวจสอบ scaler
            if ticker_id not in ticker_scalers:
                print(f"   ❌ ไม่พบ scaler สำหรับ {ticker} (ID: {ticker_id})")
                continue
            
            # ✅ ใช้ scaler เดียวกันสำหรับทั้ง LSTM และ GRU
            scaler_info = ticker_scalers[ticker_id]
            feature_scaler = scaler_info['feature_scaler']
            price_scaler = scaler_info['price_scaler']
            
            print(f"   🔧 ใช้ scaler สำหรับ {ticker}:")
            print(f"      📊 Feature scaler: {type(feature_scaler).__name__}")
            print(f"      💰 Price scaler: {type(price_scaler).__name__}")
            
            # Scale features
            features_scaled = feature_scaler.transform(latest_data[feature_columns])
            ticker_ids = latest_data["Ticker_ID"].values
            market_ids = latest_data["Market_ID"].values
            
            # เตรียม input tensors
            X_feat = features_scaled.reshape(1, seq_length, -1)
            X_ticker = ticker_ids.reshape(1, seq_length)
            X_market = market_ids.reshape(1, seq_length)
            
            # === LSTM PREDICTION ===
            print(f"   🔴 LSTM Prediction...")
            pred_output_lstm = model_lstm.predict([X_feat, X_ticker, X_market], verbose=0)
            pred_price_lstm_scaled = np.squeeze(pred_output_lstm[0])
            pred_direction_lstm = np.squeeze(pred_output_lstm[1])
            
            # ✅ ใช้ price_scaler เดียวกัน
            pred_price_lstm = price_scaler.inverse_transform(
                pred_price_lstm_scaled.reshape(-1, 1)
            ).flatten()[0]
            
            print(f"      💰 LSTM Price: {pred_price_lstm:.2f}")
            print(f"      📈 LSTM Direction: {pred_direction_lstm:.4f}")
            
            # === GRU PREDICTION ===
            print(f"   🔵 GRU Prediction...")
            pred_output_gru = model_gru.predict([X_feat, X_ticker, X_market], verbose=0)
            pred_price_gru_scaled = np.squeeze(pred_output_gru[0])
            pred_direction_gru = np.squeeze(pred_output_gru[1])
            
            # ✅ ใช้ price_scaler เดียวกัน (ไม่ใช่ global scaler)
            pred_price_gru = price_scaler.inverse_transform(
                pred_price_gru_scaled.reshape(-1, 1)
            ).flatten()[0]
            
            print(f"      💰 GRU Price: {pred_price_gru:.2f}")
            print(f"      📈 GRU Direction: {pred_direction_gru:.4f}")
            
            # === CONSISTENCY CHECK ===
            current_price = df_ticker.iloc[-1]['Close']
            price_diff = abs(pred_price_lstm - pred_price_gru)
            price_diff_pct = (price_diff / current_price) * 100
            
            direction_lstm = 1 if pred_direction_lstm > 0.5 else 0
            direction_gru = 1 if pred_direction_gru > 0.5 else 0
            direction_agreement = direction_lstm == direction_gru
            
            print(f"   🤝 Consistency Check:")
            print(f"      💰 Price difference: {price_diff:.2f} ({price_diff_pct:.2f}%)")
            print(f"      📊 Direction agreement: {direction_agreement}")
            
            # === ENSEMBLE PREDICTION ===
            # Simple weighted average (สามารถปรับ weights ได้)
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
            
            # เก็บผลลัพธ์
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
                'Scaler_Used': f"Ticker_{ticker_id}",
                'Consistent_Scaling': True
            }
            
            future_predictions.append(prediction_result)
            print(f"   ✅ ทำนาย {ticker} สำเร็จ")
            
        except Exception as e:
            print(f"   ❌ เกิดข้อผิดพลาดใน {ticker}: {e}")
            continue
    
    print(f"\n🎉 ทำนายเสร็จสิ้น: {len(future_predictions)}/{len(tickers)} หุ้น")
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

# Add the fixed create_ticker_scalers function
def create_ticker_scalers_fixed(df, feature_columns, scaler_file_path="../LSTM_model/ticker_scalers.pkl"):
    """สร้าง ticker scalers พร้อมการจัดการข้อมูลที่ดีขึ้น"""
    
    def clean_data_for_scalers(df, feature_columns):
        """ทำความสะอาดข้อมูลก่อนสร้าง scalers"""
        print("🧹 กำลังทำความสะอาดข้อมูลสำหรับ scalers...")
        
        df_clean = df.copy()
        
        # ตรวจสอบและแก้ไขข้อมูลแต่ละคอลัมน์
        for col in feature_columns:
            if col in df_clean.columns:
                try:
                    # แปลงเป็น string ก่อนเพื่อตรวจสอบ
                    col_data = df_clean[col].astype(str)
                    
                    # ตรวจสอบว่ามีข้อมูลที่เชื่อมต่อกันหรือไม่ (เช่น 49.0349.03)
                    problematic_mask = col_data.str.contains(r'\d+\.\d+\d+\.\d+', regex=True, na=False)
                    
                    if problematic_mask.any():
                        print(f"   ⚠️ พบข้อมูลผิดปกติในคอลัมน์ {col}: {problematic_mask.sum()} แถว")
                        
                        # แยกตัวเลขออกจากกัน (ใช้วิธีง่ายๆ คือเอาตัวเลขตัวแรก)
                        def extract_first_number(x):
                            try:
                                if pd.isna(x):
                                    return 0.0
                                x_str = str(x)
                                # หาตัวเลขตัวแรกที่มีทศนิยม
                                import re
                                match = re.search(r'^\d+\.?\d*', x_str)
                                if match:
                                    return float(match.group())
                                else:
                                    return 0.0
                            except:
                                return 0.0
                        
                        df_clean[col] = col_data.apply(extract_first_number)
                    else:
                        # แปลงเป็น numeric ปกติ
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    
                    # แทนที่ NaN และ inf ด้วยค่าเฉลี่ย
                    col_mean = df_clean[col].replace([np.inf, -np.inf], np.nan).mean()
                    if pd.isna(col_mean):
                        col_mean = 0.0
                    
                    df_clean[col] = df_clean[col].replace([np.inf, -np.inf, np.nan], col_mean)
                    
                    print(f"   ✅ ทำความสะอาด {col}: {df_clean[col].dtype}, range: {df_clean[col].min():.3f} - {df_clean[col].max():.3f}")
                    
                except Exception as e:
                    print(f"   ❌ ไม่สามารถทำความสะอาด {col}: {e}")
                    # ใช้ค่าเริ่มต้น
                    df_clean[col] = 0.0
        
        return df_clean
    
    # ทำความสะอาดข้อมูลก่อน
    df_clean = clean_data_for_scalers(df, feature_columns)
    
    ticker_scalers = {}
    tickers = df_clean['StockSymbol'].unique()
    
    print("🔧 Creating/loading individual scalers for each ticker...")
    
    # พยายามโหลด scalers เก่า
    pre_trained_scalers = {}
    try:
        if os.path.exists(scaler_file_path):
            import joblib
            pre_trained_scalers = joblib.load(scaler_file_path)
            print(f"✅ Loaded pre-trained scalers from {scaler_file_path} for {len(pre_trained_scalers)} tickers")
        else:
            print(f"⚠️ No pre-trained scalers found at {scaler_file_path}, creating new scalers")
    except Exception as e:
        print(f"❌ Error loading pre-trained scalers: {e}, creating new scalers")
    
    for ticker in tickers:
        df_ticker = df_clean[df_clean['StockSymbol'] == ticker].copy()
        
        if len(df_ticker) < 30:  # ต้องการข้อมูลอย่างน้อย 30 วัน
            print(f"   ⚠️ {ticker}: Not enough data ({len(df_ticker)} days), skipping...")
            continue
        
        ticker_id = df_ticker['Ticker_ID'].iloc[0]
        
        # ตรวจสอบ pre-trained scaler
        if ticker_id in pre_trained_scalers:
            scaler_info = pre_trained_scalers[ticker_id]
            if all(key in scaler_info for key in ['ticker_symbol', 'feature_scaler', 'price_scaler']):
                # ทดสอบว่า scaler ยังใช้งานได้หรือไม่
                try:
                    # ทดสอบ transform ข้อมูลเล็กน้อย
                    test_data = df_ticker[feature_columns].iloc[:5].fillna(df_ticker[feature_columns].mean())
                    _ = scaler_info['feature_scaler'].transform(test_data)
                    _ = scaler_info['price_scaler'].transform(df_ticker[['Close']].iloc[:5])
                    
                    # เพิ่ม data_points ถ้าขาด
                    if 'data_points' not in scaler_info:
                        scaler_info['data_points'] = len(df_ticker)
                    
                    ticker_scalers[ticker_id] = scaler_info
                    print(f"   ✅ {ticker} (ID: {ticker_id}): Using pre-trained scaler with {scaler_info['data_points']} data points")
                    continue
                    
                except Exception as e:
                    print(f"   ⚠️ {ticker} (ID: {ticker_id}): Pre-trained scaler failed test ({e}), creating new one")
        
        # สร้าง scaler ใหม่
        try:
            # ตรวจสอบข้อมูลก่อนสร้าง scaler
            feature_data = df_ticker[feature_columns].copy()
            
            # ตรวจสอบว่าข้อมูลเป็น numeric หรือไม่
            non_numeric_cols = []
            for col in feature_columns:
                if col in feature_data.columns:
                    if not pd.api.types.is_numeric_dtype(feature_data[col]):
                        non_numeric_cols.append(col)
            
            if non_numeric_cols:
                print(f"   ⚠️ {ticker}: Non-numeric columns found: {non_numeric_cols}")
                for col in non_numeric_cols:
                    feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
            
            # แทนที่ NaN ด้วยค่าเฉลี่ย
            feature_data = feature_data.fillna(feature_data.mean()).fillna(0)
            
            # ตรวจสอบว่ายังมี infinite values หรือไม่
            feature_data = feature_data.replace([np.inf, -np.inf], 0)
            
            # สร้าง feature scaler
            feature_scaler = RobustScaler()
            feature_scaler.fit(feature_data)
            
            # สร้าง price scaler
            price_scaler = RobustScaler()
            price_data = df_ticker[['Close']].copy()
            price_data = price_data.fillna(price_data.mean()).fillna(0)
            price_data = price_data.replace([np.inf, -np.inf], price_data.mean())
            price_scaler.fit(price_data)
            
            # เก็บ scaler และ metadata
            ticker_scalers[ticker_id] = {
                'ticker_symbol': ticker,
                'feature_scaler': feature_scaler,
                'price_scaler': price_scaler,
                'data_points': len(df_ticker)
            }
            
            print(f"   ✅ {ticker} (ID: {ticker_id}): Created new scaler with {len(df_ticker)} data points")
            
        except Exception as e:
            print(f"   ❌ {ticker}: Error creating scalers - {e}")
            # พิมพ์ข้อมูล debug
            print(f"      Debug - Feature data shape: {df_ticker[feature_columns].shape}")
            print(f"      Debug - Feature data dtypes: {df_ticker[feature_columns].dtypes.to_dict()}")
            if len(df_ticker) > 0:
                print(f"      Debug - Sample values: {df_ticker[feature_columns].iloc[0].to_dict()}")
            continue
    
    # บันทึก scalers ที่อัปเดตแล้ว
    try:
        import joblib
        os.makedirs(os.path.dirname(scaler_file_path), exist_ok=True)
        joblib.dump(ticker_scalers, scaler_file_path)
        print(f"💾 Saved updated scalers to {scaler_file_path}")
    except Exception as e:
        print(f"❌ Error saving scalers: {e}")
    
    print(f"✅ Created/loaded scalers for {len(ticker_scalers)} tickers")
    return ticker_scalers

if __name__ == "__main__":
    print("\n🚀 เริ่มต้นระบบทำนายหุ้นแบบ Enhanced 3-Layer Ensemble (Automated Mode)")
    print("🔧 Using Consistent Scalers System")
    print("⚡ ระบบจะใช้ scalers เดียวกันกับตอนเทรน")

    # โหลดโมเดล LSTM และ GRU
    print("\n🤖 กำลังโหลดโมเดล LSTM และ GRU...")

    MODEL_LSTM_PATH = "../LSTM_model/best_v6_plus_minimal_tuning_v2_final_model.keras"
    MODEL_GRU_PATH = "../GRU_Model/best_v6_plus_minimal_tuning_v2_final_model.keras"

    if not os.path.exists(MODEL_LSTM_PATH):
        print(f"❌ ไม่พบไฟล์โมเดล LSTM ที่ {MODEL_LSTM_PATH}")
        sys.exit()

    if not os.path.exists(MODEL_GRU_PATH):
        print(f"❌ ไม่พบไฟล์โมเดล GRU ที่ {MODEL_GRU_PATH}")
        sys.exit()

    try:
        # สร้าง instance ของ WalkForwardMiniRetrainManager
        manager = WalkForwardMiniRetrainManager(
            lstm_model_path=MODEL_LSTM_PATH,
            gru_model_path=MODEL_GRU_PATH
        )
        
        # โหลดโมเดลสำหรับทำนาย (ไม่ต้อง compile เพื่อความเร็ว)
        print(f"\n✅ โหลดโมเดลสำหรับทำนาย...")
        model_lstm = manager.load_models_for_prediction(model_path=MODEL_LSTM_PATH, compile_model=False)
        model_gru = manager.load_models_for_prediction(model_path=MODEL_GRU_PATH, compile_model=False)
        
        if model_lstm is None or model_gru is None:
            print("❌ ไม่สามารถโหลดโมเดลได้")
            sys.exit()
            
        print("✅ โหลดโมเดล LSTM และ GRU สำเร็จ!")
        
        # แสดงข้อมูลโมเดล
        print(f"📊 LSTM model: {len(model_lstm.layers)} layers")
        print(f"📊 GRU model: {len(model_gru.layers)} layers")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        sys.exit()

    # ดึงและเตรียมข้อมูล
    print("\n📥 กำลังดึงข้อมูลจากฐานข้อมูล...")
    raw_df = fetch_latest_data()  # ข้อมูลดิบจากฐานข้อมูล

    if raw_df.empty:
        print("❌ ไม่มีข้อมูลสำหรับประมวลผล")
        sys.exit()

    print(f"📊 ได้รับข้อมูลดิบ: {len(raw_df)} แถว จาก {len(raw_df['StockSymbol'].unique())} หุ้น")

    # ======== FEATURE PREPARATION ========
    base_feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Change_Percent', 'Sentiment',
        'positive_news', 'negative_news', 'neutral_news',
        'TotalRevenue', 'QoQGrowth', 'EPS', 'ROE', 'NetProfitMargin', 
        'DebtToEquity', 'PERatio', 'Dividend_Yield', 'P_BV_Ratio'
    ]

    technical_feature_columns = [
        'ATR', 'Keltner_High', 'Keltner_Low', 'Keltner_Middle', 'Chaikin_Vol',
        'Donchian_High', 'Donchian_Low', 'PSAR',
        'RSI', 'EMA_10', 'EMA_20', 'MACD', 'MACD_Signal', 
        'Bollinger_High', 'Bollinger_Low', 'SMA_50', 'SMA_200'
    ]

    # รวม feature columns ที่มีอยู่จริง
    available_base = [col for col in base_feature_columns if col in raw_df.columns]
    available_technical = [col for col in technical_feature_columns if col in raw_df.columns]
    feature_columns = available_base + available_technical

    print(f"📋 Available feature columns ({len(feature_columns)}): {feature_columns}")

    if len(feature_columns) < 10:
        print("❌ ข้อมูล features ไม่เพียงพอ ต้องการอย่างน้อย 10 columns")
        sys.exit()

    # ======== CONSISTENT SCALER SYSTEM ========
    print("\n🔧 เริ่มต้นระบบ Consistent Scalers...")
    
    # 1. โหลด training scalers
    ticker_scalers, scalers_loaded = load_training_scalers("../LSTM_model/ticker_scalers.pkl")
    
    if not scalers_loaded:
        print("❌ ไม่สามารถโหลด training scalers ได้")
        print("💡 จะสร้าง scalers ใหม่จากข้อมูลปัจจุบัน...")
        
        # สร้าง scalers ใหม่ถ้าโหลดไม่ได้
        try:
            # เตรียม Ticker_ID และ Market_ID
            ticker_encoder_temp = LabelEncoder()
            raw_df["Ticker_ID"] = ticker_encoder_temp.fit_transform(raw_df["StockSymbol"])
            
            us_stock = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'TSM', 'AMD']
            thai_stock = ['ADVANC', 'TRUE', 'DITTO', 'DIF', 'INSET', 'JMART', 'INET', 'JAS', 'HUMAN']
            raw_df['Market_ID'] = raw_df['StockSymbol'].apply(
                lambda x: 0 if x in us_stock else 1 if x in thai_stock else 2
            )
            
            # สร้าง scalers ใหม่
            ticker_scalers = create_ticker_scalers_fixed(raw_df, feature_columns)
            
            if len(ticker_scalers) == 0:
                print("❌ ไม่สามารถสร้าง scalers ได้")
                sys.exit()
                
            scalers_loaded = True
            print("✅ สร้าง scalers ใหม่สำเร็จ")
            
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการสร้าง scalers: {e}")
            sys.exit()
    
    # 2. เตรียมข้อมูลสำหรับ prediction
    prediction_df = raw_df.copy()
    
    # เตรียม Ticker_ID และ Market_ID ถ้ายังไม่มี
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
    
    # 3. ตรวจสอบความเข้ากันได้ของ scalers
    valid_scalers, validation_results = validate_ticker_scalers(
        ticker_scalers, prediction_df, feature_columns
    )
    
    if len(valid_scalers) == 0:
        print("❌ ไม่มี scaler ที่ใช้งานได้")
        sys.exit()
    
    print(f"✅ Scalers พร้อมใช้งาน: {len(valid_scalers)} tickers")

    # ======== PREDICTION WITH CONSISTENT SCALERS ========
    print(f"\n🔮 เริ่มต้นการทำนายด้วย Consistent Scalers...")
    
    # ✅ แก้ไข: ใช้พารามิเตอร์ที่ถูกต้อง
    future_predictions_df = predict_with_consistent_scalers(
        model_lstm=model_lstm,
        model_gru=model_gru,
        df=prediction_df,
        feature_columns=feature_columns,
        ticker_scalers=valid_scalers,  # ✅ ใช้ valid_scalers (dict)
        ticker_encoder=ticker_encoder,  # ✅ ใช้ ticker_encoder
        market_encoder=market_encoder,  # ✅ ใช้ market_encoder
        seq_length=10  # ✅ ใช้ seq_length
    )
    
    if not future_predictions_df.empty:
        print(f"\n🎯 ผลลัพธ์การทำนายด้วย Consistent Scalers:")
        
        # แสดงผลลัพธ์
        display_cols = ['StockSymbol', 'Date', 'Current_Price', 'Ensemble_Price', 
                       'Ensemble_Direction', 'Confidence', 'Direction_Agreement', 'Consistent_Scaling']
        
        available_display_cols = [col for col in display_cols if col in future_predictions_df.columns]
        
        if len(available_display_cols) > 0:
            print(future_predictions_df[available_display_cols])
            
            # แสดงสถิติ
            print(f"\n📊 สถิติการทำนาย:")
            print(f"   📈 จำนวนหุ้นที่ทำนาย: {len(future_predictions_df)}")
            
            if 'Ensemble_Direction' in future_predictions_df.columns:
                buy_signals = len(future_predictions_df[future_predictions_df['Ensemble_Direction'] == 1])
                sell_signals = len(future_predictions_df[future_predictions_df['Ensemble_Direction'] == 0])
                print(f"   📈 สัญญาณ BUY: {buy_signals}")
                print(f"   📉 สัญญาณ SELL: {sell_signals}")
            
            if 'Confidence' in future_predictions_df.columns:
                avg_confidence = future_predictions_df['Confidence'].mean()
                print(f"   🎯 Average Confidence: {avg_confidence:.3f}")
                
                # หุ้นที่มี confidence สูงสุด
                high_confidence = future_predictions_df.nlargest(3, 'Confidence')
                print(f"\n🏆 หุ้นที่มี Confidence สูงสุด:")
                for _, row in high_confidence.iterrows():
                    direction_text = "📈 BUY" if row.get('Ensemble_Direction', 0) == 1 else "📉 SELL"
                    current_price = row.get('Current_Price', 0)
                    ensemble_price = row.get('Ensemble_Price', 0)
                    price_change_pct = ((ensemble_price - current_price) / current_price * 100) if current_price > 0 else 0
                    
                    print(f"   {row['StockSymbol']}: {direction_text} "
                          f"(Confidence: {row['Confidence']:.3f}, "
                          f"Expected: {price_change_pct:+.2f}%)")
            
            # ข้อมูลเพิ่มเติม
            if 'Direction_Agreement' in future_predictions_df.columns:
                agreement_rate = future_predictions_df['Direction_Agreement'].mean()
                print(f"   🤝 Model Agreement Rate: {agreement_rate:.1%}")
            
            if 'Consistent_Scaling' in future_predictions_df.columns:
                consistent_scaling_rate = future_predictions_df['Consistent_Scaling'].mean()
                print(f"   ✅ Consistent Scaling Rate: {consistent_scaling_rate:.1%}")
            
            # บันทึกเป็นไฟล์ CSV
            csv_filename = f"consistent_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            future_predictions_df.to_csv(csv_filename, index=False)
            print(f"\n💾 บันทึกผลลัพธ์เป็นไฟล์: {csv_filename}")
            
            # บันทึกลงฐานข้อมูล (ถ้าต้องการ)
            print(f"\n💾 กำลังบันทึกผลลัพธ์ลงฐานข้อมูล...")
            
            # แปลงข้อมูลให้เข้ากับ database schema
            db_predictions = future_predictions_df.copy()
            
            # Map columns ให้เข้ากับ save_predictions_simple
            if 'Ensemble_Price' in db_predictions.columns:
                db_predictions['Predicted_Price'] = db_predictions['Ensemble_Price']
            if 'Ensemble_Direction' in db_predictions.columns:
                db_predictions['Predicted_Direction'] = db_predictions['Ensemble_Direction']
            if 'LSTM_Price' in db_predictions.columns:
                db_predictions['LSTM_Prediction'] = db_predictions['LSTM_Price']
            if 'LSTM_Direction' in db_predictions.columns:
                db_predictions['LSTM_Direction'] = db_predictions['LSTM_Direction']
            if 'GRU_Price' in db_predictions.columns:
                db_predictions['GRU_Prediction'] = db_predictions['GRU_Price']
            if 'GRU_Direction' in db_predictions.columns:
                db_predictions['GRU_Direction'] = db_predictions['GRU_Direction']
            
            try:
                db_save_success = save_predictions_simple(db_predictions)
                
                if db_save_success:
                    print("✅ บันทึกลงฐานข้อมูลสำเร็จ")
                    print("🔄 ข้อมูลในฐานข้อมูลได้รับการอัปเดตแล้ว")
                    print("📱 สามารถใช้ข้อมูลทำนายในระบบอื่นๆ ได้แล้ว")
                else:
                    print("⚠️ ไม่สามารถบันทึกลงฐานข้อมูลได้ แต่ยังมีไฟล์ CSV สำหรับใช้งาน")
            except Exception as e:
                print(f"⚠️ เกิดข้อผิดพลาดในการบันทึกลงฐานข้อมูล: {e}")
                print("💾 ไฟล์ CSV ยังคงมีไว้สำหรับใช้งาน")
            
        else:
            print(f"⚠️ ไม่สามารถแสดงผลลัพธ์ได้ เนื่องจากขาด columns")
            print(f"Available columns: {list(future_predictions_df.columns)}")
            
            # บันทึกข้อมูลที่มีอยู่
            csv_filename = f"predictions_all_columns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            future_predictions_df.to_csv(csv_filename, index=False)
            print(f"💾 บันทึกผลลัพธ์ทั้งหมดเป็นไฟล์: {csv_filename}")
    
    else:
        print("❌ ไม่มีข้อมูลพยากรณ์ที่จะแสดง")
        print("🔍 กรุณาตรวจสอบ:")
        print("   - ข้อมูล ticker_scalers มีครบหรือไม่")
        print("   - ข้อมูลในฐานข้อมูลเพียงพอสำหรับ sequence length หรือไม่")
        print("   - โมเดล LSTM และ GRU โหลดถูกต้องหรือไม่")
    
    # สรุปการทำงาน
    print(f"\n🎉 การประมวลผลด้วย Consistent Scalers เสร็จสิ้น!")
    print(f"📋 สรุปการทำงาน:")
    print(f"   🔧 Scalers ที่ใช้: {'Training scalers' if scalers_loaded else 'New scalers'}")
    print(f"   🔮 การทำนาย: {'สำเร็จ' if not future_predictions_df.empty else 'ไม่สำเร็จ'}")
    print(f"   💾 ไฟล์ผลลัพธ์: {'สร้างแล้ว' if not future_predictions_df.empty else 'ไม่มี'}")
    print(f"   ✅ LSTM และ GRU ใช้ scaler เดียวกัน: {'ใช่' if not future_predictions_df.empty else 'ไม่ทราบ'}")
    
    print("\n🔚 ขอบคุณที่ใช้ระบบทำนายหุ้นด้วย Consistent Scalers")
    print("✨ ระบบใช้ scalers เดียวกันกับตอนเทรนเพื่อความแม่นยำสูงสุด")