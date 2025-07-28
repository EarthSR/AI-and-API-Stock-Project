import sys
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
import xgboost as xgb
from sklearn.impute import SimpleImputer
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
import pickle
import io
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
    """
    ทำ Walk-Forward Validation แบบ Multi-Task (Price + Direction)
    แบ่งข้อมูลเป็น chunks ละ chunk_size วัน พร้อม Online Learning
    
    - Mini-retrain: ทุก retrain_frequency วัน (Online Learning แบบต่อเนื่อง)
    - Chunk-based: แบ่งข้อมูลเป็นช่วงๆ เพื่อใช้ข้อมูลทั้งหมด
    """

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
        thai_stock = ['ADVANC', 'INTUCH', 'TRUE', 'DITTO', 'DIF', 
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
if 8 <= current_hour < 18:
    print("📊 กำลังประมวลผลตลาดหุ้นไทย (SET)...")
    market_filter = "Thailand"
elif 19 <= current_hour or current_hour < 5:
    print("📊 กำลังประมวลผลตลาดหุ้นอเมริกา (NYSE & NASDAQ)...")
    market_filter = "America"
else:
    print("❌ ไม่อยู่ในช่วงเวลาทำการของตลาดหุ้นไทยหรืออเมริกา")
    exit()
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

# ======================== CORRECTED MAIN EXECUTION ========================

if __name__ == "__main__":
    print("\n🚀 เริ่มต้นระบบทำนายหุ้นแบบ Enhanced 3-Layer Ensemble (Automated Mode)")
    print("🔧 Using Unified Data Preparation System (Training + Online Learning Compatible)")
    print("⚡ ระบบจะตรวจสอบและรีเทรนอัตโนมัติทุก 5 วัน")

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

    # ตรวจสอบว่าควรรีเทรนหรือไม่
    print("\n🔍 ตรวจสอบการรีเทรน...")
    should_retrain_lstm, lstm_reason = should_retrain_model("LSTM", retrain_frequency_days=5)
    should_retrain_gru, gru_reason = should_retrain_model("GRU", retrain_frequency_days=5)
    
    need_retrain = should_retrain_lstm or should_retrain_gru
    
    print(f"📊 สถานะการรีเทรน:")
    print(f"   🔴 LSTM: {'ต้องรีเทรน' if should_retrain_lstm else 'ไม่ต้องรีเทรน'} ({lstm_reason})")
    print(f"   🔵 GRU:  {'ต้องรีเทรน' if should_retrain_gru else 'ไม่ต้องรีเทรน'} ({gru_reason})")

    try:
        # สร้าง instance ของ WalkForwardMiniRetrainManager
        manager = WalkForwardMiniRetrainManager(
            lstm_model_path=MODEL_LSTM_PATH,
            gru_model_path=MODEL_GRU_PATH
        )
        
        if need_retrain:
            print(f"\n🔄 ต้องรีเทรนโมเดล - โหลดโมเดลพร้อม compile...")
            model_lstm = manager.load_models_for_prediction(model_path=MODEL_LSTM_PATH, compile_model=True)
            model_gru = manager.load_models_for_prediction(model_path=MODEL_GRU_PATH, compile_model=True)
        else:
            print(f"\n✅ ไม่ต้องรีเทรน - โหลดโมเดลสำหรับทำนาย...")
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

    # ======== UNIFIED FEATURE PREPARATION ========
    # ใช้ feature columns ตามแนวทางโค้ดเทรน (ปรับให้เข้ากับฐานข้อมูล)
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

    # ======== UNIFIED DATA PREPARATION ========
    print("\n🔧 กำลังเตรียมข้อมูลด้วย Unified Data Preparation System...")
    
    try:
        # ใช้ระบบเตรียมข้อมูลแบบรวม (รวมแนวทางโค้ดเทรน + ระบบปัจจุบัน)
        ticker_scalers, prepared_df = create_walk_forward_compatible_scalers(raw_df, feature_columns)
        
        if len(ticker_scalers) == 0:
            print("❌ ไม่สามารถสร้าง ticker scalers ได้")
            sys.exit()
        
        print(f"✅ เตรียมข้อมูลสำเร็จ:")
        print(f"   📊 Prepared data: {len(prepared_df)} แถว")
        print(f"   🏷️ Ticker scalers: {len(ticker_scalers)} ตัว")
        print(f"   📈 Features: {len(feature_columns)} columns")
        
        # ตรวจสอบว่ามี target variables หรือไม่
        if 'Direction' in prepared_df.columns and 'TargetPrice' in prepared_df.columns:
            print(f"   🎯 Target variables created successfully")
        else:
            print(f"   ⚠️ Target variables may not be available for some operations")
        
        # แสดงข้อมูล ticker scalers
        print(f"\n📋 Ticker Scalers Summary:")
        for t_id, scaler_info in list(ticker_scalers.items())[:3]:  # แสดงแค่ 3 ตัวแรก
            ticker_name = scaler_info.get('ticker', 'Unknown')
            data_points = scaler_info.get('data_points', 'Unknown') 
            print(f"   • {ticker_name} (ID: {t_id}): {data_points} data points")
        if len(ticker_scalers) > 3:
            print(f"   ... และอีก {len(ticker_scalers) - 3} tickers")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการเตรียมข้อมูล: {e}")
        import traceback
        traceback.print_exc()
        sys.exit()

    # ======== ENCODER PREPARATION ========
    print("\n🔧 เตรียม encoders...")
    try:
        # ใช้ encoders ที่สร้างไว้แล้วใน prepared_df หรือสร้างใหม่
        if 'Ticker_ID' not in prepared_df.columns:
            ticker_encoder = LabelEncoder()
            prepared_df["Ticker_ID"] = ticker_encoder.fit_transform(prepared_df["StockSymbol"])
        else:
            # สร้าง encoder จากข้อมูลที่มีอยู่
            ticker_encoder = LabelEncoder()
            ticker_encoder.fit(prepared_df["StockSymbol"])
        
        if 'Market_ID' not in prepared_df.columns:
            # สร้าง Market_ID
            us_stock = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'TSM', 'AMD']
            thai_stock = ['ADVANC', 'INTUCH', 'TRUE', 'DITTO', 'DIF', 
                         'INSET', 'JMART', 'INET', 'JAS', 'HUMAN']
            prepared_df['Market_ID'] = prepared_df['StockSymbol'].apply(
                lambda x: "US" if x in us_stock else "TH" if x in thai_stock else "OTHER"
            )
            
            market_encoder = LabelEncoder()
            prepared_df['Market_ID'] = market_encoder.fit_transform(prepared_df['Market_ID'])
        else:
            # สร้าง encoder จากข้อมูลที่มีอยู่
            market_encoder = LabelEncoder()
            market_encoder.fit(prepared_df['Market_ID'].astype(str))
        
        print("✅ เตรียม encoders สำเร็จ")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการเตรียม encoders: {e}")
        sys.exit()

    # ======== AUTOMATED WORKFLOW ========
    
    if need_retrain:
        # ======================== RETRAIN MODE ========================
        print(f"\n🔄 เริ่มต้นการรีเทรนโมเดลอัตโนมัติ...")
        
        # กำหนดพารามิเตอร์
        chunk_size = 200
        retrain_freq = 5
        SEQ_LENGTH = 10
        
        print(f"\n🎯 พารามิเตอร์การรีเทรน:")
        print(f"   📦 Chunk size: {chunk_size} วัน")
        print(f"   🔄 Retrain frequency: {retrain_freq} วัน")
        print(f"   📈 Sequence length: {SEQ_LENGTH} วัน")
        print(f"   🤖 Models: LSTM + GRU (ทั้งสองโมเดล)")
        
        retrain_success = False
        
        try:
            if should_retrain_lstm:
                print(f"\n🔍 กำลังรีเทรน LSTM...")
                predictions_lstm, metrics_lstm = walk_forward_validation_multi_task_batch(
                    model=model_lstm,
                    df=prepared_df,
                    feature_columns=feature_columns,
                    ticker_scalers=ticker_scalers,
                    ticker_encoder=ticker_encoder,
                    market_encoder=market_encoder,
                    seq_length=SEQ_LENGTH,
                    retrain_frequency=retrain_freq,
                    chunk_size=chunk_size
                )
                
                if not predictions_lstm.empty:
                    predictions_lstm.to_csv('retrain_lstm_results.csv', index=False)
                    update_retrain_date("LSTM")
                    print("✅ รีเทรน LSTM สำเร็จ")
                    retrain_success = True
                
            if should_retrain_gru:
                print(f"\n🔍 กำลังรีเทรน GRU...")
                predictions_gru, metrics_gru = walk_forward_validation_multi_task_batch(
                    model=model_gru,
                    df=prepared_df,
                    feature_columns=feature_columns,
                    ticker_scalers=ticker_scalers,
                    ticker_encoder=ticker_encoder,
                    market_encoder=market_encoder,
                    seq_length=SEQ_LENGTH,
                    retrain_frequency=retrain_freq,
                    chunk_size=chunk_size
                )
                
                if not predictions_gru.empty:
                    predictions_gru.to_csv('retrain_gru_results.csv', index=False)
                    update_retrain_date("GRU")
                    print("✅ รีเทรน GRU สำเร็จ")
                    retrain_success = True
            
            if retrain_success:
                print(f"\n🎉 การรีเทรนเสร็จสิ้น! กำลังดำเนินการทำนาย...")
                print(f"💾 ไฟล์การรีเทรน: retrain_lstm_results.csv, retrain_gru_results.csv")
            else:
                print(f"\n⚠️ การรีเทรนไม่สำเร็จ แต่จะดำเนินการทำนายต่อไป...")
            
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการรีเทรน: {e}")
            print(f"⚠️ จะดำเนินการทำนายต่อไปด้วยโมเดลเดิม...")
            import traceback
            traceback.print_exc()
    
    else:
        print(f"\n✅ ไม่ต้องรีเทรน - ดำเนินการทำนายด้วยโมเดลปัจจุบัน")

    # ======================== PREDICTION MODE ========================
    
    print(f"\n🔮 เริ่มต้นการทำนายอัตโนมัติ...")
    
    # สร้าง main scalers สำหรับ prediction
    print("🔧 กำลังสร้าง main scalers สำหรับ prediction...")
    scaler_main_features = RobustScaler()
    scaler_main_target = RobustScaler()
    
    try:
        # ใช้ข้อมูลที่เตรียมแล้ว
        feature_data = prepared_df[feature_columns]
        if feature_data.isnull().any().any():
            print("⚠️ พบ NaN ในข้อมูล features, กำลังจัดการ...")
            feature_data = feature_data.fillna(feature_data.mean())
            prepared_df[feature_columns] = feature_data
        
        scaler_main_features.fit(prepared_df[feature_columns])
        scaler_main_target.fit(prepared_df[["Close"]])
        print("✅ สร้าง main scalers สำเร็จ")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการสร้าง main scalers: {e}")
        sys.exit()
    
    # ตรวจสอบข้อมูล predictions จาก LSTM/GRU
    prediction_cols = ['PredictionClose_LSTM', 'PredictionClose_GRU', 
                      'PredictionTrend_LSTM', 'PredictionTrend_GRU']
    available_predictions = [col for col in prediction_cols if col in prepared_df.columns]
    print(f"🔮 Available prediction columns: {available_predictions}")
    
    if len(available_predictions) < 4:
        print("⚠️ ไม่มีข้อมูล predictions จาก LSTM/GRU ครบ, XGBoost Meta-Learner จะไม่สามารถทำงานได้")
    
    # กำหนด SEQ_LENGTH สำหรับ prediction
    SEQ_LENGTH = 10
    
    # ทำนายด้วย Enhanced 3-Layer Ensemble
    future_predictions_df = predict_future_day_with_meta(
        model_lstm, model_gru, prepared_df, feature_columns, 
        scaler_main_features, scaler_main_target, ticker_encoder, SEQ_LENGTH
    )
    
    if not future_predictions_df.empty:
        print(f"\n🎯 ผลลัพธ์การทำนายอัตโนมัติ (Enhanced 3-Layer Ensemble):")
        
        # บันทึกผลลัพธ์
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'automated_predictions_{timestamp}.csv'
        future_predictions_df.to_csv(output_path, index=False)
        print(f"💾 บันทึกผลลัพธ์ใน {output_path}")
        
        # แสดงผลลัพธ์
        display_cols = ['StockSymbol', 'Date', 'Last_Close', 'Predicted_Price', 
                       'Price_Change_Percent', 'Predicted_Direction', 'XGB_Confidence',
                       'Ensemble_Method', 'Model_Agreement']
        
        if all(col in future_predictions_df.columns for col in display_cols):
            print(future_predictions_df[display_cols])
            
            # แสดงสถิติ
            print(f"\n📊 สถิติการทำนาย:")
            print(f"   📈 จำนวนหุ้นที่ทำนาย: {len(future_predictions_df)}")
            print(f"   📈 สัญญาณ BUY: {len(future_predictions_df[future_predictions_df['Predicted_Direction'] == 1])}")
            print(f"   📈 สัญญาณ SELL: {len(future_predictions_df[future_predictions_df['Predicted_Direction'] == 0])}")
            
            avg_confidence = future_predictions_df['XGB_Confidence'].mean()
            print(f"   📈 Average Confidence: {avg_confidence:.3f}")
            
            # หุ้นที่มี confidence สูงสุด
            high_confidence = future_predictions_df.nlargest(3, 'XGB_Confidence')
            print(f"\n🏆 หุ้นที่มี Confidence สูงสุด:")
            for _, row in high_confidence.iterrows():
                direction_text = "📈 BUY" if row['Predicted_Direction'] == 1 else "📉 SELL"
                print(f"   {row['StockSymbol']}: {direction_text} "
                      f"(Confidence: {row['XGB_Confidence']:.3f}, "
                      f"Expected: {row['Price_Change_Percent']:.2f}%)")
            
            # บันทึกผลลัพธ์ลงฐานข้อมูลอัตโนมัติ
            print(f"\n💾 กำลังบันทึกผลลัพธ์ลงฐานข้อมูลอัตโนมัติ...")
            db_save_success = save_predictions_simple(future_predictions_df)
            
            if db_save_success:
                print("✅ บันทึกลงฐานข้อมูลสำเร็จ")
                print("🔄 ข้อมูลในฐานข้อมูลได้รับการอัปเดตแล้ว")
                print("📱 สามารถใช้ข้อมูลทำนายในระบบอื่นๆ ได้แล้ว")
            else:
                print("⚠️ ไม่สามารถบันทึกลงฐานข้อมูลได้ แต่ยังมีไฟล์ CSV สำหรับใช้งาน")
        else:
            missing_cols = [col for col in display_cols if col not in future_predictions_df.columns]
            print(f"⚠️ ไม่สามารถแสดงผลลัพธ์ได้ เนื่องจากขาด columns: {missing_cols}")
            print("แต่ไฟล์ CSV ได้ถูกสร้างแล้ว")
    
    else:
        print("❌ ไม่มีข้อมูลพยากรณ์ที่จะแสดง")
    
    # สรุปการทำงาน
    print(f"\n🎉 การประมวลผลอัตโนมัติเสร็จสิ้น!")
    print(f"📋 สรุปการทำงาน:")
    print(f"   🔄 การรีเทรน: {'ดำเนินการแล้ว' if need_retrain else 'ไม่จำเป็น'}")
    print(f"   🔮 การทำนาย: {'สำเร็จ' if not future_predictions_df.empty else 'ไม่สำเร็จ'}")
    print(f"   💾 ไฟล์ผลลัพธ์: {output_path if not future_predictions_df.empty else 'ไม่มี'}")
    print(f"   🗓️ วันที่รีเทรนครั้งถัดไป: {(datetime.now() + timedelta(days=5)).strftime('%Y-%m-%d')}")
    
    print("\n🔚 ขอบคุณที่ใช้ระบบทำนายหุ้น Enhanced 3-Layer Ensemble (Automated Mode)")
    print("✨ ระบบได้ทำงานอัตโนมัติและพร้อมใช้งานทุกวัน")