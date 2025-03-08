import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, GRU, Dense, Dropout, BatchNormalization,
    Embedding, concatenate, Bidirectional, Masking,
    Conv1D, Flatten, LSTM
)
from tensorflow.keras.losses import Loss
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import ta
import matplotlib.pyplot as plt
import joblib
import logging

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
print("TensorFlow running on:", tf.config.list_physical_devices())


# --------------------------------------------------------------------
# 1) ฟังก์ชันคำนวณ Metric
# --------------------------------------------------------------------
def custom_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE) แบบไม่ให้เกิด Infinity ถ้า y_true = 0
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_mask = y_true != 0
    if not np.any(nonzero_mask):
        return np.nan
    return np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100

def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    nonzero_mask = denominator != 0
    if not np.any(nonzero_mask):
        return np.nan
    return np.mean(diff[nonzero_mask] / denominator[nonzero_mask]) * 100

# --------------------------------------------------------------------
# 2) ฟังก์ชันสร้าง Sequence (Multi-Task)
# --------------------------------------------------------------------
def create_sequences_for_ticker(features, ticker_ids, targets_price, targets_dir, seq_length=10):
    """
    สร้าง Sequence สำหรับ Multi-Task:
    คืนค่า 4 อย่าง: (X_features, X_tickers, Y_price, Y_dir)
    โดยแต่ละ Sequence ยาว seq_length
    """
    X_features, X_tickers = [], []
    Y_price, Y_dir = [], []
    for i in range(len(features) - seq_length):
        X_features.append(features[i : i + seq_length])
        X_tickers.append(ticker_ids[i : i + seq_length])
        Y_price.append(targets_price[i + seq_length])
        Y_dir.append(targets_dir[i + seq_length])
    return (
        np.array(X_features),
        np.array(X_tickers),
        np.array(Y_price),
        np.array(Y_dir),
    )

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Total Loss')
    plt.plot(history.history['val_loss'], label='Val Total Loss')
    plt.title('Total Loss (Multi-Task)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # (ตัวอย่าง) plot "price_output_mae"
    if 'price_output_mae' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['price_output_mae'], label='Train Price MAE')
        plt.plot(history.history['val_price_output_mae'], label='Val Price MAE')
        plt.title('Price Output MAE')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

logging.basicConfig(level=logging.INFO, filename='training.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------------------------------------
# 3) ตรวจสอบ GPU
# --------------------------------------------------------------------
physical_devices = tf.config.list_physical_devices('CPU')
if len(physical_devices) > 0:
    try:
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logging.info(f"Memory growth enabled for GPU: {physical_devices[0]}")
        print("Memory growth enabled for GPU:", physical_devices[0])
    except Exception as e:
        logging.error(f"Failed to set memory growth: {e}")
        print(f"Error setting memory growth: {e}")
else:
    logging.info("GPU not found, using CPU")
    print("GPU not found, using CPU")

# --------------------------------------------------------------------
# 4) โหลดข้อมูล & เตรียมฟีเจอร์
# --------------------------------------------------------------------
df = pd.read_csv('../merged_stock_sentiment_financial.csv')

df['Sentiment'] = df['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})

df['Change'] = df['Close'] - df['Open']
df['Change (%)'] = df['Close'].pct_change() * 100
upper_bound = df["Change (%)"].quantile(0.99)
lower_bound = df["Change (%)"].quantile(0.01)
df["Change (%)"] = np.clip(df["Change (%)"], lower_bound, upper_bound)

df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
df['RSI'].fillna(method='ffill', inplace=True)
df['RSI'].fillna(0, inplace=True)
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].rolling(window=9).mean()
bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
df['Bollinger_High'] = bollinger.bollinger_hband()
df['Bollinger_Low']  = bollinger.bollinger_lband()

financial_columns = [
    'Total Revenue','QoQ Growth (%)','YoY Growth (%)','Net Profit',
    'Earnings Per Share (EPS)','ROA (%)','ROE (%)','Gross Margin (%)',
    'Net Profit Margin (%)','Debt to Equity ','P/E Ratio ','P/BV Ratio ',
    'Dividend Yield (%)'
]
df_financial = df[['Date','Ticker'] + financial_columns].drop_duplicates()
df_financial[financial_columns] = df_financial[financial_columns].where(df_financial[financial_columns].ne(0)).bfill()

stock_columns = [
    'RSI','EMA_10','EMA_20','MACD','MACD_Signal',
    'Bollinger_High','Bollinger_Low','SMA_50','SMA_200'
]
df[stock_columns] = df[stock_columns].fillna(method='ffill')
df.fillna(0, inplace=True)

feature_columns = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment',
    'Total Revenue','QoQ Growth (%)', 'YoY Growth (%)', 'Net Profit', 
    'Earnings Per Share (EPS)', 'ROA (%)', 'ROE (%)', 'Gross Margin (%)', 
    'Net Profit Margin (%)', 'Debt to Equity ', 'P/E Ratio ', 'P/BV Ratio ', 
    'Dividend Yield (%)','RSI', 'EMA_10', 'EMA_20', 'MACD', 'MACD_Signal',
    'Bollinger_High', 'Bollinger_Low','SMA_50', 'SMA_200'
]

# --------------------------------------------------------------------
# 4.1) สร้าง Direction และ TargetPrice
# --------------------------------------------------------------------
df['Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df['TargetPrice'] = df['Close'].shift(-1)
df.dropna(subset=['Direction','TargetPrice'], inplace=True)

# --------------------------------------------------------------------
# 4.2) LabelEncoder Ticker
# --------------------------------------------------------------------
ticker_encoder = LabelEncoder()
df['Ticker_ID'] = ticker_encoder.fit_transform(df['Ticker'])
num_tickers = len(ticker_encoder.classes_)

# --------------------------------------------------------------------
# 5) แบ่ง Train/Test ตามวันที่
# --------------------------------------------------------------------
sorted_dates = df['Date'].unique()
train_cutoff = sorted_dates[int(len(sorted_dates) * 6 / 7)]
train_df = df[df['Date'] <= train_cutoff].copy()
test_df  = df[df['Date'] > train_cutoff].copy()

train_df.to_csv('train_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)

print("Train cutoff:", train_cutoff)
print("First date in train set:", train_df['Date'].min())
print("Last date in train set:", train_df['Date'].max())

seq_length = 10

# --------------------------------------------------------------------
# 6) Scaler แยกต่อ Ticker (Feature + Price)
# --------------------------------------------------------------------
scaler_features_dict = {}
scaler_price_dict = {}

train_dict = {}
test_dict  = {}

for t_id in range(num_tickers):
    df_train_ticker = train_df[train_df['Ticker_ID'] == t_id].copy()
    if len(df_train_ticker) <= seq_length:
        continue

    price_train = df_train_ticker['TargetPrice'].values.reshape(-1,1)
    f_train = df_train_ticker[feature_columns].values

    min_len = min(len(f_train), len(price_train))
    f_train = f_train[:min_len]
    price_train = price_train[:min_len]
    df_train_ticker = df_train_ticker.iloc[:min_len]

    sc_f = RobustScaler(quantile_range=(5,95))
    sc_p = RobustScaler(quantile_range=(5,95))
    sc_f.fit(f_train)
    sc_p.fit(price_train)

    scaler_features_dict[t_id] = sc_f
    scaler_price_dict[t_id] = sc_p
    train_dict[t_id] = df_train_ticker.copy()

for t_id in range(num_tickers):
    df_test_ticker = test_df[test_df['Ticker_ID'] == t_id].copy()
    if len(df_test_ticker) <= seq_length:
        continue
    test_dict[t_id] = df_test_ticker.copy()

# --------------------------------------------------------------------
# 7) สร้าง Sequence Multi-Task (Price + Direction)
# --------------------------------------------------------------------
X_train_list, X_train_ticker_list = [], []
y_train_price_list, y_train_dir_list = [], []

X_test_list, X_test_ticker_list = [], []
y_test_price_list, y_test_dir_list = [], []

for t_id in range(num_tickers):
    if t_id not in train_dict:
        continue

    df_train_ticker = train_dict[t_id]
    if len(df_train_ticker) < seq_length:
        continue

    sc_f = scaler_features_dict[t_id]
    sc_p = scaler_price_dict[t_id]

    f_train = df_train_ticker[feature_columns].values
    p_train = df_train_ticker['TargetPrice'].values.reshape(-1,1)
    d_train = df_train_ticker['Direction'].values  # 0/1

    min_len = min(len(f_train), len(p_train), len(d_train))
    f_train = f_train[:min_len]
    p_train = p_train[:min_len]
    d_train = d_train[:min_len]

    f_train_scaled = sc_f.transform(f_train)
    p_train_scaled = sc_p.transform(p_train)

    ticker_ids = df_train_ticker['Ticker_ID'].values[:min_len]

    Xf, Xt, Yp, Yd = create_sequences_for_ticker(
        f_train_scaled, ticker_ids, p_train_scaled, d_train, seq_length
    )
    X_train_list.append(Xf)
    X_train_ticker_list.append(Xt)
    y_train_price_list.append(Yp)
    y_train_dir_list.append(Yd)

    # ------------ Test ส่วน -------------
    if t_id not in test_dict:
        continue

    df_test_ticker = test_dict[t_id]
    if len(df_test_ticker) < seq_length:
        continue

    f_test = df_test_ticker[feature_columns].values
    p_test = df_test_ticker['TargetPrice'].values.reshape(-1,1)
    d_test = df_test_ticker['Direction'].values

    min_len_test = min(len(f_test), len(p_test), len(d_test))
    f_test = f_test[:min_len_test]
    p_test = p_test[:min_len_test]
    d_test = d_test[:min_len_test]

    f_test_scaled = sc_f.transform(f_test)
    p_test_scaled = sc_p.transform(p_test)

    ticker_ids_test = df_test_ticker['Ticker_ID'].values[:min_len_test]

    Xs, Xts, Yps, Yds = create_sequences_for_ticker(
        f_test_scaled, ticker_ids_test, p_test_scaled, d_test, seq_length
    )
    X_test_list.append(Xs)
    X_test_ticker_list.append(Xts)
    y_test_price_list.append(Yps)
    y_test_dir_list.append(Yds)

if X_train_list:
    X_train_features  = np.concatenate(X_train_list, axis=0)
    X_train_tickers   = np.concatenate(X_train_ticker_list, axis=0)
    y_train_price = np.concatenate(y_train_price_list, axis=0)
    y_train_dir   = np.concatenate(y_train_dir_list, axis=0)
else:
    X_train_features, X_train_tickers, y_train_price, y_train_dir = (np.array([]),)*4

if X_test_list:
    X_test_features   = np.concatenate(X_test_list, axis=0)
    X_test_tickers    = np.concatenate(X_test_ticker_list, axis=0)
    y_test_price  = np.concatenate(y_test_price_list, axis=0)
    y_test_dir    = np.concatenate(y_test_dir_list, axis=0)
else:
    X_test_features, X_test_tickers, y_test_price, y_test_dir = (np.array([]),)*4

print("Train shapes:")
print("X_train_features:", X_train_features.shape)
print("X_train_tickers :", X_train_tickers.shape)
print("y_train_price   :", y_train_price.shape)
print("y_train_dir     :", y_train_dir.shape)

print("Test shapes:")
print("X_test_features :", X_test_features.shape)
print("X_test_tickers  :", X_test_tickers.shape)
print("y_test_price    :", y_test_price.shape)
print("y_test_dir      :", y_test_dir.shape)

# --------------------------------------------------------------------
# 8) สร้างสถาปัตยกรรมโมเดล Multi-Task (แยก Head: Price / Direction)
# --------------------------------------------------------------------
from tensorflow.keras.layers import LayerNormalization

from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add

def build_multi_task_model_v2(
    num_tickers,
    seq_length=10,
    num_feature=30,
    embedding_dim=32,
    lstm_units=128,
    dropout_rate=0.3
):
    features_input = Input(shape=(seq_length, num_feature), name='features_input')
    ticker_input   = Input(shape=(seq_length,), name='ticker_input')

    ticker_embedding = Embedding(input_dim=num_tickers, output_dim=embedding_dim, name='ticker_embedding')(ticker_input)

    x = concatenate([features_input, ticker_embedding], axis=-1)
    x = Masking(mask_value=0.0)(x)

    # ใช้ Multi-Head Attention
    attn_out = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = Add()([x, attn_out])  # Residual Connection
    x = LayerNormalization()(x)

    # Bi-GRU Layers
    x = Bidirectional(GRU(lstm_units, return_sequences=True, activation='relu'))(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Bidirectional(GRU(lstm_units, return_sequences=False, activation='relu'))(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)

    # Price Branch
    price_branch = Dense(64, activation='relu')(x)
    price_branch = Dropout(0.2)(price_branch)
    price_output = Dense(1, activation='linear', name='price_output')(price_branch)

    # Direction Branch
    direction_branch = Dense(64, activation='relu')(x)
    direction_branch = Dropout(0.2)(direction_branch)
    direction_output = Dense(1, activation='sigmoid', name='direction_output')(direction_branch)

    model = Model(inputs=[features_input, ticker_input], outputs=[price_output, direction_output])
    return model


# --------------------------------------------------------------------
# 9) Compile โมเดล (Huber + BinaryCrossentropy)
# --------------------------------------------------------------------
model_multi = build_multi_task_model_v2(
    num_tickers=num_tickers,
    seq_length=seq_length,
    num_feature=len(feature_columns),
    embedding_dim=32,
    lstm_units=128,
    dropout_rate=0.3
)

losses = {
    'price_output': tf.keras.losses.Huber(delta=0.5),  # ปรับให้ต่ำลงเพื่อให้โมเดลอ่อนไหวต่อ outliers น้อยลง
    'direction_output': tf.keras.losses.BinaryCrossentropy()
}

loss_weights = {
    'price_output': 0.8,
    'direction_output': 0.2
}
@register_keras_serializable()
def focal_loss_fixed(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Custom Focal Loss
    """
    epsilon = 1e-8
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    loss = -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt))
    return loss

from tensorflow.keras.optimizers import AdamW
model_multi.compile(
    optimizer=AdamW(learning_rate=0.0003, weight_decay=1e-4),
    loss=losses,
    loss_weights=loss_weights,
    metrics={
      'price_output': ['mae'],
      'direction_output': ['accuracy']
    }
)


model_multi.summary()

# --------------------------------------------------------------------
# 10) Train Model พร้อม Callback
# --------------------------------------------------------------------
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    'best_multi_task_model.keras', 
    monitor='val_loss', 
    save_best_only=True, 
    mode='min'
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)

history = model_multi.fit(
    [X_train_features, X_train_tickers],
    {
      'price_output': y_train_price,
      'direction_output': y_train_dir
    },
    epochs=100,
    batch_size=32,
    shuffle=False,
    validation_split=0.1,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

plot_training_history(history)

# เซฟโมเดล (หากต้องการเซฟไฟล์สุดท้ายด้วย)
model_multi.save('multi_task_GRU_model_embedding.keras')

# --------------------------------------------------------------------
# 11) Walk-Forward Validation (Multi-Task)
# --------------------------------------------------------------------
def walk_forward_validation_multi_task(
    model, df, feature_columns, scaler_features, scaler_target, 
    ticker_encoder, seq_length=10, retrain_frequency=1
):
    """
    ทำ Walk-Forward Validation แบบ Multi-Task (Price + Direction)
    และ Online Learning ทุก ๆ retrain_frequency ก้าว
    
    Args:
        model: โมเดล Multi-Output ที่ผ่านการเทรนแล้ว
        df: DataFrame ทดสอบ (ควรเป็น test_df หรือชุดไม่เคยเห็น)
        feature_columns: ชื่อฟีเจอร์
        scaler_features: Scaler ของฟีเจอร์
        scaler_target: Scaler ของ Price
        ticker_encoder: LabelEncoder ของ Ticker
        seq_length: ความยาว sequence
        retrain_frequency: ระยะที่ Online Learning อีกครั้ง
    """
    all_predictions = []
    tickers = df['Ticker'].unique()
    
    for ticker in tickers:
        print(f"\nProcessing Ticker: {ticker}")
        ticker_id_val = ticker_encoder.transform([ticker])[0]
        df_ticker = df[df['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)
        
        if len(df_ticker) < seq_length + 1:
            print(f"Not enough data for ticker {ticker}, skipping...")
            continue
        
        for i in range(len(df_ticker) - seq_length):
            historical_data = df_ticker.iloc[i : i + seq_length]
            target_data = df_ticker.iloc[i + seq_length]
            
            # เตรียม X
            features = historical_data[feature_columns].values
            ticker_ids = historical_data['Ticker_ID'].values
            
            features_scaled = scaler_features.transform(features)
            
            X_features = features_scaled.reshape(1, seq_length, len(feature_columns))
            X_ticker   = ticker_ids.reshape(1, seq_length)
            
            # ทำนาย (ได้ 2 Output)
            pred_price_scaled, pred_dir_prob = model.predict([X_features, X_ticker], verbose=0)
            
            # inverse scale ราคากลับ
            predicted_price = scaler_target.inverse_transform(pred_price_scaled.reshape(-1,1))[0][0]
            # direction: ถ้า pred_dir_prob >= 0.5 => "Up"
            predicted_dir = 1 if pred_dir_prob[0][0] >= 0.5 else 0
            
            actual_price = target_data['Close']
            future_date  = target_data['Date']
            actual_dir   = 1 if (target_data['Close'] > historical_data.iloc[-1]['Close']) else 0
            
            # หา last_close เพื่อเช็คขึ้น/ลงเป็น string
            last_close = historical_data.iloc[-1]['Close']
            predicted_direction_str = "Up" if predicted_dir == 1 else "Down"
            actual_direction_str    = "Up" if actual_dir == 1 else "Down"
            
            all_predictions.append({
                'Ticker': ticker,
                'Date': future_date,
                'Predicted_Price': predicted_price,
                'Actual_Price': actual_price,
                'Predicted_Dir': predicted_dir,
                'Actual_Dir': actual_dir,
                'Predicted_Dir_Str': predicted_direction_str,
                'Actual_Dir_Str': actual_direction_str
            })
            
            # Online Learning
            if i % retrain_frequency == 0:
                # เตรียม Target จริง 2 ส่วน
                # ราคา => actual_price (reshape -> scaled)
                y_price_true_scaled = scaler_target.transform(np.array([[actual_price]]))
                # ทิศทาง => actual_dir
                y_dir_true = np.array([actual_dir])  # shape(1,)
                
                # fit 1 step
                model.fit(
                    [X_features, X_ticker],
                    {
                        'price_output': y_price_true_scaled,
                        'direction_output': y_dir_true
                    },
                    epochs=1,
                    batch_size=1,
                    verbose=0
                )
            
            if i % 100 == 0:
                print(f"  Processing: {i} / {len(df_ticker)-seq_length}")
    
    predictions_df = pd.DataFrame(all_predictions)
    
    # คำนวณ Metrics แยก Ticker
    metrics_dict = {}
    for ticker, group in predictions_df.groupby('Ticker'):
        actual_prices = group['Actual_Price'].values
        pred_prices   = group['Predicted_Price'].values
        
        actual_dirs = group['Actual_Dir'].values
        pred_dirs   = group['Predicted_Dir'].values
        
        mae_val  = mean_absolute_error(actual_prices, pred_prices)
        mse_val  = mean_squared_error(actual_prices, pred_prices)
        rmse_val = np.sqrt(mse_val)
        mape_val = custom_mape(actual_prices, pred_prices)
        smape_val= smape(actual_prices, pred_prices)
        r2_val   = r2_score(actual_prices, pred_prices)
        
        dir_acc  = accuracy_score(actual_dirs, pred_dirs)
        
        metrics_dict[ticker] = {
            'MAE': mae_val,
            'MSE': mse_val,
            'RMSE': rmse_val,
            'MAPE': mape_val,
            'SMAPE': smape_val,
            'R2 Score': r2_val,
            'Direction Accuracy': dir_acc,
            'Dates': group['Date'].tolist(),
            'Actual Prices': actual_prices.tolist(),
            'Predicted Prices': pred_prices.tolist(),
            'Actual Dirs': actual_dirs.tolist(),
            'Predicted Dirs': pred_dirs.tolist()
        }
    
    predictions_df.to_csv('predictions_multi_task_walkforward.csv', index=False)
    print("\n✅ Saved predictions to 'predictions_multi_task_walkforward.csv'")
    
    return predictions_df, metrics_dict

# --------------------------------------------------------------------
# 12) เรียกใช้งาน Walk-Forward Validation
# --------------------------------------------------------------------
# โหลดโมเดลที่ดีที่สุดจาก checkpoint (best_multi_task_model.h5)
best_multi_model = tf.keras.models.load_model('best_multi_task_model.keras')


predictions_df, results_per_ticker = walk_forward_validation_multi_task(
    model=best_multi_model,
    df=test_df,
    feature_columns=feature_columns,
    scaler_features=scaler_features_dict,   # เปลี่ยนให้ตรงกับพารามิเตอร์ของฟังก์ชัน
    scaler_target=scaler_price_dict,        # เปลี่ยนให้ตรงกับพารามิเตอร์ของฟังก์ชัน
    ticker_encoder=ticker_encoder,
    seq_length=seq_length,
    retrain_frequency=20
)


# แสดง Metrics แยก Ticker
for ticker, metrics in results_per_ticker.items():
    print(f"\nMetrics for {ticker}:")
    print(f"  MAE   : {metrics['MAE']:.4f}")
    print(f"  MSE   : {metrics['MSE']:.4f}")
    print(f"  RMSE  : {metrics['RMSE']:.4f}")
    print(f"  MAPE  : {metrics['MAPE']:.4f}")
    print(f"  SMAPE : {metrics['SMAPE']:.4f}")
    print(f"  R2    : {metrics['R2 Score']:.4f}")
    print(f"  Direction Accuracy : {metrics['Direction Accuracy']:.4f}")

# --------------------------------------------------------------------
# 13) เซฟ Metrics ลงไฟล์
# --------------------------------------------------------------------
metrics_df = pd.DataFrame.from_dict(results_per_ticker, orient='index')
metrics_df.to_csv('metrics_per_ticker_multi_task.csv', index=True)
print("\nSaved metrics per ticker to 'metrics_per_ticker_multi_task.csv'")
