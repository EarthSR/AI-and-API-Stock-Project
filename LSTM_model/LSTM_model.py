import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Embedding, concatenate, Bidirectional, Layer, Masking, Conv1D, Flatten, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber, MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import CosineDecay, ExponentialDecay
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import ta
import matplotlib.pyplot as plt
import joblib
import logging
from tensorflow.keras.losses import Loss
from tensorflow.keras.saving import register_keras_serializable

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
print("TensorFlow running on:", tf.config.list_physical_devices())

# ------------------------------------------------------------------------------------
# ฟังก์ชันคำนวณ Error
# ------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------
# ตั้งค่า Logging
# ------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, filename='training.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------------------------------
# ฟังก์ชันสร้าง Sequence (ปรับให้รับ Target 2 ส่วน: price กับ direction)
# ------------------------------------------------------------------------------------
def create_sequences_for_ticker(features, ticker_ids, targets_price, targets_dir, seq_length=10):
    """
    คืนค่า 4 รายการ: 
      X_features, X_tickers, Y_price, Y_dir
    สำหรับแต่ละ Sequence ยาว seq_length
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

# ------------------------------------------------------------------------------------
# ฟังก์ชัน Plot History
# ------------------------------------------------------------------------------------
def plot_training_history(history):
    """
    แสดงกราฟ Loss รวม (จาก Multi-Output) และ Loss แยก, Metric ของแต่ละ Output
    """
    # โดยเริ่มต้น Keras จะรวม loss = weighted sum ของทั้ง 2 outputs
    # ใน history.history จะมี key ต่าง ๆ เช่น
    #   'loss', 'price_output_loss', 'direction_output_loss', 
    #   'price_output_mae', 'direction_output_accuracy', 'val_loss', ...
    keys = list(history.history.keys())
    print("Keys in history:", keys)
    
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Total Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Total Loss')
    plt.plot(history.history['val_loss'], label='Val Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Subplot 2: Price Output Loss
    # ถ้ามี price_output_loss ใน history
    if 'price_output_loss' in history.history:
        plt.figure()
        plt.plot(history.history['price_output_loss'], label='Train Price Loss')
        plt.plot(history.history['val_price_output_loss'], label='Val Price Loss')
        plt.title('Price Output Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_price_loss.png')
        plt.close()

    # Subplot 3: Direction Output Loss
    if 'direction_output_loss' in history.history:
        plt.figure()
        plt.plot(history.history['direction_output_loss'], label='Train Dir Loss')
        plt.plot(history.history['val_direction_output_loss'], label='Val Dir Loss')
        plt.title('Direction Output Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_dir_loss.png')
        plt.close()

    # Subplot 4: Price Output MAE
    if 'price_output_mae' in history.history:
        plt.figure()
        plt.plot(history.history['price_output_mae'], label='Train Price MAE')
        plt.plot(history.history['val_price_output_mae'], label='Val Price MAE')
        plt.title('Price Output MAE')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_price_mae.png')
        plt.close()

    # Subplot 5: Direction Output Accuracy
    if 'direction_output_accuracy' in history.history:
        plt.figure()
        plt.plot(history.history['direction_output_accuracy'], label='Train Dir Accuracy')
        plt.plot(history.history['val_direction_output_accuracy'], label='Val Dir Accuracy')
        plt.title('Direction Output Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_dir_acc.png')
        plt.close()

    # บันทึกเฉพาะ subplot แรกสุด
    plt.tight_layout()
    plt.savefig('training_total_loss.png')
    plt.close()


def plot_predictions(y_true, y_pred, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='red', alpha=0.7)
    plt.title(f'True vs Predicted Prices for {ticker}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'predictions_{ticker}.png')
    plt.close()

def plot_residuals(y_true, y_pred, ticker):
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(residuals)), residuals, alpha=0.5)
    plt.hlines(y=0, xmin=0, xmax=len(residuals), colors='red')
    plt.title(f'Residuals for {ticker}')
    plt.xlabel('Sample')
    plt.ylabel('Residual')
    plt.tight_layout()
    plt.savefig(f'residuals_{ticker}.png')
    plt.close()

# ------------------------------------------------------------------------------------
# ฟังก์ชัน Loss แบบ Cosine Similarity (หากต้องการ)
# ------------------------------------------------------------------------------------
def cosine_similarity_loss(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return 1 - K.sum(y_true * y_pred, axis=-1)

# ลงทะเบียน Loss
tf.keras.utils.get_custom_objects()["cosine_similarity_loss"] = cosine_similarity_loss

# ตรวจสอบ GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logging.info(f"Memory growth enabled for GPU: {physical_devices[0]}")
        print("Memory growth enabled for GPU:", physical_devices[0])
    except Exception as e:
        logging.error(f"Failed to set memory growth: {e}")
        print(f"Error setting memory growth: {e}")
else:
    logging.info("GPU not found, using CPU")
    print("GPU not found, using CPU")

# ------------------------------------------------------------------------------------
# 1) โหลดและเตรียมข้อมูล
# ------------------------------------------------------------------------------------
df = pd.read_csv('../merged_stock_sentiment_financial.csv')

# แปลง Sentiment
df['Sentiment'] = df['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})

# สร้างฟีเจอร์เพิ่มเติม
df['Change'] = df['Close'] - df['Open']
df['Change (%)'] = df['Close'].pct_change() * 100
upper_bound = df["Change (%)"].quantile(0.99)
lower_bound = df["Change (%)"].quantile(0.01)
df["Change (%)"] = np.clip(df["Change (%)"], lower_bound, upper_bound)

import ta
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
df['Bollinger_Low'] = bollinger.bollinger_lband()

financial_columns = [
    'Total Revenue', 'QoQ Growth (%)', 'YoY Growth (%)', 'Net Profit', 
    'Earnings Per Share (EPS)', 'ROA (%)', 'ROE (%)', 'Gross Margin (%)', 
    'Net Profit Margin (%)', 'Debt to Equity ', 'P/E Ratio ', 
    'P/BV Ratio ', 'Dividend Yield (%)'
]
df_financial = df[['Date', 'Ticker'] + financial_columns].drop_duplicates()
df_financial[financial_columns] = df_financial[financial_columns].where(df_financial[financial_columns].ne(0)).bfill()

stock_columns = [
    'RSI', 'EMA_10', 'EMA_20', 'MACD', 'MACD_Signal',
    'Bollinger_High', 'Bollinger_Low', 'SMA_50', 'SMA_200'
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

# สร้าง Direction = 1 ถ้าพรุ่งนี้ราคาสูงขึ้น, 0 ถ้าลงหรือนิ่ง
df['Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# สร้าง TargetPrice = ราคาปิดวันถัดไป
df['TargetPrice'] = df['Close'].shift(-1)

# ลบแถวที่มี NaN ใน Target
df.dropna(subset=['Direction', 'TargetPrice'], inplace=True)

# Label Encode Ticker
ticker_encoder = LabelEncoder()
df['Ticker_ID'] = ticker_encoder.fit_transform(df['Ticker'])
num_tickers = len(ticker_encoder.classes_)

# แยก Train / Test
sorted_dates = df['Date'].unique()
train_cutoff = sorted_dates[int(len(sorted_dates) * 6 / 7)]
train_df = df[df['Date'] <= train_cutoff].copy()
test_df  = df[df['Date'] > train_cutoff].copy()

train_df.to_csv('train_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)
print("Train cutoff:", train_cutoff)
print("First date in train set:", train_df['Date'].min())
print("Last date in train set:", train_df['Date'].max())

# ------------------------------------------------------------------------------------
# 2) เตรียม Target ทั้ง 2: price กับ direction
# ------------------------------------------------------------------------------------
train_features = train_df[feature_columns].values
test_features  = test_df[feature_columns].values

train_ticker_id = train_df['Ticker_ID'].values
test_ticker_id  = test_df['Ticker_ID'].values

train_price = train_df['TargetPrice'].values.reshape(-1, 1)
test_price  = test_df['TargetPrice'].values.reshape(-1, 1)

train_dir = train_df['Direction'].values  # 1D
test_dir  = test_df['Direction'].values   # 1D

# สเกล Feature
scaler_features = RobustScaler(quantile_range=(5, 95))
train_features_scaled = scaler_features.fit_transform(train_features)
test_features_scaled  = scaler_features.transform(test_features)

# สเกล Price
scaler_target = RobustScaler(quantile_range=(5, 95))
train_price_scaled = scaler_target.fit_transform(train_price)
test_price_scaled  = scaler_target.transform(test_price)

joblib.dump(scaler_features, 'scaler_features.pkl')
joblib.dump(scaler_target, 'scaler_target.pkl')

np.save('test_features.npy', test_features_scaled)
np.save('test_price.npy', test_price_scaled)
print("✅ บันทึก test_features.npy และ test_price.npy สำเร็จ!")

seq_length = 10

# ------------------------------------------------------------------------------------
# 3) สร้าง Sequence (ต่อ Ticker) สำหรับ Multi-Task (Price + Direction)
# ------------------------------------------------------------------------------------
X_train_list, X_train_ticker_list = [], []
y_price_train_list, y_dir_train_list = [], []

X_test_list, X_test_ticker_list = [], []
y_price_test_list, y_dir_test_list = [], []

for t_id in range(num_tickers):
    df_train_ticker = train_df[train_df['Ticker_ID'] == t_id]
    if len(df_train_ticker) > seq_length:
        idx_train = df_train_ticker.index
        mask_train = np.isin(train_df.index, idx_train)
        
        f_t = train_features_scaled[mask_train]
        t_t = train_ticker_id[mask_train]
        p_t = train_price_scaled[mask_train]
        d_t = train_dir[mask_train]
        
        (Xf, Xt, Yp, Yd) = create_sequences_for_ticker(
            f_t, t_t, p_t, d_t, seq_length
        )
        X_train_list.append(Xf)
        X_train_ticker_list.append(Xt)
        y_price_train_list.append(Yp)
        y_dir_train_list.append(Yd)

    df_test_ticker = test_df[test_df['Ticker_ID'] == t_id]
    if len(df_test_ticker) > seq_length:
        idx_test = df_test_ticker.index
        mask_test = np.isin(test_df.index, idx_test)

        f_s = test_features_scaled[mask_test]
        t_s = test_ticker_id[mask_test]
        p_s = test_price_scaled[mask_test]
        d_s = test_dir[mask_test]

        (Xs, Xts, Yps, Yds) = create_sequences_for_ticker(
            f_s, t_s, p_s, d_s, seq_length
        )
        X_test_list.append(Xs)
        X_test_ticker_list.append(Xts)
        y_price_test_list.append(Yps)
        y_dir_test_list.append(Yds)

# รวมเป็นอาเรย์ใหญ่
if len(X_train_list) > 0:
    X_price_train = np.concatenate(X_train_list, axis=0)
    X_ticker_train = np.concatenate(X_train_ticker_list, axis=0)
    y_price_train = np.concatenate(y_price_train_list, axis=0)
    y_dir_train   = np.concatenate(y_dir_train_list, axis=0)
else:
    X_price_train, X_ticker_train, y_price_train, y_dir_train = (np.array([]),)*4

if len(X_test_list) > 0:
    X_price_test = np.concatenate(X_test_list, axis=0)
    X_ticker_test = np.concatenate(X_test_ticker_list, axis=0)
    y_price_test = np.concatenate(y_price_test_list, axis=0)
    y_dir_test   = np.concatenate(y_dir_test_list, axis=0)
else:
    X_price_test, X_ticker_test, y_price_test, y_dir_test = (np.array([]),)*4

print("X_price_train shape :", X_price_train.shape)
print("X_ticker_train shape:", X_ticker_train.shape)
print("y_price_train shape :", y_price_train.shape)
print("y_dir_train shape   :", y_dir_train.shape)

num_feature = train_features_scaled.shape[1]
# ลดขนาด LSTM และเพิ่ม Attention Layer
features_input = Input(shape=(seq_length, num_feature), name='features_input')
ticker_input   = Input(shape=(seq_length,), name='ticker_input')

embedding_dim = 16
ticker_embedding = Embedding(input_dim=num_tickers, output_dim=embedding_dim, name='ticker_embedding')(ticker_input)

# รวม features + ticker_embedding
merged = concatenate([features_input, ticker_embedding], axis=-1)

# LSTM Layer ลดขนาดลง
x = LSTM(64, return_sequences=True)(merged)
x = Dropout(0.3)(x)
x = LSTM(32, return_sequences=True)(x)
x = Dropout(0.3)(x)
x = LSTM(16, return_sequences=True)(x)
x = Dropout(0.3)(x)

# **เพิ่ม Attention Layer**
attention_output = Attention()([x, x])
x = concatenate([x, attention_output])
x = Flatten()(x)  # แปลงข้อมูลให้เป็น 1D
x = BatchNormalization()(x)

# 2 Outputs
price_output = Dense(1, name='price_output')(x)
direction_output = Dense(1, activation='sigmoid', name='direction_output')(x)

# สร้างโมเดล
model_multi = Model(inputs=[features_input, ticker_input], outputs=[price_output, direction_output])

# **Focal Loss สำหรับ Direction**
def focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-8
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt))
        return loss
    return focal_loss_fixed

# เปลี่ยน Optimizer เป็น AdamW
from tensorflow.keras.optimizers import AdamW
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
model_multi.compile(
    optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-5),
    loss={'price_output': MeanSquaredError(), 'direction_output': focal_loss()},
    loss_weights={'price_output': 0.7, 'direction_output': 0.3},
    metrics={'price_output': ['mae'], 'direction_output': ['accuracy']}
)

model_multi.summary()

logging.info("เริ่มฝึกโมเดล Multi-Task (Price & Direction) - รุ่นปรับปรุง")

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_multi_task_model.keras', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

history = model_multi.fit(
    [X_price_train, X_ticker_train],
    {
        'price_output': y_price_train,
        'direction_output': y_dir_train
    },
    epochs=100,
    batch_size=32,
    verbose=1,
    shuffle=True,
    validation_split=0.1,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

plot_training_history(history)  # เรียกฟังก์ชัน plot ของเดิมได้เลย

model_multi.save('multi_task_price_dir_model.keras')
logging.info("บันทึกโมเดล Multi-Task (รุ่นปรับปรุง) เรียบร้อยแล้ว")
# ------------------------------------------------------------------------------------
# 5) ฟังก์ชัน Walk-Forward Validation (Multi-Task) + Online Learning
# ------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------
# 6) เรียกใช้งาน Walk-Forward Validation สำหรับ Multi-Task
# ------------------------------------------------------------------------------------
best_multi_model = load_model("best_multi_task_model.h5", custom_objects={"focal_loss_fixed": focal_loss_fixed})


predictions_df, results_per_ticker = walk_forward_validation_multi_task(
    model = best_multi_model,
    df = test_df,
    feature_columns = feature_columns,
    scaler_features = scaler_features,
    scaler_target = scaler_target,
    ticker_encoder = ticker_encoder,
    seq_length = seq_length,
    retrain_frequency = 20
)

# แสดงผล Metrics
for ticker, metrics in results_per_ticker.items():
    print(f"\nMetrics for {ticker}:")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  MSE:  {metrics['MSE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.4f}")
    print(f"  SMAPE:{metrics['SMAPE']:.4f}")
    print(f"  R2 Score: {metrics['R2 Score']:.4f}")
    print(f"  Direction Accuracy: {metrics['Direction Accuracy']:.4f}")

# เซฟ Metrics ราย Ticker ลง CSV
selected_columns = [
    'MAE','MSE','RMSE','MAPE','SMAPE','R2 Score','Direction Accuracy'
]
metrics_df = pd.DataFrame.from_dict(results_per_ticker, orient='index')
metrics_df.to_csv('metrics_per_ticker_multi_task.csv', index=True)
print("\nSaved metrics per ticker to 'metrics_per_ticker_multi_task.csv'")

# เซฟข้อมูล Actual vs Predicted รวม
all_data = []
for ticker, data in results_per_ticker.items():
    for date_val, actual_p, pred_p, actual_d, pred_d in zip(
        data['Dates'], data['Actual Prices'], data['Predicted Prices'], 
        data['Actual Dirs'], data['Predicted Dirs']
    ):
        all_data.append([ticker, date_val, actual_p, pred_p, actual_d, pred_d])
prediction_df = pd.DataFrame(all_data, columns=[
    'Ticker','Date','Actual_Price','Predicted_Price','Actual_Dir','Predicted_Dir'
])
prediction_df.to_csv('all_predictions_per_day_multi_task.csv', index=False)
print("Saved actual and predicted (price & direction) to 'all_predictions_per_day_multi_task.csv'")
