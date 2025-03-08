import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Embedding, concatenate, Bidirectional, Layer, Masking, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import CosineDecay, ExponentialDecay
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import ta
import matplotlib.pyplot as plt
import joblib
import logging


# ------------------------------------------------------------------------------------
# ส่วนนี้ประกาศฟังก์ชัน custom_mape() และ smape() ถ้ายังไม่มีในโค้ด
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

def create_sequences_for_ticker(features, ticker_ids, targets, seq_length=10):
    X_features, X_tickers, Y = [], [], []
    for i in range(len(features) - seq_length):
        X_features.append(features[i:i+seq_length])
        X_tickers.append(ticker_ids[i:i+seq_length])  # sequence ของ ticker_id
        Y.append(targets[i+seq_length])
    return np.array(X_features), np.array(X_tickers), np.array(Y)

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    # กราฟ Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # กราฟ MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE During Training')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')

def plot_predictions(y_true, y_pred, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='red', alpha=0.7)
    plt.title(f'True vs Predicted Prices for {ticker}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'predictions_{ticker}.png')  # บันทึกกราฟแยกแต่ละหุ้น
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
    plt.savefig(f'residuals_{ticker}.png')  # บันทึกกราฟแยกแต่ละหุ้น
    plt.close()

# ฟังก์ชัน Loss แบบ Cosine Similarity
def cosine_similarity_loss(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return 1 - K.sum(y_true * y_pred, axis=-1)

# ลงทะเบียน Loss ให้ TensorFlow
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
df['Sentiment'] = df['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})

# สร้างฟีเจอร์เพิ่มเติม
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
df['Bollinger_Low'] = bollinger.bollinger_lband()

# ข้อมูลงบการเงิน (ตรงนี้หากมี NaN หรือ 0 ก็ใช้วิธี bfill หรือ ffill ได้ตามต้องการ)
financial_columns = ['Total Revenue', 'QoQ Growth (%)', 'YoY Growth (%)', 'Net Profit', 
                     'Earnings Per Share (EPS)', 'ROA (%)', 'ROE (%)', 'Gross Margin (%)', 
                     'Net Profit Margin (%)', 'Debt to Equity ', 'P/E Ratio ', 
                     'P/BV Ratio ', 'Dividend Yield (%)']
df_financial = df[['Date', 'Ticker'] + financial_columns].drop_duplicates()
df_financial[financial_columns] = df_financial[financial_columns].where(df_financial[financial_columns].ne(0)).bfill()

# เติมค่าสำหรับฟีเจอร์เทคนิค
stock_columns = ['RSI', 'EMA_10', 'EMA_20', 'MACD', 'MACD_Signal', 
                 'Bollinger_High', 'Bollinger_Low','SMA_50', 'SMA_200']
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

# Label Encode Ticker
ticker_encoder = LabelEncoder()
df['Ticker_ID'] = ticker_encoder.fit_transform(df['Ticker'])
num_tickers = len(ticker_encoder.classes_)

# Train/Test Split จากวันที่
sorted_dates = df['Date'].unique()
train_cutoff = sorted_dates[int(len(sorted_dates) * 6 / 7)]
train_df = df[df['Date'] <= train_cutoff].copy()
test_df = df[df['Date'] > train_cutoff].copy()

train_df.to_csv('train_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)
print("Train cutoff:", train_cutoff)
print("First date in train set:", train_df['Date'].min())
print("Last date in train set:", train_df['Date'].max())

# ------------------------------------------------------------------------------------
# 2) เตรียม Target ให้เป็น "ราคาปิด (Close)" ของวันถัดไป (shift(-1))
# ------------------------------------------------------------------------------------
train_targets_price = train_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
train_df = train_df.iloc[:-1]
test_targets_price = test_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
test_df = test_df.iloc[:-1]

train_features = train_df[feature_columns].values
test_features = test_df[feature_columns].values

train_ticker_id = train_df['Ticker_ID'].values
test_ticker_id = test_df['Ticker_ID'].values

# สเกลข้อมูล
scaler_features = RobustScaler(quantile_range=(5, 95))
train_features_scaled = scaler_features.fit_transform(train_features)
test_features_scaled = scaler_features.transform(test_features)

scaler_target = RobustScaler(quantile_range=(5, 95))
train_targets_scaled = scaler_target.fit_transform(train_targets_price)
test_targets_scaled = scaler_target.transform(test_targets_price)

joblib.dump(scaler_features, 'scaler_features.pkl')
joblib.dump(scaler_target, 'scaler_target.pkl')

np.save('test_features.npy', test_features_scaled)
np.save('test_targets.npy', test_targets_scaled)
print("✅ บันทึก test_features.npy และ test_targets.npy สำเร็จ!")

seq_length = 10

# ------------------------------------------------------------------------------------
# 3) สร้าง Sequence สำหรับเทรนและทดสอบ (ต่อ Ticker)
# ------------------------------------------------------------------------------------
X_train_list, X_train_ticker_list, y_train_list = [], [], []
X_test_list, X_test_ticker_list, y_test_list = [], [], []

for t_id in range(num_tickers):
    # ส่วน Train
    df_train_ticker = train_df[train_df['Ticker_ID'] == t_id]
    if len(df_train_ticker) > seq_length:
        indices = df_train_ticker.index
        mask_train = np.isin(train_df.index, indices)
        f_t = train_features_scaled[mask_train]
        t_t = train_ticker_id[mask_train]
        target_t = train_targets_scaled[mask_train]
        X_t, X_ti, y_t = create_sequences_for_ticker(f_t, t_t, target_t, seq_length)
        X_train_list.append(X_t)
        X_train_ticker_list.append(X_ti)
        y_train_list.append(y_t)

    # ส่วน Test
    df_test_ticker = test_df[test_df['Ticker_ID'] == t_id]
    if len(df_test_ticker) > seq_length:
        indices = df_test_ticker.index
        mask_test = np.isin(test_df.index, indices)
        f_s = test_features_scaled[mask_test]
        t_s = test_ticker_id[mask_test]
        target_s = test_targets_scaled[mask_test]
        X_s, X_si, y_s = create_sequences_for_ticker(f_s, t_s, target_s, seq_length)
        X_test_list.append(X_s)
        X_test_ticker_list.append(X_si)
        y_test_list.append(y_s)

X_price_train = np.concatenate(X_train_list, axis=0) if X_train_list else np.array([])
X_ticker_train = np.concatenate(X_train_ticker_list, axis=0) if X_train_ticker_list else np.array([])
y_price_train = np.concatenate(y_train_list, axis=0) if y_train_list else np.array([])

X_price_test = np.concatenate(X_test_list, axis=0) if X_test_list else np.array([])
X_ticker_test = np.concatenate(X_test_ticker_list, axis=0) if X_test_ticker_list else np.array([])
y_price_test = np.concatenate(y_test_list, axis=0) if y_test_list else np.array([])

# ------------------------------------------------------------------------------------
# 4) สร้างโมเดลสำหรับทำนาย "ราคาปิด"
# ------------------------------------------------------------------------------------
embedding_dim = 32  
LSTM_units = 64  
dropout_rate = 0.2  
initial_learning_rate = 0.001  

# Learning Rate Scheduler
lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Input layer
num_feature = len(feature_columns)
features_input = Input(shape=(seq_length, num_feature), name='features_input')
ticker_input = Input(shape=(seq_length,), name='ticker_input')

# Embedding สำหรับ Ticker
ticker_embedding = Embedding(input_dim=num_tickers, output_dim=embedding_dim, name='ticker_embedding')(ticker_input)

# รวม Feature + Ticker Embedding
merged = concatenate([features_input, ticker_embedding], axis=-1)

# Masking (ป้องกัน Padding)
masked = Masking(mask_value=0.0)(merged)

# Conv1D Layer
x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(masked)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)

# Bi-LSTM Layer
x = Bidirectional(LSTM(LSTM_units, return_sequences=True, activation='relu'))(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)

x = Bidirectional(LSTM(LSTM_units // 2, activation='relu'))(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)

# Flatten
x = Flatten()(x)

# Dense Layer
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)

# Output = 1
output = Dense(1, activation='linear')(x)

model = Model(inputs=[features_input, ticker_input], outputs=output)
model.compile(optimizer=optimizer, loss=cosine_similarity_loss, metrics=['mae'])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_price_model.keras', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-5
)

logging.info("เริ่มฝึกโมเดลสำหรับราคาหุ้นรวม (ใช้ Embedding สำหรับ Ticker)")
history = model.fit(
    [X_price_train, X_ticker_train], y_price_train,
    epochs=1000,
    batch_size=32,
    verbose=1,
    shuffle=False,
    validation_split=0.1,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

plot_training_history(history)

model.save('price_prediction_LSTM_model_embedding.keras')
logging.info("บันทึกโมเดลราคาหุ้นรวมเรียบร้อยแล้ว")

# ------------------------------------------------------------------------------------
# 5) ฟังก์ชัน Walk-Forward Validation สำหรับทำนาย “ราคาปิด” และ Online Learning
# ------------------------------------------------------------------------------------
def walk_forward_validation(model, df, feature_columns, scaler_features, scaler_target, ticker_encoder, seq_length=10, retrain_frequency=20):
    """
    ทำ Walk-Forward Validation แบบแท่งต่อแท่ง (หรือวันต่อวัน) สำหรับแต่ละ Ticker
    พร้อม Online Learning ทุก ๆ retrain_frequency ก้าว

    Args:
        model: โมเดลที่ผ่านการเทรนแล้ว
        df: DataFrame ที่ต้องการทดสอบ (ควรเป็น test_df หรือชุดข้อมูลที่ไม่เคยเห็นมาก่อน)
        feature_columns: ชื่อฟีเจอร์
        scaler_features: ตัว Scaler ของฟีเจอร์
        scaler_target: ตัว Scaler ของ Target (ราคาปิด)
        ticker_encoder: LabelEncoder สำหรับ Ticker
        seq_length: ความยาว sequence
        retrain_frequency: ระยะที่ให้ Online Learning อีกครั้ง (เช่นทุก ๆ 20 แท่ง)
    """
    all_predictions = []
    tickers = df['Ticker'].unique()

    for ticker in tickers:
        print(f"\nProcessing Ticker: {ticker}")
        ticker_id = ticker_encoder.transform([ticker])[0]
        df_ticker = df[df['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)

        # ข้ามถ้า Data น้อยกว่า seq_length
        if len(df_ticker) < seq_length + 1:
            print(f"Not enough data for ticker {ticker}, skipping...")
            continue

        for i in range(len(df_ticker) - seq_length):
            historical_data = df_ticker.iloc[i:i+seq_length]      # 10 แท่ง (seq_length)
            target_data = df_ticker.iloc[i+seq_length]            # แท่งที่จะทำนาย (แท่งถัดไป)
            
            # เตรียม X สำหรับโมเดล
            features = historical_data[feature_columns].values
            ticker_ids = historical_data['Ticker_ID'].values
            features_scaled = scaler_features.transform(features)

            X_features = features_scaled.reshape(1, seq_length, len(feature_columns))
            X_ticker = ticker_ids.reshape(1, seq_length)

            # ทำนายราคาปิด (Scaled -> Inverse)
            predicted_scaled = model.predict([X_features, X_ticker], verbose=0)
            predicted_price = scaler_target.inverse_transform(predicted_scaled.reshape(-1, 1))[0][0]
            actual_price = target_data['Close']
            future_date = target_data['Date']

            # หา last_close เพื่อดูทิศทางว่าขึ้นหรือลง
            last_close = historical_data.iloc[-1]['Close']
            predicted_direction = "Up" if predicted_price >= last_close else "Down"
            actual_direction = "Up" if actual_price >= last_close else "Down"

            all_predictions.append({
                'Ticker': ticker,
                'Date': future_date,
                'Predicted_Price': predicted_price,
                'Actual_Price': actual_price,
                'Predicted Direction': predicted_direction,
                'Actual Direction': actual_direction
            })

            # Online Learning (retrain_frequency)
            if i % retrain_frequency == 0:
                # เอา Actual มาสอนต่อ 1 batch
                y_true_scaled = scaler_target.transform(np.array([[actual_price]]))
                model.fit([X_features, X_ticker], y_true_scaled, epochs=1, batch_size=4, verbose=0)

            if i % 100 == 0:
                print(f"  Processing: {i}/{len(df_ticker)-seq_length}")

    predictions_df = pd.DataFrame(all_predictions)

    # คำนวณ Metrics แยกตาม Ticker
    metrics_dict = {}
    for ticker, group in predictions_df.groupby('Ticker'):
        actuals = group['Actual_Price'].values
        preds = group['Predicted_Price'].values

        mae = mean_absolute_error(actuals, preds)
        mse = mean_squared_error(actuals, preds)
        rmse = np.sqrt(mse)
        mape = custom_mape(actuals, preds)
        r2 = r2_score(actuals, preds)
        smape_val = smape(actuals, preds)

        direction_accuracy = np.mean((group['Predicted Direction'] == group['Actual Direction']).astype(int))

        metrics_dict[ticker] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'SMAPE': smape_val,
            'R2 Score': r2,
            'Direction Accuracy': direction_accuracy,
            'Dates': group['Date'].tolist(),
            'Actuals': actuals.tolist(),
            'Predictions': preds.tolist(),
            'Predicted Directions': group['Predicted Direction'].tolist(),
            'Actual Directions': group['Actual Direction'].tolist()
        }

    predictions_df.to_csv('predictions_price_walkforward.csv', index=False)
    print("\n✅ Saved predictions to 'predictions_price_walkforward.csv'")
    return predictions_df, metrics_dict

# ------------------------------------------------------------------------------------
# 6) ทดสอบ Walk-Forward Validation ด้วยโมเดลที่เพิ่งเซฟไป
#    (ตัวอย่างเรียกจาก price_prediction_LSTM_model_embedding.keras)
# ------------------------------------------------------------------------------------
best_model = load_model('price_prediction_LSTM_model_embedding.keras',
                        custom_objects={'cosine_similarity_loss': cosine_similarity_loss})

predictions_df, results_per_ticker = walk_forward_validation(
    model = best_model,
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
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"MSE:  {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.4f}")
    print(f"SMAPE:{metrics['SMAPE']:.4f}")
    print(f"R2 Score: {metrics['R2 Score']:.4f}")
    print(f"Direction Accuracy: {metrics['Direction Accuracy']:.4f}")

# บันทึก Metrics ราย Ticker ลง CSV
selected_columns = ['MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE', 'R2 Score', 'Direction Accuracy']
metrics_df = pd.DataFrame.from_dict(results_per_ticker, orient='index')
metrics_df.to_csv('metrics_per_ticker_price.csv', index=True)
print("\nSaved metrics per ticker to 'metrics_per_ticker_price.csv'")

# บันทึกข้อมูล Actual vs Predicted รวมกัน
all_data = []
for ticker, data in results_per_ticker.items():
    for date_val, actual_val, pred_val in zip(data['Dates'], data['Actuals'], data['Predictions']):
        all_data.append([ticker, date_val, actual_val, pred_val])
prediction_df = pd.DataFrame(all_data, columns=['Ticker', 'Date', 'Actual_Price', 'Predicted_Price'])
prediction_df.to_csv('all_predictions_per_day_price.csv', index=False)
print("Saved actual and predicted prices to 'all_predictions_per_day_price.csv'")
