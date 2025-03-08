import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, BatchNormalization, Embedding, concatenate, Bidirectional, Masking, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import ta
import matplotlib.pyplot as plt
import joblib
import logging

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
# 2) ฟังก์ชันสร้าง Sequence
# --------------------------------------------------------------------
def create_sequences_for_ticker(features, ticker_ids, targets, seq_length=10):
    """
    features, ticker_ids, targets ต้องมีความยาวเท่ากันก่อนเรียกฟังก์ชันนี้
    """
    X_features, X_tickers, Y = [], [], []
    for i in range(len(features) - seq_length):
        X_features.append(features[i:i+seq_length])
        X_tickers.append(ticker_ids[i:i+seq_length])
        Y.append(targets[i+seq_length])  # ถ้า i+seq_length เกิน bounds จะ IndexError ทันที
    return np.array(X_features), np.array(X_tickers), np.array(Y)

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE During Training')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')

# --------------------------------------------------------------------
# 3) Custom Loss: Cosine Similarity
# --------------------------------------------------------------------
def cosine_similarity_loss(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return 1 - K.sum(y_true * y_pred, axis=-1)

tf.keras.utils.get_custom_objects()["cosine_similarity_loss"] = cosine_similarity_loss

logging.basicConfig(level=logging.INFO, filename='training.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')


# --------------------------------------------------------------------
# 4) ตรวจสอบ GPU
# --------------------------------------------------------------------
physical_devices = tf.config.list_physical_devices('GPU')
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
# 5) โหลดข้อมูล & เตรียมฟีเจอร์
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

ticker_encoder = LabelEncoder()
df['Ticker_ID'] = ticker_encoder.fit_transform(df['Ticker'])
num_tickers = len(ticker_encoder.classes_)

# --------------------------------------------------------------------
# 6) แบ่ง Train/Test ตามวันที่
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

# --------------------------------------------------------------------
# 7) สร้าง target เป็นราคาปิดวันถัดไป (shift(-1))
#    และบังคับให้ train_features / train_targets มี len เท่ากัน
# --------------------------------------------------------------------
train_targets_price = train_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
train_df = train_df.iloc[:-1]
test_targets_price  = test_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
test_df  = test_df.iloc[:-1]

train_features = train_df[feature_columns].values
test_features  = test_df[feature_columns].values

# --- เช็คว่าเท่ากันไหม ---
print(">> Before alignment:")
print("  train_features length:", len(train_features))
print("  train_targets length:", len(train_targets_price))
if len(train_features) > len(train_targets_price):
    train_features = train_features[:len(train_targets_price)]  # ตัด feature ท้าย
elif len(train_targets_price) > len(train_features):
    train_targets_price = train_targets_price[:len(train_features)]

print("  After fix, train_features length:", len(train_features))
print("  After fix, train_targets length:", len(train_targets_price))

# สำหรับ test
print(">> Before alignment (test):")
print("  test_features length:", len(test_features))
print("  test_targets length:", len(test_targets_price))
if len(test_features) > len(test_targets_price):
    test_features = test_features[:len(test_targets_price)]
elif len(test_targets_price) > len(test_features):
    test_targets_price = test_targets_price[:len(test_features)]

print("  After fix, test_features length:", len(test_features))
print("  After fix, test_targets length:", len(test_targets_price))

# (Option) เซฟไฟล์ .npy ถ้าต้องการ
np.save('train_features_price.npy', train_features)
np.save('test_features_price.npy', test_features)
np.save('train_targets_price.npy', train_targets_price)
np.save('test_targets_price.npy', test_targets_price)

seq_length = 10

# --------------------------------------------------------------------
# 8) สร้าง Scaler แยกต่อ Ticker
# --------------------------------------------------------------------
scaler_features_dict = {}
scaler_target_dict   = {}

train_dict = {}
test_dict  = {}

for t_id in range(num_tickers):
    df_train_ticker = train_df[train_df['Ticker_ID'] == t_id].copy()
    if len(df_train_ticker) <= seq_length:
        continue

    # shift(-1) ใน DataFrame ของ Ticker นั้น
    y_train = df_train_ticker['Close'].shift(-1).dropna().values.reshape(-1,1)
    df_train_ticker = df_train_ticker.iloc[:-1]  # ตัดท้าย 1 แถว

    # ตรวจสอบความยาว
    f_t = df_train_ticker[feature_columns].values
    if len(f_t) > len(y_train):
        f_t = f_t[:len(y_train)]
    elif len(y_train) > len(f_t):
        y_train = y_train[:len(f_t)]

    # Fit Scaler
    sc_f = RobustScaler(quantile_range=(5,95))
    sc_y = RobustScaler(quantile_range=(5,95))

    sc_f.fit(f_t)
    sc_y.fit(y_train)

    scaler_features_dict[t_id] = sc_f
    scaler_target_dict[t_id]   = sc_y
    # เก็บ df_train_ticker (ต้องตัดท้ายเหมือนกัน)
    df_train_ticker = df_train_ticker.iloc[:len(y_train)]
    train_dict[t_id] = df_train_ticker.copy()

# test_dict
for t_id in range(num_tickers):
    df_test_ticker = test_df[test_df['Ticker_ID'] == t_id].copy()
    if len(df_test_ticker) <= seq_length:
        continue

    df_test_ticker = df_test_ticker.iloc[:-1]  # ตัดท้าย 1 แถว
    test_dict[t_id] = df_test_ticker.copy()

# --------------------------------------------------------------------
# 9) สร้าง Sequence สำหรับ train/test
# --------------------------------------------------------------------
X_train_list, X_train_ticker_list, y_train_list = [], [], []
X_test_list,  X_test_ticker_list,  y_test_list  = [], [], []

for t_id in range(num_tickers):
    if t_id not in scaler_features_dict:
        continue

    sc_f = scaler_features_dict[t_id]
    sc_y = scaler_target_dict[t_id]

    df_train_ticker = train_dict.get(t_id, None)
    if df_train_ticker is not None:
        f_train = df_train_ticker[feature_columns].values
        y_train = df_train_ticker['Close'].shift(-1, fill_value=0).values.reshape(-1,1)
        y_train = y_train[:-1]  # ตัดท้าย 1 แถว (ตอน shift)
        if len(f_train) > len(y_train):
            f_train = f_train[:len(y_train)]
        elif len(y_train) > len(f_train):
            y_train = y_train[:len(f_train)]

        ticker_ids = df_train_ticker['Ticker_ID'].values
        ticker_ids = ticker_ids[:len(y_train)]  # กันพลาด

        f_train_scaled = sc_f.transform(f_train)
        y_train_scaled = sc_y.transform(y_train)

        # เรียกสร้าง sequence
        X_t, X_ti, y_t = create_sequences_for_ticker(f_train_scaled, ticker_ids, y_train_scaled, seq_length)
        X_train_list.append(X_t)
        X_train_ticker_list.append(X_ti)
        y_train_list.append(y_t)

    df_test_ticker = test_dict.get(t_id, None)
    if df_test_ticker is not None:
        f_test = df_test_ticker[feature_columns].values
        y_test = df_test_ticker['Close'].shift(-1, fill_value=0).values.reshape(-1,1)
        y_test = y_test[:-1]

        # ปรับความยาวให้เท่ากัน
        if len(f_test) > len(y_test):
            f_test = f_test[:len(y_test)]
        elif len(y_test) > len(f_test):
            y_test = y_test[:len(f_test)]

        ticker_ids = df_test_ticker['Ticker_ID'].values
        ticker_ids = ticker_ids[:len(y_test)]

        f_test_scaled = sc_f.transform(f_test)
        y_test_scaled = sc_y.transform(y_test)

        X_s, X_si, y_s = create_sequences_for_ticker(f_test_scaled, ticker_ids, y_test_scaled, seq_length)
        X_test_list.append(X_s)
        X_test_ticker_list.append(X_si)
        y_test_list.append(y_s)

X_price_train = np.concatenate(X_train_list, axis=0) if X_train_list else np.array([])
X_ticker_train = np.concatenate(X_train_ticker_list, axis=0) if X_train_ticker_list else np.array([])
y_price_train = np.concatenate(y_train_list, axis=0) if y_train_list else np.array([])

X_price_test = np.concatenate(X_test_list, axis=0) if X_test_list else np.array([])
X_ticker_test = np.concatenate(X_test_ticker_list, axis=0) if X_test_ticker_list else np.array([])
y_price_test = np.concatenate(y_test_list, axis=0) if y_test_list else np.array([])

print("Train shape:", X_price_train.shape, X_ticker_train.shape, y_price_train.shape)
print("Test shape: ", X_price_test.shape,  X_ticker_test.shape,  y_price_test.shape)

# --------------------------------------------------------------------
# 10) สร้างโมเดล (GRU + Ticker Embedding)
# --------------------------------------------------------------------
embedding_dim = 32  
GRU_units = 64  
dropout_rate = 0.2  
initial_learning_rate = 0.001  

lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

num_feature = len(feature_columns)
features_input = Input(shape=(seq_length, num_feature), name='features_input')
ticker_input   = Input(shape=(seq_length,),         name='ticker_input')

ticker_embedding = Embedding(input_dim=num_tickers, output_dim=embedding_dim, name='ticker_embedding')(ticker_input)
merged = concatenate([features_input, ticker_embedding], axis=-1)
masked = Masking(mask_value=0.0)(merged)

x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(masked)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)

x = Bidirectional(GRU(GRU_units, return_sequences=True, activation='relu'))(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)

x = Bidirectional(GRU(GRU_units // 2, activation='relu'))(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)

output = Dense(1, activation='linear')(x)

model = Model(inputs=[features_input, ticker_input], outputs=output)
model.compile(optimizer=optimizer, loss=cosine_similarity_loss, metrics=['mae'])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_price_model.keras', monitor='val_loss', save_best_only=True, mode='min')

history = model.fit(
    [X_price_train, X_ticker_train],
    y_price_train,
    epochs=100,
    batch_size=32,
    shuffle=False,
    validation_split=0.1,
    callbacks=[early_stopping, checkpoint]
)

plot_training_history(history)
model.save('price_prediction_GRU_model_embedding.keras')


# --------------------------------------------------------------------
# 11) ฟังก์ชัน Walk-Forward Validation
# --------------------------------------------------------------------
def walk_forward_validation(model, df, feature_columns, 
                            scaler_features_dict, scaler_target_dict, 
                            ticker_encoder, seq_length=10, retrain_frequency=20):
    all_predictions = []
    tickers = df['Ticker'].unique()

    for ticker in tickers:
        print(f"\nProcessing Ticker: {ticker}")
        t_id = ticker_encoder.transform([ticker])[0]

        if t_id not in scaler_features_dict:
            print(f"No scaler for {ticker}, skipping.")
            continue

        df_ticker = df[df['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)
        if len(df_ticker) < seq_length + 1:
            continue

        sc_f = scaler_features_dict[t_id]
        sc_y = scaler_target_dict[t_id]

        for i in range(len(df_ticker) - seq_length):
            historical_data = df_ticker.iloc[i:i+seq_length]
            target_data     = df_ticker.iloc[i+seq_length]

            features = historical_data[feature_columns].values
            features_scaled = sc_f.transform(features)

            X_features = features_scaled.reshape(1, seq_length, len(feature_columns))
            X_ticker = historical_data['Ticker_ID'].values.reshape(1, seq_length)

            # predict
            pred_scaled = model.predict([X_features, X_ticker], verbose=0)
            pred_log_price = sc_y.inverse_transform(pred_scaled.reshape(-1,1))[0][0]

            # ถ้าคุณไม่ได้ log1p ราคามาก่อน ก็ไม่ต้อง expm1 ตรงนี้
            # (โค้ดเดิมบางคนใช้ np.expm1 เพราะทำ log1p() )
            # สมมติไม่ใช้ log -> predicted_price = pred_log_price
            predicted_price = pred_log_price

            # actual
            actual_price = target_data['Close']  
            # ถ้าเคย log1p(actual_price) ก็ต้อง expm1 ตอนนี้เหมือนกัน
            # actual_price = np.expm1(actual_price)

            future_date = target_data['Date']
            last_close = historical_data.iloc[-1]['Close']
            # ถ้า log -> last_close = np.expm1(last_close)

            predicted_direction = "Up" if predicted_price >= last_close else "Down"
            actual_direction    = "Up" if actual_price   >= last_close else "Down"

            all_predictions.append({
                'Ticker': ticker,
                'Date': future_date,
                'Predicted_Price': predicted_price,
                'Actual_Price': actual_price,
                'Predicted Direction': predicted_direction,
                'Actual Direction': actual_direction
            })

            # online learning
            if i % retrain_frequency == 0:
                y_true_scaled = sc_y.transform([[actual_price]])
                model.fit([X_features, X_ticker], y_true_scaled, epochs=1, batch_size=4, verbose=0)

            if i % 100 == 0:
                print(f"  Step {i}/{len(df_ticker)-seq_length}")

    predictions_df = pd.DataFrame(all_predictions)

    # คำนวณ Metrics
    metrics_dict = {}
    for ticker, group in predictions_df.groupby('Ticker'):
        actuals = group['Actual_Price'].values
        preds   = group['Predicted_Price'].values

        mae  = mean_absolute_error(actuals, preds)
        mse  = mean_squared_error(actuals, preds)
        rmse = np.sqrt(mse)
        mape = custom_mape(actuals, preds)
        smape_val = smape(actuals, preds)
        r2   = r2_score(actuals, preds)
        direction_accuracy = np.mean((group['Predicted Direction'] == group['Actual Direction']).astype(int))

        metrics_dict[ticker] = {
            'MAE': mae, 'MSE': mse, 'RMSE': rmse,
            'MAPE': mape, 'SMAPE': smape_val,
            'R2 Score': r2,
            'Direction Accuracy': direction_accuracy,
            'Dates': group['Date'].tolist(),
            'Actuals': actuals.tolist(),
            'Predictions': preds.tolist()
        }

    predictions_df.to_csv('predictions_price_walkforward.csv', index=False)
    return predictions_df, metrics_dict


# --------------------------------------------------------------------
# 12) ทดสอบ Walk-Forward Validation
# --------------------------------------------------------------------
best_model = load_model('price_prediction_GRU_model_embedding.keras',
                        custom_objects={'cosine_similarity_loss': cosine_similarity_loss})

predictions_df, results_per_ticker = walk_forward_validation(
    model = best_model,
    df = test_df,  # ชุด Test
    feature_columns = feature_columns,
    scaler_features_dict = scaler_features_dict,
    scaler_target_dict   = scaler_target_dict,
    ticker_encoder       = ticker_encoder,
    seq_length           = seq_length,
    retrain_frequency    = 20
)

for ticker, metrics in results_per_ticker.items():
    print(f"\nMetrics for {ticker}:")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  MSE: {metrics['MSE']:.4f}")
    print(f"  RMSE:{metrics['RMSE']:.4f}")
    print(f"  MAPE:{metrics['MAPE']:.4f}")
    print(f"  SMAPE:{metrics['SMAPE']:.4f}")
    print(f"  R2:  {metrics['R2 Score']:.4f}")
    print(f"  Direction Accuracy: {metrics['Direction Accuracy']:.4f}")

metrics_df = pd.DataFrame.from_dict(results_per_ticker, orient='index')
metrics_df.to_csv('metrics_per_ticker_price.csv', index=True)

all_data = []
for ticker, data in results_per_ticker.items():
    for date_val, actual_val, pred_val in zip(data['Dates'], data['Actuals'], data['Predictions']):
        all_data.append([ticker, date_val, actual_val, pred_val])
prediction_df = pd.DataFrame(all_data, columns=['Ticker','Date','Actual_Price','Predicted_Price'])
prediction_df.to_csv('all_predictions_per_day_price.csv', index=False)
