import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, concatenate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
import ta
import logging
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.losses import MeanSquaredError

# ตั้งค่า logging
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
    plt.show()

def plot_predictions(y_true, y_pred, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='red', alpha=0.7)
    plt.title(f'True vs Predicted Prices for {ticker}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_residuals(y_true, y_pred, ticker):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(residuals)), residuals, alpha=0.5)
    plt.hlines(y=0, xmin=0, xmax=len(residuals), colors='red')
    plt.title(f'Residuals for {ticker}')
    plt.xlabel('Sample')
    plt.ylabel('Residual')
    plt.show()

# ตรวจสอบ GPU
# ตรวจสอบ GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    # Set visible devices to the first GPU (or any other specific one)
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    # Enable memory growth for the first GPU
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

# โหลดข้อมูล
df_stock = pd.read_csv("cleaned_data.csv", parse_dates=["Date"]).sort_values(by=["Ticker", "Date"])
df_news = pd.read_csv("news_with_sentiment_gpu.csv")
df_news['Date'] = pd.to_datetime(df_news['Date'], errors='coerce')
df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')
print(df_news['Date'].dtype)
print(df_stock['Date'].dtype)
df_news['Sentiment'] = df_news['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
df_news['Confidence'] = df_news['Confidence']
df = pd.merge(df_stock, df_news[['Date', 'Sentiment', 'Confidence']], on='Date', how='left')

# เติมค่าที่ขาดหายไป
df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

# เพิ่มฟีเจอร์
df['Change'] = df['Close'] - df['Open']
df['Change (%)'] = (df['Change'] / df['Open']) * 100

df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
df['RSI'].fillna(method='ffill', inplace=True)
df['RSI'].fillna(0, inplace=True)

df['SMA_5'] = df['Close'].rolling(window=5).mean()  # SMA 50 วัน
df['SMA_10'] = df['Close'].rolling(window=10).mean()  # SMA 200 วัน
# คำนวณ MACD ด้วย EMA 12 และ EMA 26
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
# คำนวณ MACD = EMA(12) - EMA(26)
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].rolling(window=9).mean()  
bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
df['Bollinger_High'] = bollinger.bollinger_hband()
df['Bollinger_Low'] = bollinger.bollinger_lband()

df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment',
                    'RSI', 'SMA_10', 'SMA_5', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low']
# feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment', 'Confidence']

# Label Encode Ticker
ticker_encoder = LabelEncoder()
df['Ticker_ID'] = ticker_encoder.fit_transform(df['Ticker'])
num_tickers = len(ticker_encoder.classes_)

# แบ่งข้อมูล Train/Val/Test ตามเวลา
# สมมติเราแบ่งตาม quantile ของวันที่ หรือกำหนดโดยตรง
sorted_dates = df['Date'].unique()
train_cutoff = sorted_dates[int(len(sorted_dates) * 6 / 7)]  # ขอบเขตที่ 6 ปี


# ข้อมูล train, test
train_df = df[df['Date'] <= train_cutoff].copy()
test_df = df[df['Date'] > train_cutoff].copy()

print("Train cutoff:", train_cutoff)
print("First date in train set:", train_df['Date'].min())
print("Last date in train set:", train_df['Date'].max())


# สร้าง target โดย shift(-1)
train_targets_price = train_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
train_df = train_df.iloc[:-1]

test_targets_price = test_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
test_df = test_df.iloc[:-1]

train_features = train_df[feature_columns].values
test_features = test_df[feature_columns].values

train_ticker_id = train_df['Ticker_ID'].values
test_ticker_id = test_df['Ticker_ID'].values

# ใช้ Robust Scaling สำหรับจัดการ outliers
scaler = RobustScaler()
numeric_columns_to_scale = ['Open', 'Close', 'High', 'Low', 'Volume']
df_stock[numeric_columns_to_scale] = scaler.fit_transform(df_stock[numeric_columns_to_scale])

# สเกลข้อมูลจากชุดฝึก (train) เท่านั้น
scaler_features = RobustScaler()
train_features_scaled = scaler_features.fit_transform(train_features)  # ใช้ fit_transform กับชุดฝึก
test_features_scaled = scaler_features.transform(test_features)  # ใช้ transform กับชุดทดสอบ

scaler_target = RobustScaler()
train_targets_scaled = scaler_target.fit_transform(train_targets_price)
test_targets_scaled = scaler_target.transform(test_targets_price)

joblib.dump(scaler_features, 'scaler_features.pkl')  # บันทึก scaler ฟีเจอร์
joblib.dump(scaler_target, 'scaler_target.pkl')     # บันทึก scaler เป้าหมาย

seq_length = 10

# สร้าง sequences แยกตาม Ticker
X_train_list, X_train_ticker_list, y_train_list = [], [], []
X_val_list, X_val_ticker_list, y_val_list = [], [], []
X_test_list, X_test_ticker_list, y_test_list = [], [], []

for t_id in range(num_tickers):
    # Train
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
        
    # Test
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

if len(X_train_list) > 0:
    X_price_train = np.concatenate(X_train_list, axis=0)
    X_ticker_train = np.concatenate(X_train_ticker_list, axis=0)
    y_price_train = np.concatenate(y_train_list, axis=0)
else:
    X_price_train, X_ticker_train, y_price_train = np.array([]), np.array([]), np.array([])

if len(X_val_list) > 0:
    X_price_val = np.concatenate(X_val_list, axis=0)
    X_ticker_val = np.concatenate(X_val_ticker_list, axis=0)
    y_price_val = np.concatenate(y_val_list, axis=0)
else:
    X_price_val, X_ticker_val, y_price_val = np.array([]), np.array([]), np.array([])

if len(X_test_list) > 0:
    X_price_test = np.concatenate(X_test_list, axis=0)
    X_ticker_test = np.concatenate(X_test_ticker_list, axis=0)
    y_price_test = np.concatenate(y_test_list, axis=0)
else:
    X_price_test, X_ticker_test, y_price_test = np.array([]), np.array([]), np.array([])

num_feature = train_features_scaled.shape[1]  # จำนวน features ทางเทคนิค

# สร้างโมเดล LSTM + Embedding
features_input = Input(shape=(seq_length, num_feature), name='features_input')
ticker_input = Input(shape=(seq_length,), name='ticker_input')

print(f"Shape of X_price_train: {X_price_train.shape}")
print(f"Shape of X_ticker_train: {X_ticker_train.shape}")


embedding_dim = 32
ticker_embedding = Embedding(input_dim=num_tickers, output_dim=embedding_dim, name='ticker_embedding')(ticker_input)

merged = concatenate([features_input, ticker_embedding], axis=-1)

x = LSTM(64, return_sequences=True)(merged)
x = Dropout(0.2)(x)
x = LSTM(32)(x)
x = Dropout(0.2)(x)
output = Dense(1)(x)

model = Model(inputs=[features_input, ticker_input], outputs=output)
model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_price_model.keras', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

logging.info("เริ่มฝึกโมเดลสำหรับราคาหุ้นรวม (ใช้ Embedding สำหรับ Ticker)")

history = model.fit(
    [X_price_train, X_ticker_train], y_price_train,
    epochs=200,
    batch_size=32,
    verbose=1,
    shuffle=False,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)


model.save('price_prediction_LSTM_model_embedding.keras')
logging.info("บันทึกโมเดลราคาหุ้นรวมเรียบร้อยแล้ว")

def predict_next_day_with_retraining_LSTM(model, test_df, feature_columns, seq_length=10):
    """
    Predict next day's price and retrain with actual data using only test_df with LSTM
    """
    predictions = []
    actual_values = []
    
    # Sort data by date to ensure sequential prediction
    test_df = test_df.sort_values('Date')
    
    for i in range(len(test_df) - 1):  # -1 because we need next day's actual value
        current_date = test_df.iloc[i]['Date']
        next_date = test_df.iloc[i + 1]['Date']
        current_ticker = test_df.iloc[i]['Ticker']
        
         # Print progress
        if i % 100 == 0:  # Print every 100 iterations
            print(f"Processing: {i}/{len(test_df)-1}")
        
        # Get historical data for this prediction using only test_df
        historical_data = test_df.iloc[:i+1]
        historical_data = historical_data[historical_data['Ticker'] == current_ticker].tail(seq_length)
        
        if len(historical_data) < seq_length:
            continue
        
        # Prepare features for prediction
        features = historical_data[feature_columns].values
        ticker_ids = historical_data['Ticker_ID'].values
        
        # Scale features
        features_scaled = scaler_features.transform(features)
        
        # Reshape for LSTM input (3D input: [samples, timesteps, features])
        X_features = features_scaled.reshape(1, seq_length, len(feature_columns))
        X_ticker = ticker_ids.reshape(1, seq_length)
        
        # Make prediction
        pred = model.predict([X_features, X_ticker], verbose=0)
        pred_unscaled = scaler_target.inverse_transform(pred)[0][0]
        
        # Get actual value (next day's price)
        actual = test_df.iloc[i + 1]['Close']
        
        predictions.append(pred_unscaled)
        actual_values.append(actual)
        
        # Retrain model with new data point using test_df only
        if test_df.iloc[i + 1]['Ticker'] == current_ticker:
            # Prepare new training data from test_df
            new_features = test_df.iloc[i][feature_columns].values.reshape(1, -1)
            new_features_scaled = scaler_features.transform(new_features)
            new_target = test_df.iloc[i + 1]['Close'].reshape(1, -1)
            new_target_scaled = scaler_target.transform(new_target)
            
            # Create sequence for training
            train_seq_features = features_scaled
            train_seq_ticker = ticker_ids
            
            # Fit LSTM model for one epoch with the new data
            model.fit(
                [train_seq_features.reshape(1, seq_length, len(feature_columns)),
                 train_seq_ticker.reshape(1, seq_length)],
                new_target_scaled,
                epochs=1,
                batch_size=1,  # Update with batch size of 1 since we are training with a single new sample
                verbose=0
            )
    
    return predictions, actual_values

# Execute the prediction with retraining using test_df only
predictions, actual_values = predict_next_day_with_retraining_LSTM(
    model, 
    test_df, 
    feature_columns
)


# Calculate metrics
mae = mean_absolute_error(actual_values, predictions)
mse = mean_squared_error(actual_values, predictions)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(actual_values, predictions)
r2 = r2_score(actual_values, predictions)

print("\nTest Metrics with Retraining:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"R2 Score: {r2:.4f}")

# Plot results (Actual vs Predicted)
plt.figure(figsize=(15, 6))
plt.plot(actual_values, label='Actual', color='blue')
plt.plot(predictions, label='Predicted', color='red', alpha=0.7)
plt.title('Next Day Predictions with Retraining')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Save the plot to a file
plt.savefig('next_day_predictions_with_retraining.png')

# Plot residuals (Actual - Predicted)
residuals = np.array(actual_values) - np.array(predictions)
plt.figure(figsize=(15, 6))
plt.scatter(range(len(residuals)), residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')  # Line at y = 0
plt.title('Prediction Residuals with Retraining')
plt.xlabel('Time')
plt.ylabel('Residual')
plt.show()

# Save residuals plot to a file
plt.savefig('prediction_residuals_with_retraining.png')

model.save('price_prediction_LSTM_model_embedding_aftertest.keras')