import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Dropout, Embedding, concatenate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
import ta
from sklearn.preprocessing import RobustScaler
import logging
from tensorflow.keras.losses import MeanSquaredError

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

# โหลดข้อมูล
df_stock = pd.read_csv("cleaned_data.csv", parse_dates=["Date"]).sort_values(by=["Ticker", "Date"])
df_news = pd.read_csv("news_with_sentiment_gpu.csv")
df_news['Date'] = pd.to_datetime(df_news['Date'], errors='coerce')
df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')
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
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].rolling(window=9).mean()  
bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
df['Bollinger_High'] = bollinger.bollinger_hband()
df['Bollinger_Low'] = bollinger.bollinger_lband()

df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment',
                    'RSI', 'SMA_10', 'SMA_5', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low']

# Label Encode Ticker
ticker_encoder = LabelEncoder()
df['Ticker_ID'] = ticker_encoder.fit_transform(df['Ticker'])
num_tickers = len(ticker_encoder.classes_)

# แบ่งข้อมูล Train/Val/Test ตามเวลา
sorted_dates = df['Date'].unique()
train_cutoff = sorted_dates[int(len(sorted_dates) * 6 / 7)]  # ขอบเขตที่ 6 ปี

# ข้อมูล train, test
train_df = df[df['Date'] <= train_cutoff].copy()
test_df = df[df['Date'] > train_cutoff].copy()

# สร้าง target โดย shift(-1)
train_targets_price = train_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
train_df = train_df.iloc[:-1]

test_targets_price = test_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
test_df = test_df.iloc[:-1]

train_features = train_df[feature_columns].values
test_features = test_df[feature_columns].values

train_ticker_id = train_df['Ticker_ID'].values
test_ticker_id = test_df['Ticker_ID'].values

# สเกลข้อมูลจากเทรนเท่านั้น
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

joblib.dump(scaler_features, 'rnn_scaler_features.pkl')  # บันทึก scaler ฟีเจอร์
joblib.dump(scaler_target, 'rnn_scaler_target.pkl')     # บันทึก scaler เป้าหมาย

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
        X_s, X_st, y_s = create_sequences_for_ticker(f_s, t_s, target_s, seq_length)
        X_test_list.append(X_s)
        X_test_ticker_list.append(X_st)
        y_test_list.append(y_s)

# รวมข้อมูลทั้งหมด
X_train = np.concatenate(X_train_list)
X_train_ticker = np.concatenate(X_train_ticker_list)
y_train = np.concatenate(y_train_list)

X_test = np.concatenate(X_test_list)
X_test_ticker = np.concatenate(X_test_ticker_list)
y_test = np.concatenate(y_test_list)

# สร้างโมเดล RNN
features_input = Input(shape=(seq_length, len(feature_columns)), name='features_input')
ticker_input = Input(shape=(seq_length,), name='ticker_input')

embedding_dim = 32
ticker_embedding = Embedding(input_dim=num_tickers, output_dim=embedding_dim, name='ticker_embedding')(ticker_input)

merged = concatenate([features_input, ticker_embedding], axis=-1)

x = SimpleRNN(64, return_sequences=True)(merged)
x = Dropout(0.2)(x)
x = SimpleRNN(32)(x) 
x = Dropout(0.2)(x)
output = Dense(1)(x)

model = Model(inputs=[features_input, ticker_input], outputs=output)
model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])

# ตั้งค่า callback
early_stopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_price_model_rnn.keras', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)


history = model.fit(
    [X_train, X_train_ticker], y_train,
    epochs=1000,
    batch_size=32,
    verbose=1,
    shuffle=False,
    validation_split=0.1,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)
# ประเมินผล
logging.info("ประเมินโมเดลที่ดีที่สุด")
best_model = load_model('best_price_model_rnn.keras')
y_test_pred_scaled = best_model.predict([X_test, X_test_ticker])

# การกลับคืนค่า
y_test_pred = scaler_target.inverse_transform(y_test_pred_scaled)
y_test_true = scaler_target.inverse_transform(y_test)

# ประเมินผล

mae = mean_absolute_error(y_test_true, y_test_pred)
mse = mean_squared_error(y_test_true, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_true, y_test_pred)
mape = mean_absolute_percentage_error(y_test_true, y_test_pred)

print("\nTest Metrics with Retraining:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"R2 Score: {r2:.4f}")

# แสดงผลกราฟ
plot_predictions(y_test_true, y_test_pred, 'Stock Prediction')
plot_residuals(y_test_true, y_test_pred, 'Stock Prediction')


# ฟังก์ชันสำหรับการทำนายและปรับโมเดลใหม่ (Retraining)
def predict_next_day_with_retraining_RNN(model, test_df, feature_columns, seq_length=10):
    """
    Predict next day's price and retrain with actual data using only test_df 
    """
    predictions = []
    actual_values = []
    
    # เรียงข้อมูลตามวันที่
    test_df = test_df.sort_values('Date')
    
    for i in range(len(test_df) - 1):  # -1 เพราะเราต้องใช้ค่าเป้าหมายของวันถัดไป
        current_date = test_df.iloc[i]['Date']
        current_ticker = test_df.iloc[i]['Ticker']
        
        # ดึงข้อมูลย้อนหลังตาม seq_length
        historical_data = test_df.iloc[:i+1]
        historical_data = historical_data[historical_data['Ticker'] == current_ticker].tail(seq_length)
        
        if len(historical_data) < seq_length:
            continue
        
        # เตรียมข้อมูลฟีเจอร์สำหรับการทำนาย
        features = historical_data[feature_columns].values
        ticker_ids = historical_data['Ticker_ID'].values
        
        # ปรับสเกลข้อมูล
        features_scaled = scaler_features.transform(features)
        X_features = features_scaled.reshape(1, seq_length, len(feature_columns))
        X_ticker = ticker_ids.reshape(1, seq_length)
        
        # ทำการทำนาย
        pred = model.predict([X_features, X_ticker], verbose=0)
        pred_unscaled = scaler_target.inverse_transform(pred)[0][0]
        
        # ดึงค่าจริงของวันถัดไป
        actual = test_df.iloc[i + 1]['Close']
        
        predictions.append(pred_unscaled)
        actual_values.append(actual)
        
        # ปรับปรุงโมเดลด้วยข้อมูลจริงของวันถัดไป
        if test_df.iloc[i + 1]['Ticker'] == current_ticker:
            new_features = test_df.iloc[i][feature_columns].values.reshape(1, -1)
            new_features_scaled = scaler_features.transform(new_features)
            new_target = test_df.iloc[i + 1]['Close'].reshape(1, -1)
            new_target_scaled = scaler_target.transform(new_target)
            
            train_seq_features = features_scaled
            train_seq_ticker = ticker_ids
            
            model.fit(
                [train_seq_features.reshape(1, seq_length, len(feature_columns)),
                 train_seq_ticker.reshape(1, seq_length)],
                new_target_scaled,
                epochs=1,
                batch_size=1,
                verbose=0
            )
    
    return predictions, actual_values

predictions, actual_values = predict_next_day_with_retraining_RNN(
    best_model,
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

# บันทึกโมเดล
model.save('price_prediction_RNN_model.keras')
logging.info("บันทึกโมเดลราคาหุ้นรวมเรียบร้อยแล้ว")

