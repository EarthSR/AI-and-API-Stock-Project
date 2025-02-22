import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Dropout, Embedding, concatenate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
import ta
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
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
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
df['Change (%)'] = df['Close'].pct_change() * 100
df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
df['RSI'].fillna(method='ffill', inplace=True)
df['RSI'].fillna(0, inplace=True)
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].rolling(window=9).mean()  
bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
df['Bollinger_High'] = bollinger.bollinger_hband()
df['Bollinger_Low'] = bollinger.bollinger_lband()

df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment',
                   'RSI', 'EMA_10', 'EMA_20', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low']

# Label Encode Ticker
ticker_encoder = LabelEncoder()
df['Ticker_ID'] = ticker_encoder.fit_transform(df['Ticker'])
num_tickers = len(ticker_encoder.classes_)

# แบ่งข้อมูล Train/Val/Test ตามเวลา
sorted_dates = df['Date'].unique()
train_cutoff = sorted_dates[int(len(sorted_dates) * 6 / 7)]  

# ข้อมูล train, test
train_df = df[df['Date'] <= train_cutoff].copy()
test_df = df[df['Date'] > train_cutoff].copy()

train_targets_price = train_df['Change (%)'].shift(-1).fillna(0).values.reshape(-1, 1)
train_df = train_df.iloc[:-1]

test_targets_price = test_df['Change (%)'].shift(-1).fillna(0).values.reshape(-1, 1)
test_df = test_df.iloc[:-1]

train_features = train_df[feature_columns].values
test_features = test_df[feature_columns].values

train_ticker_id = train_df['Ticker_ID'].values
test_ticker_id = test_df['Ticker_ID'].values

train_targets_price = np.nan_to_num(train_targets_price, nan=0)  
test_targets_price = np.nan_to_num(test_targets_price, nan=0)  

scaler_features = RobustScaler()
train_features_scaled = scaler_features.fit_transform(train_features)  # ใช้ fit_transform กับชุดฝึก
test_features_scaled = scaler_features.transform(test_features)  # ใช้ transform กับชุดทดสอบ

scaler_target = RobustScaler()
train_targets_scaled = scaler_target.fit_transform(train_targets_price)
test_targets_scaled = scaler_target.transform(test_targets_price)

# บันทึก Scalers
joblib.dump(scaler_features, 'rnn_scaler_features.pkl')
joblib.dump(scaler_target, 'rnn_scaler_target.pkl')

seq_length = 10

# สร้าง sequences แยกตาม Ticker
X_train_list, X_train_ticker_list, y_train_list = [], [], []
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

X_train = np.concatenate(X_train_list, axis=0)
X_train_ticker = np.concatenate(X_train_ticker_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)

X_test = np.concatenate(X_test_list, axis=0)
X_test_ticker = np.concatenate(X_test_ticker_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

# สร้างโมเดล RNN (SimpleRNN)
features_input = Input(shape=(seq_length, len(feature_columns)), name='features_input')
ticker_input = Input(shape=(seq_length,), name='ticker_input')

embedding_dim = 32
ticker_embedding = Embedding(input_dim=num_tickers, output_dim=embedding_dim, name='ticker_embedding')(ticker_input)

merged = concatenate([features_input, ticker_embedding], axis=-1)

x = SimpleRNN(64, return_sequences=True, activation='relu')(merged)
x = Dropout(0.2)(x)
x = SimpleRNN(32, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(1)(x)

model = Model(inputs=[features_input, ticker_input], outputs=output)
model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])

# แสดงสรุปโมเดล
model.summary()

# ฝึกโมเดล
early_stopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_price_rnn_model.keras', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

logging.info("เริ่มฝึกโมเดล RNN สำหรับราคาหุ้น")

history = model.fit(
    [X_train, X_train_ticker], y_train,
    epochs=1000,
    batch_size=32,
    validation_data=([X_test, X_test_ticker], y_test),
    verbose=1,
    shuffle=True,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# แสดงกราฟการฝึก
plot_training_history(history)

# ทำนายค่าจากโมเดล
y_pred_scaled = model.predict([X_test, X_test_ticker])

# ย้อนกลับค่าที่ทำนาย
y_pred = scaler_target.inverse_transform(y_pred_scaled)
y_test_original = scaler_target.inverse_transform(y_test)

# แสดงกราฟการทำนาย
plot_predictions(y_test_original, y_pred, "Stock")

# แสดง Residuals
plot_residuals(y_test_original, y_pred, "Stock")


model.save('price_prediction_SimpleRNN_model.keras')
logging.info("บันทึกโมเดล SimpleRNN ราคาหุ้นรวมเรียบร้อยแล้ว")

def walk_forward_validation(model, df, feature_columns, scaler_features, scaler_target, ticker_encoder, seq_length=10):
    """
    ฟังก์ชันนี้จะทำการทำนายแบบ walk-forward สำหรับแต่ละ ticker
    พร้อมทั้งรีเทรนโมเดลเล็กน้อย (online learning) และเก็บผลลัพธ์สำหรับการคำนวณ metrics
    """
    all_predictions = []

    tickers = df['Ticker'].unique()
    for ticker in tickers:
        print(f"\nProcessing Ticker: {ticker}")
        ticker_id = ticker_encoder.transform([ticker])[0]
        df_ticker = df[df['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)

        if len(df_ticker) < seq_length + 1:
            print(f"Not enough data for ticker {ticker}, skipping...")
            continue

        # Loop ผ่านข้อมูลทีละ sequence (target เป็นวันถัดไป)
        for i in range(len(df_ticker) - seq_length):
            historical_data = df_ticker.iloc[i:i+seq_length]
            target_data = df_ticker.iloc[i+seq_length]  # target สำหรับพยากรณ์
            features = historical_data[feature_columns].values
            ticker_ids = historical_data['Ticker_ID'].values

            # สเกลฟีเจอร์และจัดรูปแบบข้อมูลให้เป็น 3D input
            features_scaled = scaler_features.transform(features)
            X_features = features_scaled.reshape(1, seq_length, len(feature_columns))
            X_ticker = ticker_ids.reshape(1, seq_length)

            # พยากรณ์ Change (%)
            pred = model.predict([X_features, X_ticker], verbose=0)
            pred_change_pct = scaler_target.inverse_transform(pred.reshape(-1, 1))[0][0]
            actual_change_pct = target_data['Change (%)']
            future_date = target_data['Date']

            all_predictions.append({
                'Ticker': ticker,
                'Date': future_date,
                'Predicted Change (%)': pred_change_pct,
                'Actual Change (%)': actual_change_pct
            })

            # รีเทรนโมเดลด้วยข้อมูลจริง (online learning)
            new_target_scaled = scaler_target.transform([[actual_change_pct]])
            model.fit([X_features, X_ticker], new_target_scaled, epochs=3, batch_size=4, verbose=0)

            if i % 100 == 0:
                print(f"  Processing: {i}/{len(df_ticker)-seq_length}")

    # สร้าง DataFrame จากผลการทำนาย
    predictions_df = pd.DataFrame(all_predictions)

    # คำนวณ Metrics สำหรับแต่ละ ticker
    metrics_dict = {}
    for ticker, group in predictions_df.groupby('Ticker'):
        actuals = group['Actual Change (%)'].values
        preds = group['Predicted Change (%)'].values
        mae = mean_absolute_error(actuals, preds)
        mse = mean_squared_error(actuals, preds)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(actuals, preds)
        r2 = r2_score(actuals, preds)
        metrics_dict[ticker] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2 Score': r2,
            'Dates': group['Date'].tolist(),
            'Actuals': actuals.tolist(),
            'Predictions': preds.tolist()
        }

    # บันทึกผลการทำนายลง CSV
    predictions_df.to_csv('predictions_change_pct.csv', index=False)
    print("\n✅ Saved deduplicated predictions for all tickers to 'predictions_change_pct.csv'")

    return predictions_df, metrics_dict


# ประเมินผลและพยากรณ์แยกตามแต่ละหุ้นโดยใช้ Walk-Forward Validation
predictions_df, results_per_ticker = walk_forward_validation(
    model=load_model('price_prediction_SimpleRNN_model.keras'),
    df=test_df,
    feature_columns=feature_columns,
    scaler_features=scaler_features,
    scaler_target=scaler_target,
    ticker_encoder=ticker_encoder,
    seq_length=seq_length
)

# แสดงผลสรุปเมตริกส์สำหรับแต่ละหุ้น
for ticker, metrics in results_per_ticker.items():
    print(f"\nMetrics for {ticker}:")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.4f}")
    print(f"R2 Score: {metrics['R2']:.4f}")

# บันทึกเมตริกส์ลงไฟล์ CSV สำหรับการวิเคราะห์เพิ่มเติม
metrics_df = pd.DataFrame.from_dict(results_per_ticker, orient='index')
metrics_df.to_csv('metrics_per_ticker.csv', index=True)

print("\nSaved metrics per ticker to 'metrics_per_ticker.csv'")

# รวบรวม Actual และ Prediction ของทุก ticker ลง CSV
all_data = []
for ticker, data in results_per_ticker.items():
    for date_val, actual_val, pred_val in zip(data['Dates'], data['Actuals'], data['Predictions']):
        all_data.append([ticker, date_val, actual_val, pred_val])

prediction_df = pd.DataFrame(all_data, columns=['Ticker', 'Date', 'Actual', 'Predicted'])
prediction_df.to_csv('all_predictions_per_day.csv', index=False)

print("Saved actual and predicted prices to 'all_predictions_per_day.csv'")
