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

# # ตั้งค่า logging
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
    plt.savefig('training_history.png')  # บันท��กกรา��การเทรน

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
df = pd.read_csv('../merged_stock_sentiment_financial.csv')

df['Sentiment'] = df['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})



# เพิ่มฟีเจอร์
df['Change'] = df['Close'] - df['Open']
df['Change (%)'] = df['Close'].pct_change()
df['Change (%)'] = np.clip(df['Change (%)'], -50, 50)
df['Change (%)'] *= 100  # ทำให้เป็นเปอร์เซ็นต์
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
upper_bound = df["Change (%)"].quantile(0.99)
lower_bound = df["Change (%)"].quantile(0.01)
df["Change (%)"] = np.clip(df["Change (%)"], lower_bound, upper_bound)


# ✅ ใช้ Backward Fill (เติมค่าล่าสุดก่อนหน้า แทนที่จะเติมค่าทุกวัน)
financial_columns = ['Total Revenue', 'QoQ Growth (%)', 'YoY Growth (%)', 'Net Profit', 
                     'Earnings Per Share (EPS)', 'ROA (%)', 'ROE (%)', 'Gross Margin (%)', 
                     'Net Profit Margin (%)', 'Debt to Equity ', 'P/E Ratio ',
                     'P/BV Ratio ', 'Dividend Yield (%)']
# ✅ กรองเฉพาะแถวที่มีข้อมูลงบการเงิน ไม่เอาวันที่ซ้ำกัน
df_financial = df[['Date', 'Ticker'] + financial_columns].drop_duplicates()
# ✅ เติมค่าที่ขาดหายไปในงบการเงิน ด้วย Backfill
df_financial[financial_columns] = df_financial[financial_columns].where(df_financial[financial_columns].ne(0)).bfill()


# ✅ ใช้ Forward Fill เฉพาะตัวชี้วัดทางเทคนิคของหุ้น
stock_columns = ['RSI', 'EMA_10', 'EMA_20', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low']
df[stock_columns] = df[stock_columns].fillna(method='ffill')

# ✅ ตรวจสอบว่าไม่มีค่าซ้ำทุกวัน
print(df[['Date', 'Ticker', 'Total Revenue', 'Net Profit']].tail(20))

df.fillna(0, inplace=True)

feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment','Total Revenue','QoQ Growth (%)', 
                   'YoY Growth (%)', 'Net Profit', 'Earnings Per Share (EPS)', 'ROA (%)', 'ROE (%)', 
                   'Gross Margin (%)', 'Net Profit Margin (%)', 'Debt to Equity ', 'P/E Ratio ',
                   'P/BV Ratio ', 'Dividend Yield (%)','RSI', 'EMA_10', 'EMA_20', 'MACD', 'MACD_Signal',
                   'Bollinger_High', 'Bollinger_Low']

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

train_df.to_csv('train_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)
print("Train cutoff:", train_cutoff)
print("First date in train set:", train_df['Date'].min())
print("Last date in train set:", train_df['Date'].max())


# สร้าง target โดย shift(-1)
train_targets_price = train_df['Change (%)'].shift(-1).dropna().values.reshape(-1, 1)
train_df = train_df.iloc[:-1]

test_targets_price = test_df['Change (%)'].shift(-1).dropna().values.reshape(-1, 1)
test_df = test_df.iloc[:-1]

train_features = train_df[feature_columns].values
test_features = test_df[feature_columns].values

train_ticker_id = train_df['Ticker_ID'].values
test_ticker_id = test_df['Ticker_ID'].values

# 🔎 ค้นหาว่าคอลัมน์ไหนใน train_features มีค่า inf
for i, col in enumerate(feature_columns):
    if np.any(np.isinf(train_features[:, i])):
        print(f"⚠️ พบค่า Infinity ในคอลัมน์: {col}")


# สเกลข้อมูลจากชุดฝึก (train) เท่านั้น
scaler_features = RobustScaler()
train_features_scaled = scaler_features.fit_transform(train_features)  # ใช้ fit_transform กับชุดฝึก
test_features_scaled = scaler_features.transform(test_features)  # ใช้ transform กับชุดทดสอบ

scaler_target = MinMaxScaler(feature_range=(-1, 1))
train_targets_scaled = scaler_target.fit_transform(train_targets_price)  # ใช้ fit_transform กับชุดฝึก
test_targets_scaled = scaler_target.transform(test_targets_price)  # ใช้ transform กับชุดทดสอบ


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

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_price_model.keras', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

logging.info("เริ่มฝึกโมเดลสำหรับราคาหุ้นรวม (ใช้ Embedding สำหรับ Ticker)")

history = model.fit(
    [X_price_train, X_ticker_train], y_price_train,
    epochs=2000,
    batch_size=32,
    verbose=1,
    shuffle=False,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# แสดงกราฟการฝึก
plot_training_history(history)

model.save('price_prediction_LSTM_model_embedding.keras')
logging.info("บันทึกโมเดลราคาหุ้นรวมเรียบร้อยแล้ว")

def walk_forward_validation(model, df, feature_columns, scaler_features, scaler_target, ticker_encoder, seq_length=10):
    """
    ฟังก์ชันนี้จะทำการทำนายแบบ walk-forward สำหรับแต่ละ ticker
    พร้อมทั้งรีเทรนโมเดลเล็กน้อย (online learning) และเก็บผลลัพธ์สำหรับการคำนวณ metrics
    โดยเพิ่มการตรวจเช็คทิศทาง (Up/Down) ของการเปลี่ยนแปลงด้วย
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
            
            # ตรวจเช็คทิศทางของการเปลี่ยนแปลง
            predicted_direction = "Up" if pred_change_pct >= 0 else "Down"
            actual_direction = "Up" if actual_change_pct >= 0 else "Down"

            all_predictions.append({
                'Ticker': ticker,
                'Date': future_date,
                'Predicted Change (%)': pred_change_pct,
                'Actual Change (%)': actual_change_pct,
                'Predicted Direction': predicted_direction,
                'Actual Direction': actual_direction
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
        # คำนวณ directional accuracy โดยเปรียบเทียบทิศทางที่ทำนายกับทิศทางจริง
        direction_accuracy = np.mean((group['Predicted Direction'] == group['Actual Direction']).astype(int))
        
        metrics_dict[ticker] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2 Score': r2,
            'Direction Accuracy': direction_accuracy,
            'Dates': group['Date'].tolist(),
            'Actuals': actuals.tolist(),
            'Predictions': preds.tolist(),
            'Predicted Directions': group['Predicted Direction'].tolist(),
            'Actual Directions': group['Actual Direction'].tolist()
        }

    # บันทึกผลการทำนายลง CSV
    predictions_df.to_csv('predictions_change_pct.csv', index=False)
    print("\n✅ Saved deduplicated predictions for all tickers to 'predictions_change_pct.csv'")

    return predictions_df, metrics_dict

# ประเมินผลและพยากรณ์แยกตามแต่ละหุ้นโดยใช้ Walk-Forward Validation
predictions_df, results_per_ticker = walk_forward_validation(
    model=load_model('./price_prediction_LSTM_model_embedding.keras'),
    df=test_df,
    feature_columns=feature_columns,
    scaler_features=scaler_features,
    scaler_target=scaler_target,
    ticker_encoder=ticker_encoder,
    seq_length=seq_length
)

for ticker, metrics in results_per_ticker.items():
    print(f"\nMetrics for {ticker}:")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.4f}")
    print(f"R2 Score: {metrics['R2 Score']:.4f}")

# บันทึกเมตริกส์ลงไฟล์ CSV สำหรับการวิเคราะห์เพิ่มเติม
# บันทึกเมตริกส์ลงไฟล์ CSV สำหรับการวิเคราะห์เพิ่มเติม
selected_columns = ['MAE', 'MSE', 'RMSE', 'MAPE', 'R2 Score'] 
metrics_df = pd.DataFrame.from_dict(results_per_ticker, orient='index')
filtered_metrics_df = metrics_df[selected_columns]
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