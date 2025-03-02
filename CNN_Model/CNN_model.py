import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler, StandardScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dense, Dropout, Embedding, 
                                     concatenate, GlobalAveragePooling1D, BatchNormalization)
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import logging
import ta

# สร้าง sequences สำหรับแต่ละ ticker
def create_sequences_for_ticker(features, ticker_ids, targets, seq_length=10):
    X_features, X_tickers, Y = [], [], []
    for i in range(len(features) - seq_length):
        X_features.append(features[i:i+seq_length])
        X_tickers.append(ticker_ids[i:i+seq_length])
        Y.append(targets[i+seq_length])
    return np.array(X_features), np.array(X_tickers), np.array(Y)

# กราฟการฝึก
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # กราฟ MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('MAE During Training')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

# คำนวณกราฟ Predicted vs Actual
def plot_predictions(y_true, y_pred, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='red', alpha=0.7)
    plt.title(f'True vs Predicted Prices for {ticker}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# คำนวณ Residuals
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

# โหลดข้อมูลและทำ preprocessing
df = pd.read_csv('../merged_stock_sentiment_financial.csv')
df['Sentiment'] = df['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
df['Change'] = df['Close'] - df['Open']
df['Change (%)'] = df['Close'].pct_change()
df['Change (%)'] = np.clip(df['Change (%)'], -50, 50)
df['Change (%)'] *= 100
df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
df['RSI'].fillna(method='ffill', inplace=True)
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

# เตรียมข้อมูล financial
financial_columns = ['Total Revenue', 'QoQ Growth (%)', 'YoY Growth (%)', 'Net Profit', 
                     'Earnings Per Share (EPS)', 'ROA (%)', 'ROE (%)', 'Gross Margin (%)', 
                     'Net Profit Margin (%)', 'Debt to Equity ', 'P/E Ratio ',
                     'P/BV Ratio ', 'Dividend Yield (%)']
df_financial = df[['Date', 'Ticker'] + financial_columns].drop_duplicates()
df_financial[financial_columns] = df_financial[financial_columns].where(df_financial[financial_columns].ne(0)).bfill()

# เติม missing สำหรับฟีเจอร์ทางเทคนิค
stock_columns = ['RSI', 'EMA_10', 'EMA_20', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low']
df[stock_columns] = df[stock_columns].fillna(method='ffill')
print(df[['Date', 'Ticker', 'Total Revenue', 'Net Profit']].tail(20))
df.fillna(0, inplace=True)

feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment',
                   'Total Revenue','QoQ Growth (%)', 'YoY Growth (%)', 'Net Profit', 
                   'Earnings Per Share (EPS)', 'ROA (%)', 'ROE (%)', 'Gross Margin (%)', 
                   'Net Profit Margin (%)', 'Debt to Equity ', 'P/E Ratio ', 'P/BV Ratio ', 
                   'Dividend Yield (%)','RSI', 'EMA_10', 'EMA_20', 'MACD', 'MACD_Signal',
                   'Bollinger_High', 'Bollinger_Low']

# Label Encode สำหรับ Ticker
ticker_encoder = LabelEncoder()
df['Ticker_ID'] = ticker_encoder.fit_transform(df['Ticker'])
num_tickers = len(ticker_encoder.classes_)

# แบ่งข้อมูล Train/Val/Test ตามเวลา
sorted_dates = df['Date'].unique()
train_cutoff = sorted_dates[int(len(sorted_dates) * 6 / 7)]

train_df = df[df['Date'] <= train_cutoff].copy()
test_df = df[df['Date'] > train_cutoff].copy()

# สร้าง target โดยใช้ shift(-1)
train_targets_price = train_df['Change (%)'].shift(-1).dropna().values.reshape(-1, 1)
train_df = train_df.iloc[:-1]
test_targets_price = test_df['Change (%)'].shift(-1).dropna().values.reshape(-1, 1)
test_df = test_df.iloc[:-1]

train_features = train_df[feature_columns].values
test_features = test_df[feature_columns].values
train_ticker_id = train_df['Ticker_ID'].values
test_ticker_id = test_df['Ticker_ID'].values

# ตรวจสอบค่า Infinity
for i, col in enumerate(feature_columns):
    if np.any(np.isinf(train_features[:, i])):
        print(f"⚠️ พบค่า Infinity ในคอลัมน์: {col}")

# Scale ข้อมูลโดยใช้ชุด train เท่านั้น
scaler_features = RobustScaler()
train_features_scaled = scaler_features.fit_transform(train_features)
test_features_scaled = scaler_features.transform(test_features)
scaler_target = MinMaxScaler(feature_range=(-1, 1))
train_targets_scaled = scaler_target.fit_transform(train_targets_price)
test_targets_scaled = scaler_target.transform(test_targets_price)

# สร้าง Sequences
seq_length = 10
X_train_list, X_train_ticker_list, y_train_list = [], [], []
X_test_list, X_test_ticker_list, y_test_list = [], [], []

for t_id in range(num_tickers):
    df_train_ticker = train_df[train_df['Ticker_ID'] == t_id]
    if len(df_train_ticker) > seq_length:
        indices = df_train_ticker.index
        mask_train = np.isin(train_df.index, indices)
        f_t = train_features_scaled[mask_train]
        t_t = train_df['Ticker_ID'][mask_train].values
        target_t = train_targets_scaled[mask_train]
        X_t, X_ti, y_t = create_sequences_for_ticker(f_t, t_t, target_t, seq_length)
        X_train_list.append(X_t)
        X_train_ticker_list.append(X_ti)
        y_train_list.append(y_t)
        
    # สำหรับ Test
    df_test_ticker = test_df[test_df['Ticker_ID'] == t_id]
    if len(df_test_ticker) > seq_length:
        indices = df_test_ticker.index
        mask_test = np.isin(test_df.index, indices)
        f_s = test_features_scaled[mask_test]
        t_s = test_df['Ticker_ID'][mask_test].values
        target_s = test_targets_scaled[mask_test]
        X_s, X_si, y_s = create_sequences_for_ticker(f_s, t_s, target_s, seq_length)
        X_test_list.append(X_s)
        X_test_ticker_list.append(X_si)
        y_test_list.append(y_s)

# Concatenate Data
X_train = np.concatenate(X_train_list, axis=0)
X_train_ticker = np.concatenate(X_train_ticker_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)

X_test = np.concatenate(X_test_list, axis=0)
X_test_ticker = np.concatenate(X_test_ticker_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

# สร้างโมเดล CNN โดยมี L2 regularization ในชั้น Dense
features_input = Input(shape=(seq_length, len(feature_columns)), name='features_input')
ticker_input = Input(shape=(seq_length,), name='ticker_input')

embedding_dim = 32
ticker_embedding = Embedding(input_dim=num_tickers, output_dim=embedding_dim, name='ticker_embedding')(ticker_input)

merged = concatenate([features_input, ticker_embedding], axis=-1)

x = Conv1D(32, kernel_size=3, padding='same', kernel_regularizer=l2(0.01))(merged)
x = BatchNormalization()(x)
x = tf.keras.activations.relu(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(64, kernel_size=3, padding='same', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = tf.keras.activations.relu(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(128, kernel_size=3, padding='same', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = tf.keras.activations.relu(x)
x = MaxPooling1D(pool_size=2)(x)

x = GlobalAveragePooling1D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1)(x)

model = Model(inputs=[features_input, ticker_input], outputs=output)
model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])
model.summary()

# Callbacks สำหรับเทรน
early_stopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_price_cnn_model.keras', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

history = model.fit(
    [X_train, X_train_ticker], y_train,
    epochs=2000,
    batch_size=32,
    validation_data=([X_test, X_test_ticker], y_test),
    verbose=1,
    shuffle=False,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# แสดงกราฟการฝึก
plot_training_history(history)

# ทำนายและย้อนกลับ scale
y_pred_scaled = model.predict([X_test, X_test_ticker])

# ย้อนกลับค่าที่ทำนาย
y_pred = scaler_target.inverse_transform(y_pred_scaled)
y_test_original = scaler_target.inverse_transform(y_test)

# บันทึกโมเดล
model.save('price_prediction_CNN_model.keras', save_format='h5')
logging.info("บันทึกโมเดล CNN ราคาหุ้นรวมเรียบร้อยแล้ว")



# === Walk-Forward Validation พร้อมตรวจเช็คทิศทาง ===

def walk_forward_validation(model, df, feature_columns, scaler_features, scaler_target, ticker_encoder, seq_length=10):
    """
    ทำนายแบบ walk-forward สำหรับแต่ละ ticker พร้อม online learning
    และตรวจเช็คทิศทาง (Up/Down) ของการเปลี่ยนแปลง
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
        for i in range(len(df_ticker) - seq_length):
            historical_data = df_ticker.iloc[i:i+seq_length]
            target_data = df_ticker.iloc[i+seq_length]
            features = historical_data[feature_columns].values
            ticker_ids = historical_data['Ticker_ID'].values
            features_scaled = scaler_features.transform(features)
            X_features = features_scaled.reshape(1, seq_length, len(feature_columns))
            X_ticker = ticker_ids.reshape(1, seq_length)
            pred = model.predict([X_features, X_ticker], verbose=0)
            pred_change_pct = scaler_target.inverse_transform(pred.reshape(-1, 1))[0][0]
            actual_change_pct = target_data['Change (%)']
            future_date = target_data['Date']
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
            # Online learning: ปรับโมเดลด้วยข้อมูลจริง
            new_target_scaled = scaler_target.transform([[actual_change_pct]])
            model.fit([X_features, X_ticker], new_target_scaled, epochs=3, batch_size=4, verbose=0)
            if i % 100 == 0:
                print(f"  Processing: {i}/{len(df_ticker)-seq_length}")
    predictions_df = pd.DataFrame(all_predictions)
    metrics_dict = {}
    for ticker, group in predictions_df.groupby('Ticker'):
        actuals = group['Actual Change (%)'].values
        preds = group['Predicted Change (%)'].values
        mae = mean_absolute_error(actuals, preds)
        mse = mean_squared_error(actuals, preds)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(actuals, preds)
        r2 = r2_score(actuals, preds)
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
    predictions_df.to_csv('predictions_change_pct.csv', index=False)
    print("\n✅ Saved deduplicated predictions for all tickers to 'predictions_change_pct.csv'")
    return predictions_df, metrics_dict

predictions_df, results_per_ticker = walk_forward_validation(
    model=load_model('./price_prediction_CNN_model.keras'),
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

metrics_df = pd.DataFrame(results_per_ticker).T
metrics_df.to_csv('metrics_per_ticker.csv', index=True)
print("\nSaved metrics per ticker to 'metrics_per_ticker.csv'")

all_data = []
for ticker, data in results_per_ticker.items():
    for date_val, actual_val, pred_val in zip(data['Dates'], data['Actuals'], data['Predictions']):
        all_data.append([ticker, date_val, actual_val, pred_val])
prediction_df = pd.DataFrame(all_data, columns=['Ticker', 'Date', 'Actual', 'Predicted'])
prediction_df.to_csv('all_predictions_per_day.csv', index=False)
print("Saved actual and predicted prices to 'all_predictions_per_day.csv'")
