import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Flatten, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
import matplotlib.pyplot as plt

# 1. โหลดข้อมูลจากไฟล์ CSV
df_stock = pd.read_csv('cleaned_data.csv', parse_dates=['Date']).sort_values(by=['Ticker', 'Date'])
df_news = pd.read_csv('news_with_sentiment_gpu.csv')

# 2. รวมข้อมูลข่าวและหุ้น
df_news['Sentiment'] = df_news['Sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
df = pd.merge(df_stock, df_news[['Date', 'Sentiment', 'Confidence']], on='Date', how='left')

# 3. เติมค่าที่ขาดหายไป
df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

# 4. เพิ่มฟีเจอร์
df['Change'] = df['Close'] - df['Open']
df['Change (%)'] = (df['Change'] / df['Open']) * 100

# 5. เลือกฟีเจอร์ที่ต้องการ
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment', 'Confidence']
df['Ticker_ID'] = LabelEncoder().fit_transform(df['Ticker'])

# 6. แบ่งข้อมูล Train/Val/Test ตามเวลา
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
sorted_dates = df['Date'].sort_values().unique()
train_cutoff = sorted_dates[int(len(sorted_dates) * train_ratio)]
val_cutoff = sorted_dates[int(len(sorted_dates) * (train_ratio + val_ratio))]

train_df = df[df['Date'] <= train_cutoff].copy()
val_df = df[(df['Date'] > train_cutoff) & (df['Date'] <= val_cutoff)].copy()
test_df = df[df['Date'] > val_cutoff].copy()

# 7. สร้าง target โดย shift(-1)
train_targets_price = train_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
train_df = train_df.iloc[:-1]

val_targets_price = val_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
val_df = val_df.iloc[:-1]

test_targets_price = test_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
test_df = test_df.iloc[:-1]

train_features = train_df[feature_columns].values
val_features = val_df[feature_columns].values
test_features = test_df[feature_columns].values

train_ticker_id = train_df['Ticker_ID'].values
val_ticker_id = val_df['Ticker_ID'].values
test_ticker_id = test_df['Ticker_ID'].values

# 8. สเกลข้อมูล
scaler_features = MinMaxScaler()
train_features_scaled = scaler_features.fit_transform(train_features)
val_features_scaled = scaler_features.transform(val_features)
test_features_scaled = scaler_features.transform(test_features)

scaler_target = MinMaxScaler()
train_targets_scaled = scaler_target.fit_transform(train_targets_price)
val_targets_scaled = scaler_target.transform(val_targets_price)
test_targets_scaled = scaler_target.transform(test_targets_price)

# บันทึก Scaler
joblib.dump(scaler_features, 'mlp_scaler_features.pkl')
joblib.dump(scaler_target, 'mlp_scaler_target.pkl')

# 9. เตรียมข้อมูลสำหรับโมเดล
seq_length = 10

def create_sequences(features, ticker_ids, targets, seq_length):
    X_features, X_tickers, Y = [], [], []
    for i in range(len(features) - seq_length):
        X_features.append(features[i:i+seq_length])
        X_tickers.append(ticker_ids[i:i+seq_length])
        Y.append(targets[i+seq_length])
    return np.array(X_features), np.array(X_tickers), np.array(Y)

X_train, X_train_ticker, y_train = create_sequences(train_features_scaled, train_ticker_id, train_targets_scaled, seq_length)
X_val, X_val_ticker, y_val = create_sequences(val_features_scaled, val_ticker_id, val_targets_scaled, seq_length)
X_test, X_test_ticker, y_test = create_sequences(test_features_scaled, test_ticker_id, test_targets_scaled, seq_length)

# 10. สร้างโมเดล MLP
features_input = Input(shape=(seq_length, train_features_scaled.shape[1]), name='features_input')
ticker_input = Input(shape=(seq_length,), name='ticker_input')

embedding_dim = 32
ticker_embedding = Embedding(input_dim=df['Ticker_ID'].max() + 1, output_dim=embedding_dim, name='ticker_embedding')(ticker_input)
ticker_embedding_flat = Flatten()(ticker_embedding)

features_flat = Flatten()(features_input)
merged = concatenate([features_flat, ticker_embedding_flat])

# ลด Dense Layer เหลือ 2 Layers
x = Dense(128, activation='relu')(merged)
x = Dropout(0.3)(x)  # Dropout หลัง Dense Layer แรก
x = Dense(32, activation='relu')(x)
output = Dense(1)(x)  # Output Layer

model = Model(inputs=[features_input, ticker_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# 11. ฝึกสอนโมเดล
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model_mlp.keras', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001)

history = model.fit(
    [X_train, X_train_ticker], y_train,
    validation_data=([X_val, X_val_ticker], y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# 12. ประเมินผล
y_pred = model.predict([X_test, X_test_ticker])
y_pred_rescaled = scaler_target.inverse_transform(y_pred)
y_test_rescaled = scaler_target.inverse_transform(y_test)

mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)

print("Evaluation on Test Set:")
print(f"MAE (Mean Absolute Error): {mae}")
print(f"MSE (Mean Squared Error): {mse}")
print(f"RMSE (Root Mean Squared Error): {rmse}")
print(f"R² Score: {r2}")
print(f"MAPE (Mean Absolute Percentage Error): {mape}")

# 13. วาดกราฟ True vs Predicted
def plot_predictions(y_true, y_pred, title="True vs Predicted Prices"):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="True Values", color="blue")
    plt.plot(y_pred, label="Predicted Values", color="red", alpha=0.7)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

plot_predictions(y_test_rescaled[:200], y_pred_rescaled[:200], title="True vs Predicted Prices (Test Set)")

# 14. วาดกราฟ Residuals
def plot_residuals(y_true, y_pred, title="Residuals"):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(residuals)), residuals, alpha=0.5, color="purple")
    plt.hlines(y=0, xmin=0, xmax=len(residuals), colors="red")
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Residual")
    plt.show()

plot_residuals(y_test_rescaled, y_pred_rescaled, title="Residuals (Test Set)")


# 16. บันทึกโมเดลในรูปแบบ HDF5 (.h5)
model.save('mlp_stock_prediction.h5')
print("Model saved as 'mlp_stock_prediction.h5'")
