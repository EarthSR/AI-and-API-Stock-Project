import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm  # Import tqdm for the progress bar
import tensorflow as tf


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')  # ใช้ GPU ตัวแรก
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU:", physical_devices[0])
else:
    print("GPU not found, using CPU")

# สมมติว่า 'df_stock' คือ DataFrame ของข้อมูลหุ้น
df_stock = pd.read_csv("NDQ_Stock_History_10Y.csv", parse_dates=["Date"])
df_stock = df_stock.sort_values(by="Date")

# คำนวณ Change และ Change (%)
df_stock['Change'] = df_stock['Close'] - df_stock['Open']
df_stock['Change (%)'] = (df_stock['Change'] / df_stock['Open']) * 100

# คำนวณ RSI (14 วัน)
def calculate_rsi(data, window=14):
    delta = data.diff()  # การเปลี่ยนแปลงของราคา
    gain = delta.where(delta > 0, 0)  # ราคาที่เพิ่มขึ้น
    loss = -delta.where(delta < 0, 0)  # ราคาที่ลดลง
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()  # ค่าเฉลี่ยของ gain
    avg_loss = loss.rolling(window=window, min_periods=1).mean()  # ค่าเฉลี่ยของ loss
    
    rs = avg_gain / avg_loss  # Relative Strength
    rsi = 100 - (100 / (1 + rs))  # RSI
    
    return rsi

# คำนวณ RSI และเพิ่มลงใน DataFrame
df_stock['RSI'] = calculate_rsi(df_stock['Close'], window=14)

# คำนวณ Moving Averages (MA) - 50 วัน และ 200 วัน
df_stock['SMA_50'] = df_stock['Close'].rolling(window=50).mean()
df_stock['SMA_200'] = df_stock['Close'].rolling(window=200).mean()

# คำนวณ MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = data.ewm(span=long_window, min_periods=1, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

# คำนวณ MACD และเพิ่มลงใน DataFrame
df_stock['MACD'], df_stock['MACD_Signal'], df_stock['MACD_Histogram'] = calculate_macd(df_stock['Close'])

# สมมติว่า 'df_news' คือ DataFrame ของข้อมูลข่าว
df_news = pd.read_csv("news_with_sentiment_gpu.csv")

# ประมวลผลข้อมูลข่าว
# แปลง Sentiment (Positive=1, Negative=-1, Neutral=0) และ Confidence เป็นตัวเลข
df_news['Sentiment'] = df_news['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
df_news['Confidence'] = df_news['Confidence'] / 100  # เปลี่ยน Confidence เป็นค่าระหว่าง 0-1

# ทำการรวมข้อมูลข่าวและข้อมูลหุ้นตามวันที่
df = pd.merge(df_stock, df_news[['Date', 'Sentiment', 'Confidence']], on='Date', how='left')

# ใช้ฟีเจอร์ Open, High, Low, Close, Volume, Change (%) และ Sentiment, Confidence, RSI, MA, MACD
features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment', 'Confidence', 'RSI', 'SMA_50', 'SMA_200', 'MACD', 'MACD_Signal', 'MACD_Histogram']].values
targets = df['Close'].shift(-1).dropna().values  # ทำนายราคาปิดวันถัดไป

# Normalize features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# สร้างลำดับข้อมูล (Sequence)
def create_sequences(features, targets, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        if i + seq_length < len(targets):  # ตรวจสอบให้แน่ใจว่าไม่เกินขนาดของ targets
            X.append(features[i:i + seq_length])
            y.append(targets[i + seq_length])  # Targets are shifted by seq_length days
    return np.array(X), np.array(y)



seq_length = 10  # ใช้ข้อมูล 10 วันก่อนหน้าเพื่อทำนายราคาวันถัดไป
X, y = create_sequences(features_scaled, targets, seq_length)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล GRU
model = Sequential([
    GRU(64, activation='relu', return_sequences=True, input_shape=(seq_length, X.shape[2])),
    Dropout(0.2),
    GRU(32, activation='relu'),
    Dropout(0.2),
    Dense(1)  # ทำนายราคาหุ้น
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Create a custom callback to integrate tqdm progress bar
class TQDMProgressBar(Callback):
    def on_epoch_end(self, epoch, logs=None):
        tqdm.write(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']:.4f} - mae: {logs['mae']:.4f} "
                   f"- val_loss: {logs['val_loss']:.4f} - val_mae: {logs['val_mae']:.4f}")

# Use tqdm to display progress for training
epochs = 50
batch_size = 32

# Add TQDMProgressBar callback to show the training progress
progress_bar = TQDMProgressBar()

# Start the model training and use the progress bar
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0, callbacks=[progress_bar])

model.save('stock_prediction_gru_model.h5')

loaded_model = load_model('stock_prediction_gru_model.h5')

# ทำนายผล
predictions = loaded_model.predict(X_test)


# คำนวณตัวประเมินความถูกต้อง
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

# แสดงผลตัวประเมิน
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# แสดงกราฟผลการฝึกฝน (Loss และ MAE)
plt.figure(figsize=(12, 6))

# กราฟ Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# กราฟ MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()

plt.tight_layout()
plt.show()
