import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

# สมมติว่า 'df_stock' คือ DataFrame ของข้อมูลหุ้น
df_stock = pd.read_csv("NDQ_Stock_History_10Y.csv", parse_dates=["Date"])
df_stock = df_stock.sort_values(by="Date")

# คำนวณ Change และ Change (%)
df_stock['Change'] = df_stock['Close'] - df_stock['Open']
df_stock['Change (%)'] = (df_stock['Change'] / df_stock['Open']) * 100

# สมมติว่า 'df_news' คือ DataFrame ของข้อมูลข่าว
df_news = pd.read_csv("news_with_sentiment_gpu.csv")

# ประมวลผลข้อมูลข่าว
# แปลง Sentiment (Positive=1, Negative=-1, Neutral=0) และ Confidence เป็นตัวเลข
df_news['Sentiment'] = df_news['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
df_news['Confidence'] = df_news['Confidence'] / 100  # เปลี่ยน Confidence เป็นค่าระหว่าง 0-1

# ทำการรวมข้อมูลข่าวและข้อมูลหุ้นตามวันที่
df = pd.merge(df_stock, df_news[['Date', 'Sentiment', 'Confidence']], on='Date', how='left')

# ใช้ฟีเจอร์ Open, High, Low, Close, Volume, Change (%) และ Sentiment, Confidence
features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment', 'Confidence']].values
targets = df['Close'].shift(-1).dropna().values  # ทำนายราคาปิดวันถัดไป

# Normalize features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# สร้างลำดับข้อมูล (Sequence)
def create_sequences(features, targets, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(targets[i + seq_length])
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

# ฝึกโมเดล
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# ทำนายผล
predictions = model.predict(X_test)
