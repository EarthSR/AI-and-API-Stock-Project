import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. โหลดข้อมูลหุ้น
stock_data = pd.read_csv('cleaned_data.csv')
stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce', utc=True).dt.date

# เลื่อนเป้าหมายราคาปิด (Close) ไปวันถัดไป
stock_data['Next Day Close'] = stock_data['Close'].shift(-1)

# ลบแถวที่ไม่มีเป้าหมาย
stock_data = stock_data.dropna(subset=['Next Day Close'])

# 2. โหลดข้อมูลข่าว
news_data = pd.read_csv('news_with_sentiment_gpu.csv')
news_data['Date'] = pd.to_datetime(news_data['date'], errors='coerce', utc=True).dt.date

# ตั้งค่า index เป็นวันที่
stock_data.set_index('Date', inplace=True)
news_data.set_index('Date', inplace=True)

# รวมข้อมูลหุ้นและข่าวโดยใช้วันที่
combined_data = pd.merge(stock_data, news_data, how='inner', left_index=True, right_index=True)

# แปลงค่าคอลัมน์ Sentiment ให้เป็นตัวเลข
sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
combined_data['Sentiment'] = combined_data['Sentiment'].map(sentiment_mapping)

# เลือก features และ target
features = combined_data[['Open', 'High', 'Low', 'Volume', 'Change', 'Change (%)', 'Sentiment']]
y = combined_data['Next Day Close']

# ปรับขนาดข้อมูล
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# แบ่งข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# สร้างโมเดล MLP
model = Sequential([
    Dense(64, activation='relu', input_dim=len(features.columns)),
    Dense(32, activation='relu'),
    Dense(1)
])

# ตั้งค่าการฝึกสอนโมเดล
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# กำหนด Early Stopping และ ReduceLROnPlateau
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6
)

# ฝึกสอนโมเดล
history = model.fit(
    X_train, y_train,
    epochs=6,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr]
)

# แยกวันที่และชื่อหุ้น
dates = combined_data.index
stock_names = combined_data['Ticker']

# การทำนายราคาหุ้น
predictions = model.predict(X_test)

# คำนวณเปอร์เซ็นต์ความคลาดเคลื่อน
error_tolerance = 0.05  # 5%
accuracy = np.mean(np.abs((predictions.flatten() - y_test) / y_test) <= error_tolerance) * 100
print(f"Accuracy: {accuracy:.2f}%")

# สร้าง DataFrame เพื่อแสดงผลการทำนาย รวมชื่อหุ้น
results = pd.DataFrame({
    'Stock Name': stock_names.iloc[y_test.index],
    'Date': dates[y_test.index],
    'Predicted Next Day Close': predictions.flatten(),
    'Actual Next Day Close': y_test
}).set_index(['Stock Name', 'Date'])

print(results.head())

