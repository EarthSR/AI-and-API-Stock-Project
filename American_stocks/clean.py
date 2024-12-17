import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# 1. โหลดข้อมูล
file_path = "./stock_data_from_dates.csv"
data = pd.read_csv(file_path, usecols=['Ticker', 'Date', 'Open', 'Close', 'High', 'Low', 'Volume'])

# กรองข้อมูลเฉพาะ 5 ตัวย่อหุ้นที่ต้องการ
tickers_to_filter = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL']
data = data[data['Ticker'].isin(tickers_to_filter)]

# ตรวจสอบข้อมูล
print("ค่าที่พบในคอลัมน์ 'Ticker':")
print(data['Ticker'].unique())

tickers_to_filter = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL']
print(data[data['Ticker'].isin(tickers_to_filter)])  # ตรวจสอบว่ามีข้อมูลที่กรองหรือไม่

# เติม Missing Values
numeric_columns = data.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

# ตรวจสอบข้อมูล Date หลังแปลงเป็น datetime
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    print("\nตรวจสอบข้อมูล Date หลังแปลงเป็น datetime:")
    print(data['Date'].head())

# ใช้ Robust Scaling สำหรับจัดการ outliers
scaler = RobustScaler()
numeric_columns_to_scale = ['Open', 'Close', 'High', 'Low', 'Volume']
data[numeric_columns_to_scale] = scaler.fit_transform(data[numeric_columns_to_scale])

# ตรวจสอบและจัดการกับ NaN ก่อนบันทึกข้อมูล
data = data.dropna()  # ลบแถวที่มี NaN

# บันทึกข้อมูลเป็น CSV
data.to_csv("cleaned_data.csv", index=False)
print("บันทึกข้อมูลเรียบร้อยแล้ว")
