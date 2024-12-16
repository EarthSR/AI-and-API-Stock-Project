import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# 1. โหลดข้อมูล
file_path = "./stock_America/NDQ_Stock_History_10Y.csv"
data = pd.read_csv(file_path, usecols=['Ticker', 'Date', 'Open', 'Close', 'High', 'Low', 'Volume', 'Change', 'Change (%)'])

# ตรวจสอบข้อมูล
print("ค่าที่พบในคอลัมน์ 'Ticker':")
print(data['Ticker'].unique())

# เติม Missing Values
numeric_columns = data.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

# แปลงวันที่และจัดการกับ Time Zones
data['Date'] = pd.to_datetime(data['Date'], errors='coerce', utc=False)

# คำนวณ Daily Returns
data = data.sort_values(by=['Ticker', 'Date'])
data['Daily_Return'] = data.groupby('Ticker')['Close'].pct_change()

# คำนวณ Volatility
volatility = data.groupby('Ticker')['Daily_Return'].std().reset_index()
volatility.columns = ['Ticker', 'Volatility']
volatility.to_csv("volatility_data.csv", index=False)

# แยกหุ้นผันผวนสูง
volatility_threshold = 0.05
high_volatility_stocks_list = volatility[volatility['Volatility'] > volatility_threshold]['Ticker'].tolist()
high_volatility_data = data[data['Ticker'].isin(high_volatility_stocks_list)]
normal_stocks_data = data[~data['Ticker'].isin(high_volatility_stocks_list)]

# รวมข้อมูลของหุ้นผันผวนสูงและต่ำ
combined_data = pd.concat([high_volatility_data, normal_stocks_data], axis=0)

# ใช้ Robust Scaling สำหรับจัดการ outliers
scaler = RobustScaler()
numeric_columns_to_scale = ['Open', 'Close', 'High', 'Low', 'Volume']
combined_data[numeric_columns_to_scale] = scaler.fit_transform(combined_data[numeric_columns_to_scale])

# ตรวจสอบและจัดการกับ NaN ก่อนบันทึกข้อมูล
combined_data = combined_data.dropna()  # ลบแถวที่มี NaN

# บันทึกข้อมูลเป็น CSV
combined_data.to_csv("cleaned_data.csv", index=False)
