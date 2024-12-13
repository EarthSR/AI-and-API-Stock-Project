import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# โหลดข้อมูล
# เปลี่ยน path ให้ตรงกับไฟล์ข้อมูลจริง
file_path = "./stock_America/NDQ_Stock_History_10Y.csv"
data = pd.read_csv(file_path)

# 1. จัดการข้อมูล Missing Values
print("จำนวนข้อมูลที่หายไปในแต่ละคอลัมน์:")
print(data.isnull().sum())

# ลบแถวที่ไม่มี Ticker
if 'Ticker' in data.columns:
    data = data[~data['Ticker'].isnull()]
    print("\nหลังลบแถวที่ไม่มี Ticker:")
    print(data.isnull().sum())

# เติมค่าข้อมูลที่หายไปในคอลัมน์ตัวเลขด้วยค่าเฉลี่ย
numeric_columns = data.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

print("\nข้อมูลหลังเติมค่า Missing Values:")
print(data.isnull().sum())

# 2. จัดการ Outliers ด้วย IQR Method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# ลบ Outliers ในคอลัมน์ตัวเลข
for column in numeric_columns:
    data = remove_outliers(data, column)

print("\nข้อมูลหลังลบ Outliers:")
print(data.describe())

# 3. ปรับคอลัมน์ Date ให้เป็น datetime
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    print("\nตรวจสอบข้อมูล Date หลังแปลงเป็น datetime:")
    print(data['Date'].head())

# 4. บันทึกข้อมูลที่จัดการแล้ว
output_path = "cleaned_data.csv"
data.to_csv(output_path, index=False)
print(f"\nข้อมูลที่จัดการแล้วถูกบันทึกที่: {output_path}")
