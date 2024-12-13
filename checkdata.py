import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# 1. โหลดข้อมูล
data = pd.read_csv('./American_stocks/stock_America/NDQ_Stock_History_10Y.csv')

# 2. ตรวจสอบข้อมูลเบื้องต้น
print("ข้อมูลทั้งหมด:")
print(data.head())

print("\nข้อมูลเชิงสถิติ:")
print(data.describe())

print("\nประเภทข้อมูลของแต่ละคอลัมน์:")
print(data.dtypes)

# 3. ตรวจสอบข้อมูลที่หายไป (Missing Values)
missing_data = data.isnull().sum()
print("\nจำนวนข้อมูลที่หายไปในแต่ละคอลัมน์:")
print(missing_data)



# 4. เติมข้อมูลที่หายไปด้วยค่ากลาง (Mean Imputation)
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data.select_dtypes(include=[np.number])), columns=data.select_dtypes(include=[np.number]).columns)
data_imputed['Ticker'] = data['Ticker']
data_imputed['Date'] = data['Date']


# 5. ตรวจสอบ Outliers
def detect_outliers(df):
    outliers = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[column] = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

outliers = detect_outliers(data_imputed)
print("\nข้อมูลที่เป็น outliers:")
for column, values in outliers.items():
    print(f"{column}: {len(values)} outliers")

# 6. แยกคอลัมน์ Ticker และ Date ออกจากข้อมูลที่เป็นตัวเลข
numeric_data = data_imputed.drop(['Ticker', 'Date'], axis=1)

# 7. ตรวจสอบความสัมพันธ์ระหว่างฟีเจอร์ (Correlation)
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# 8. แสดงข้อมูลที่เติมแล้ว
print("\nข้อมูลที่เติมค่าขาดหายแล้ว:")
print(data_imputed.head())
