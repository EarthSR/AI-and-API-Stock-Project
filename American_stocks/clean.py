import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

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

# ลบ Outliers
def remove_outliers_with_custom_multiplier(df, stock_column, numeric_columns, multiplier_dict):
    filtered_data = pd.DataFrame()
    for stock in df[stock_column].unique():
        stock_data = df[df[stock_column] == stock]
        for column in numeric_columns:
            if stock_data[column].nunique() > 1:
                Q1, Q3 = stock_data[column].quantile(0.25), stock_data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound, upper_bound = Q1 - multiplier_dict.get(stock, 3.0) * IQR, Q3 + multiplier_dict.get(stock, 3.0) * IQR
                stock_data = stock_data[(stock_data[column] >= lower_bound) & (stock_data[column] <= upper_bound)]
        filtered_data = pd.concat([filtered_data, stock_data], ignore_index=True)
    return filtered_data

high_volatility_cleaned_data = remove_outliers_with_custom_multiplier(high_volatility_data, 'Ticker', numeric_columns, {ticker: 5.0 for ticker in high_volatility_stocks_list})
normal_cleaned_data = remove_outliers_with_custom_multiplier(normal_stocks_data, 'Ticker', numeric_columns, {ticker: 3.0 for ticker in normal_stocks_data['Ticker'].unique()})

# รวมข้อมูลทั้งหมด
final_cleaned_data = pd.concat([high_volatility_cleaned_data, normal_cleaned_data], ignore_index=True)

# แปลง Date เป็น datetime
final_cleaned_data['Date'] = pd.to_datetime(final_cleaned_data['Date'], errors='coerce')

# บันทึกข้อมูล
final_cleaned_data.to_csv("cleaned_data.csv", index=False)
print("\nข้อมูลที่จัดการแล้วทั้งหมดถูกบันทึกที่: cleaned_data.csv")
