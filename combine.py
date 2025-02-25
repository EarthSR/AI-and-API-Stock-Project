import pandas as pd

# โหลดข้อมูล Sentiment
sentiment_df_th = pd.read_csv("./Finbert/daily_sentiment_result_th.csv")
sentiment_df_us = pd.read_csv("./Finbert/daily_sentiment_result_us.csv")

# โหลดข้อมูลหุ้น
stock_df_th = pd.read_csv("./Finbert/stock_data_with_marketcap_thai.csv")
stock_df_us = pd.read_csv("./Finbert/stock_data_from_dates.csv")

# โหลดข้อมูลการเงิน
financial_thai_df = pd.read_csv("./AntData/financial_thai_data.csv")
financial_us_df = pd.read_csv("./AntData/financial_america_data.csv")

columns_to_keep = [
            'Stock', 'Year',
            'Total Revenue', 'YoY Growth (%)', 'Net Profit', 'Earnings Per Share (EPS)', 
            'ROA (%)', 'ROE (%)', 'Gross Margin (%)', 'Net Profit Margin (%)',
            'Debt to Equity (x)','P/E Ratio (x)', 'P/BV Ratio (x)', 'Dividend Yield (%)'
]
financial_thai_df = financial_thai_df[columns_to_keep]
financial_us_df = financial_us_df[columns_to_keep]

# ลบคอลัมน์ที่มีค่า null หรือ NaN ทั้งหมด
financial_thai_df.dropna(axis=1, how='all', inplace=True)
financial_us_df.dropna(axis=1, how='all', inplace=True)

# แปลงคอลัมน์วันที่เป็น datetime เพื่อให้สามารถรวมข้อมูลได้ถูกต้อง
sentiment_df_th["date"] = pd.to_datetime(sentiment_df_th["date"])
sentiment_df_us["date"] = pd.to_datetime(sentiment_df_us["date"])
stock_df_th["Date"] = pd.to_datetime(stock_df_th["Date"])
stock_df_us["Date"] = pd.to_datetime(stock_df_us["Date"])

# เปลี่ยนชื่อคอลัมน์ Sentiment Category เป็น Sentiment
sentiment_df_th.rename(columns={'Sentiment Category': 'Sentiment'}, inplace=True)
sentiment_df_us.rename(columns={'Sentiment Category': 'Sentiment'}, inplace=True)

# ดึงปีจาก Date ใน stock data เพื่อใช้รวมกับข้อมูลการเงิน
financial_thai_df["Year"] = financial_thai_df["Year"].astype(int)
financial_us_df["Year"] = financial_us_df["Year"].astype(int)

# รวมข้อมูลโดยใช้ Date เป็นตัวเชื่อม
merged_df_th = stock_df_th.merge(
    sentiment_df_th[['date', 'Sentiment']],
    left_on='Date',
    right_on='date',
    how='left'
)

merged_df_us = stock_df_us.merge(
    sentiment_df_us[['date', 'Sentiment']],
    left_on='Date',
    right_on='date',
    how='left'
)

# ลบคอลัมน์ 'date' ที่ซ้ำกันออก
merged_df_th.drop(columns=['date'], inplace=True)
merged_df_us.drop(columns=['date'], inplace=True)

# รวมข้อมูลของไทยและสหรัฐฯ เข้าด้วยกัน
merged_df = pd.concat([merged_df_th, merged_df_us], ignore_index=True)

# ดึงปีจาก Date เพื่อนำไปใช้รวมกับข้อมูลการเงิน
merged_df["Year"] = merged_df["Date"].dt.year

# รวมข้อมูลการเงินโดยใช้ Stock (Ticker) และ Year เป็นตัวเชื่อม
merged_df = merged_df.merge(
    financial_thai_df,
    left_on=['Ticker', 'Year'],
    right_on=['Stock', 'Year'],
    how='left'
)
merged_df = merged_df.merge(
    financial_us_df,
    left_on=['Ticker', 'Year'],
    right_on=['Stock', 'Year'],
    how='left'
)

# ลบข้อมูลที่มีค่าว่างในหุ้น
merged_df.dropna(subset=['Ticker', 'Date', 'Close'], inplace=True)

# เลือกคอลัมน์ที่ลงท้ายด้วย '_y'
columns_with_y = [col for col in merged_df.columns if col.endswith('_y')]

merged_df.drop(columns=columns_with_y, inplace=True)

# ลบ '_x' และ '(x)' ออกจากชื่อคอลัมน์โดยไม่เติม space
merged_df.columns = merged_df.columns.str.replace(r'(_x|\(x\))', '', regex=True)



# บันทึกข้อมูลที่รวมแล้วลงไฟล์ CSV
merged_df.to_csv("merged_stock_sentiment_financial.csv", index=False)

# แสดงข้อมูลที่รวมแล้ว
print(merged_df.head())