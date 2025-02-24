import pandas as pd

# โหลดข้อมูล Sentiment
sentiment_df_th = pd.read_csv("./Finbert/daily_sentiment_result_th.csv")
sentiment_df_us = pd.read_csv("./Finbert/daily_sentiment_result_us.csv")

# โหลดข้อมูลหุ้น
stock_df_th = pd.read_csv("./Finbert/stock_data_with_marketcap_thai.csv")
stock_df_us = pd.read_csv("./Finbert/stock_data_from_dates.csv")

# แปลงคอลัมน์วันที่เป็น datetime เพื่อให้สามารถรวมข้อมูลได้ถูกต้อง
sentiment_df_th["date"] = pd.to_datetime(sentiment_df_th["date"])
sentiment_df_us["date"] = pd.to_datetime(sentiment_df_us["date"])
stock_df_th["Date"] = pd.to_datetime(stock_df_th["Date"])
stock_df_us["Date"] = pd.to_datetime(stock_df_us["Date"])

# เปลี่ยนชื่อคอลัมน์ Sentiment Category เป็น Sentiment
sentiment_df_th.rename(columns={'Sentiment Category': 'Sentiment'}, inplace=True)
sentiment_df_us.rename(columns={'Sentiment Category': 'Sentiment'}, inplace=True)

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

# ลบคอลัมน์ 'date' และ 'Market Cap' ที่ซ้ำกันออก
merged_df_th.drop(columns=['date', 'Market Cap'], inplace=True)
merged_df_us.drop(columns=['date'], inplace=True)

# รวมข้อมูลของไทยและสหรัฐฯ เข้าด้วยกัน
merged_df = pd.concat([merged_df_th, merged_df_us], ignore_index=True)

# ลบข้อมูลที่มีค่าว่างในหุ้น
merged_df.dropna(subset=['Ticker', 'Date', 'Close'], inplace=True)

# บันทึกข้อมูลที่รวมแล้วลงไฟล์ CSV
merged_df.to_csv("merged_stock_sentiment.csv", index=False)

# แสดงข้อมูลที่รวมแล้ว
print(merged_df_th.head())
print(merged_df_us.head())
