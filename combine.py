import pandas as pd

# โหลดข้อมูล Sentiment
sentiment_df_th = pd.read_csv("./Finbert/daily_sentiment_result_th.csv")
sentiment_df_us = pd.read_csv("./Finbert/daily_sentiment_result_us.csv")

# โหลดข้อมูลหุ้น
stock_df_th = pd.read_csv("./Finbert/stock_data_with_marketcap_thai.csv")
stock_df_us = pd.read_csv("./Finbert/stock_data_from_dates.csv")

# โหลดข้อมูลการเงิน
financial_thai_df = pd.read_csv("./AntData/Financial_Thai_Quarter.csv")
financial_us_df = pd.read_csv("./AntData/Financial_America_Quarter.csv")

columns_to_keep = [
    'Stock', 'Quarter','QoQ Growth (%)',
    'Total Revenue', 'YoY Growth (%)', 'Net Profit', 'Earnings Per Share (EPS)', 
    'ROA (%)', 'ROE (%)', 'Gross Margin (%)', 'Net Profit Margin (%)',
    'Debt to Equity (x)','P/E Ratio (x)', 'P/BV Ratio (x)', 'Dividend Yield (%)'
]
financial_thai_df = financial_thai_df[columns_to_keep]
financial_us_df = financial_us_df[columns_to_keep]

# ฟังก์ชันแปลง `4Q2024` → `2024Q4`
def fix_quarter_format(quarter_str):
    quarter_str = str(quarter_str).strip()
    if len(quarter_str) == 6 and quarter_str[0].isdigit() and quarter_str[1] == "Q":
        return quarter_str[-4:] + "Q" + quarter_str[0]  # เปลี่ยน `4Q2024` → `2024Q4`
    return quarter_str

# ฟังก์ชันแปลง `Quarter` เป็นวันที่ประกาศงบ
def quarter_to_announcement_date(quarter_str):
    try:
        year, q = int(quarter_str[:4]), int(quarter_str[-1])  # แยกปีและไตรมาส
        if q == 1:
            return pd.Timestamp(f'{year}-04-01')  # ไตรมาสที่ 1
        elif q == 2:
            return pd.Timestamp(f'{year}-07-01')  # ไตรมาสที่ 2
        elif q == 3:
            return pd.Timestamp(f'{year}-10-01')  # ไตรมาสที่ 3
        elif q == 4:
            return pd.Timestamp(f'{year}-01-01')  # ไตรมาสที่ 4
    except:
        return pd.NaT  # หากเกิดข้อผิดพลาดในการแปลง ให้คืนค่าเป็น NaT


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

# แปลงคอลัมน์ Quarter ให้เป็นรูปแบบ `2024Q4`
financial_thai_df['Quarter'] = financial_thai_df['Quarter'].apply(fix_quarter_format)
financial_us_df['Quarter'] = financial_us_df['Quarter'].apply(fix_quarter_format)

# แปลงคอลัมน์ Quarter ให้เป็นวันที่ประกาศงบ
financial_thai_df['Quarter Date'] = financial_thai_df['Quarter'].apply(quarter_to_announcement_date)
financial_us_df['Quarter Date'] = financial_us_df['Quarter'].apply(quarter_to_announcement_date)

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

# ตรวจสอบว่า 'Quarter' มีใน merged_df หรือไม่
if 'Quarter' not in merged_df.columns:
    # หากไม่มี 'Quarter' ให้เพิ่มคอลัมน์ Quarter ด้วยการใช้ Date เพื่อดึงไตรมาส
    merged_df['Quarter'] = merged_df['Date'].dt.to_period('Q').astype(str)

# ตรวจสอบชื่อคอลัมน์ใน merged_df
print("Merged Columns: ", merged_df.columns)

# ใช้ Ticker และ Quarter เป็นตัวเชื่อมในการรวมข้อมูลการเงิน
merged_df = merged_df.merge(
    financial_thai_df,
    left_on=['Ticker', 'Quarter'],
    right_on=['Stock', 'Quarter'],
    how='left'
)

merged_df = merged_df.merge(
    financial_us_df,
    left_on=['Ticker', 'Quarter'],
    right_on=['Stock', 'Quarter'],
    how='left'
)

# ตรวจสอบชื่อคอลัมน์ทั้งหมดใน merged_df
print("Merged Columns after financial merge: ", merged_df.columns)

# ลบข้อมูลที่มีค่าว่างในหุ้น
merged_df.dropna(subset=['Ticker', 'Date', 'Close'], inplace=True)

# เช็คคอลัมน์ที่ลงท้ายด้วย '_x' ถ้ามี NaN ให้ไปเติมค่าในคอลัมน์ที่ลงท้ายด้วย '_y'
for col in merged_df.columns:
    if col.endswith('_x'):
        # คอลัมน์ที่ลงท้ายด้วย _x และ _y ที่ตรงกัน
        y_col = col.replace('_x', '_y')
        # เช็คว่า _x เป็น NaN หรือไม่ ถ้าเป็น NaN ให้เติมค่าจาก _y
        merged_df[col] = merged_df[col].fillna(merged_df[y_col])
        # ลบคอลัมน์ _y
        merged_df.drop(columns=[y_col], inplace=True)


# ลบ '_x' และ '(x)' ออกจากชื่อคอลัมน์โดยไม่เติม space
merged_df.columns = merged_df.columns.str.replace(r'(_x|\(x\))', '', regex=True)

# บันทึกข้อมูลที่รวมแล้วลงไฟล์ CSV
merged_df.to_csv("merged_stock_sentiment_financial.csv", index=False)

# แสดงข้อมูลที่รวมแล้ว
print(merged_df.head())
