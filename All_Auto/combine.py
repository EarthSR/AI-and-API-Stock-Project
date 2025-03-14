import pandas as pd
import sys
import os

# ✅ ป้องกัน UnicodeEncodeError (ข้ามอีโมจิที่ไม่รองรับ)
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# โหลดข้อมูล Sentiment
sentiment_df_th = pd.read_csv("./Finbert/daily_sentiment_result_th.csv")
sentiment_df_us = pd.read_csv("./Finbert/daily_sentiment_result_us.csv")

# โหลดข้อมูลหุ้น
stock_df_th = pd.read_csv("./Finbert/stock_data_thai.csv")
stock_df_us = pd.read_csv("./Finbert/stock_data_usa.csv")

# โหลดข้อมูลการเงิน
financial_thai_df = pd.read_csv("./Finbert/Financial_Thai_Quarter.csv")
financial_us_df = pd.read_csv("./Finbert/Financial_America_Quarter.csv")

print("🔍 คอลัมน์ใน Financial_Thai_Quarter.csv:", financial_thai_df.columns.tolist())
print("🔍 คอลัมน์ใน Financial_America_Quarter.csv:", financial_us_df.columns.tolist())

# ✅ ลบช่องว่างและแปลงชื่อคอลัมน์
financial_thai_df.columns = financial_thai_df.columns.str.strip()
financial_us_df.columns = financial_us_df.columns.str.strip()

# ✅ แปลง EV / EBITDA เป็น EVEBITDA
financial_thai_df.rename(columns={"EV / EBITDA": "EVEBITDA"}, inplace=True)
financial_us_df.rename(columns={"EV / EBITDA": "EVEBITDA"}, inplace=True)

# เปลี่ยนชื่อคอลัมน์ Sentiment Category เป็น Sentiment
sentiment_df_th.rename(columns={'Sentiment Category': 'Sentiment'}, inplace=True)
sentiment_df_us.rename(columns={'Sentiment Category': 'Sentiment'}, inplace=True)


# ✅ ตรวจสอบอีกครั้งว่าชื่อคอลัมน์ถูกต้อง
print("📌 คอลัมน์ใน Financial_Thai_Quarter.csv (หลัง Rename):", financial_thai_df.columns.tolist())
print("📌 คอลัมน์ใน Financial_America_Quarter.csv (หลัง Rename):", financial_us_df.columns.tolist())

# เพิ่ม MarketCap ใน columns_to_keep
columns_to_keep = [
    'Stock', 'Quarter', 'QoQ Growth (%)', 'Total Revenue', 'YoY Growth (%)', 
    'Net Profit', 'Earnings Per Share (EPS)', 'ROA (%)', 'ROE (%)', 'Gross Margin (%)', 
    'Net Profit Margin (%)', 'Debt to Equity (x)', 'P/E Ratio (x)', 'P/BV Ratio (x)', 
    'Dividend Yield (%)', 'EVEBITDA', 'MarketCap'  # เพิ่ม 'MarketCap'
]

# หลังจากที่โหลดข้อมูลการเงินแล้ว ให้เก็บเฉพาะคอลัมน์ที่ต้องการ
financial_thai_df = financial_thai_df[columns_to_keep]
financial_us_df = financial_us_df[columns_to_keep]


# ฟังก์ชันแปลง 4Q2024 → 2024Q4
def fix_quarter_format(quarter_str):
    quarter_str = str(quarter_str).strip()
    if len(quarter_str) == 6 and quarter_str[0].isdigit() and quarter_str[1] == "Q":
        return quarter_str[-4:] + "Q" + quarter_str[0]  # เปลี่ยน 4Q2024 → 2024Q4
    return quarter_str

# ฟังก์ชันแปลง Quarter เป็นวันที่ประกาศงบ (แก้ไขให้ถูกต้อง)
def quarter_to_announcement_date(quarter_str):
    try:
        year, q = int(quarter_str[:4]), int(quarter_str[-1])  # แยกปีและไตรมาส
        quarter_dates = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}  # เปลี่ยน Q1 เป็น ม.ค.
        if q == 4:  
            year += 1  # Q4 ใช้ปีถัดไป
        return pd.Timestamp(f"{year}-{quarter_dates[q]}")
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

# ✅ ตรวจสอบว่ามี 'Sentiment Category' ในไฟล์ CSV หรือไม่
if 'Sentiment Category' not in sentiment_df_th.columns:
    sentiment_df_th['Sentiment Category'] = 'Neutral'  # ถ้าไม่มี ให้สร้างขึ้นมา
if 'Sentiment Category' not in sentiment_df_us.columns:
    sentiment_df_us['Sentiment Category'] = 'Neutral'

# แปลงคอลัมน์ Quarter ให้เป็นรูปแบบ 2024Q4
financial_thai_df['Quarter'] = financial_thai_df['Quarter'].apply(fix_quarter_format)
financial_us_df['Quarter'] = financial_us_df['Quarter'].apply(fix_quarter_format)

# แปลงคอลัมน์ Quarter ให้เป็นวันที่ประกาศงบ
financial_thai_df['Quarter Date'] = financial_thai_df['Quarter'].apply(quarter_to_announcement_date)
financial_us_df['Quarter Date'] = financial_us_df['Quarter'].apply(quarter_to_announcement_date)

# ✅ รวมข้อมูลโดยใช้ Date เป็นตัวเชื่อม พร้อมเก็บ 'Sentiment Category'
merged_df_th = stock_df_th.merge(
    sentiment_df_th[['date', 'Sentiment', 'Sentiment Category']], 
    left_on='Date',
    right_on='date',
    how='left'
)

merged_df_us = stock_df_us.merge(
    sentiment_df_us[['date', 'Sentiment', 'Sentiment Category']], 
    left_on='Date',
    right_on='date',
    how='left'
)

# ✅ ลบคอลัมน์ 'date' ที่ซ้ำกันออก
merged_df_th.drop(columns=['date'], inplace=True)
merged_df_us.drop(columns=['date'], inplace=True)

# รวมข้อมูลของไทยและสหรัฐฯ เข้าด้วยกัน
merged_df = pd.concat([merged_df_th, merged_df_us], ignore_index=True)

# ตรวจสอบว่า 'Quarter' มีใน merged_df หรือไม่
if 'Quarter' not in merged_df.columns:
    # หากไม่มี 'Quarter' ให้เพิ่มคอลัมน์ Quarter ด้วยการใช้ Date เพื่อดึงไตรมาส
    merged_df['Quarter'] = merged_df['Date'].dt.to_period('Q').astype(str)

# รวมข้อมูลจาก financial_thai_df และ financial_us_df พร้อมทั้ง 'MarketCap'
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

# ตรวจสอบว่า MarketCap ถูกนำมารวมด้วยหรือไม่
if 'MarketCap' not in merged_df.columns:
    print("❌ ไม่พบคอลัมน์ MarketCap ใน merged_df")

# ✅ ถ้า 'Sentiment' เป็น NaN ให้ใช้ 'Sentiment Category' แทน
merged_df['Sentiment'] = merged_df['Sentiment'].fillna(merged_df['Sentiment Category'])

# ✅ ถ้ายังมี NaN ให้เติมค่า "Neutral"
merged_df['Sentiment'] = merged_df['Sentiment'].fillna("Neutral")

# ✅ ลบคอลัมน์ 'Sentiment Category' ทิ้งหลังจากรวมค่าเสร็จ
merged_df.drop(columns=['Sentiment Category'], inplace=True)


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

# จัดเรียงลำดับคอลัมน์ โดยให้ 'Date', 'Ticker', 'Quarter Date' อยู่ด้านหน้าเสมอ
front_columns = [col for col in ['Date', 'Ticker', 'Quarter Date'] if col in merged_df.columns]
other_columns = [col for col in merged_df.columns if col not in front_columns]
merged_df = merged_df[front_columns + other_columns]

def clean_data_based_on_dates(df):
    # กำหนดคอลัมน์ที่ต้องการลบข้อมูลหาก Date และ Quarter Date ไม่ตรงกัน
    columns_to_clean = [
        'QoQ Growth (%)', 'Total Revenue', 'YoY Growth (%)', 'Net Profit', 
        'Earnings Per Share (EPS)', 'ROA (%)', 'ROE (%)', 'Gross Margin (%)', 
        'Net Profit Margin (%)', 'Debt to Equity (x)', 'P/E Ratio (x)', 'P/BV Ratio (x)', 
        'Dividend Yield (%)',
        'Debt to Equity ', 'P/E Ratio ', 'P/BV Ratio ', 
        'EVEBITDA'  # ✅ เพิ่ม EVEBITDA เข้าไปในเงื่อนไขการลบข้อมูล
    ]
    
    # เช็คว่าค่าของ Date และ Quarter Date ตรงกันหรือไม่
    for col in columns_to_clean:
        df.loc[df['Date'] != df['Quarter Date'], col] = None  # ✅ กำหนดเป็น None ถ้า Date != Quarter Date
    return df

# เรียกใช้งานฟังก์ชัน Clean
merged_df = clean_data_based_on_dates(merged_df)

financial_columns = ['QoQ Growth (%)', 'Total Revenue', 'YoY Growth (%)', 'Net Profit', 
        'Earnings Per Share (EPS)', 'ROA (%)', 'ROE (%)', 'Gross Margin (%)', 
        'Net Profit Margin (%)', 'Debt to Equity (x)', 'P/E Ratio (x)', 'P/BV Ratio (x)', 
        'Dividend Yield (%)',
        'Debt to Equity ', 'P/E Ratio ', 'P/BV Ratio ']
# ลบแถวที่ Close == 0 และไม่มีข้อมูลงบการเงินเลย (ทุกค่าใน financial_columns เป็น NaN)
merged_df = merged_df[~((merged_df['Close'] == 0) & (merged_df[financial_columns].isna().all(axis=1)))]

# ไม่ลบ 'MarketCap' จาก merged_df
columns_to_remove = ["Debt to Equity (x)", "P/E Ratio (x)", "P/BV Ratio (x)", "Quarter"]
merged_df = merged_df.drop(columns=columns_to_remove, errors='ignore')

# เติม 'MarketCap' เฉพาะกรณีที่ Date ตรงกับ Quarter Date
merged_df['MarketCap'] = merged_df.apply(
    lambda row: row['MarketCap'] if row['Date'] == row['Quarter Date'] else None, axis=1
)

# บันทึกข้อมูลที่รวมแล้วลงไฟล์ CSV
merged_df.to_csv("merged_stock_sentiment_financial.csv", index=False)

# แสดงข้อมูลที่รวมแล้ว
print(merged_df.head())

# เพิ่มการเติมข้อมูลในคอลัมน์ Quarter Date และ Stock หลังจากโค้ดหลักทำงานเสร็จ
# เติมค่า 'Quarter Date' ที่ขาดหายไปตามวันที่ (Date)
merged_df['Quarter Date'] = merged_df['Quarter Date'].fillna(
    merged_df['Date'].dt.to_period('Q').astype(str)
)

# เติมค่า 'Stock' ที่ขาดหายไปจาก Ticker
merged_df['Stock'] = merged_df['Stock'].fillna(merged_df['Ticker'])

# ตรวจสอบข้อมูลที่เติม
print("ข้อมูลหลังจากเติมข้อมูล Quarter Date และ Stock:")
print(merged_df[['Date', 'Ticker', 'Quarter Date', 'Stock']].head())

# บันทึกข้อมูลที่รวมแล้วลงไฟล์ CSV
merged_df.to_csv("merged_stock_sentiment_financial.csv", index=False)

