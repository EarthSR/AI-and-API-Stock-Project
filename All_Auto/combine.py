import pandas as pd
import sys
from datetime import datetime

# เอาเวลาปัจจุบัน (ตามเวลาท้องถิ่นเครื่องที่รันโค้ด)
current_hour = datetime.now().hour

if 8 <= current_hour < 18:
    print("📊 กำลังประมวลผลตลาดหุ้นไทย...")
    sentiment_df = pd.read_csv("./News/daily_sentiment_result_th.csv")
    stock_df = pd.read_csv("./Stock/stock_data_thai.csv")
    financial_df = pd.read_csv("./Stock/Financial_Thai_Quarter.csv")
elif 19 <= current_hour or current_hour < 5:
    print("📊 กำลังประมวลผลตลาดหุ้นอเมริกา...")
    sentiment_df = pd.read_csv("./News/daily_sentiment_result_us.csv")
    stock_df = pd.read_csv("./Stock/stock_data_usa.csv")
    financial_df = pd.read_csv("./Stock/Financial_America_Quarter.csv")
else:
    print("❌ ไม่อยู่ในช่วงเวลาทำการของตลาดหุ้นไทยหรืออเมริกา")
    sys.exit()

# ลบช่องว่างหัวท้ายชื่อคอลัมน์
financial_df.columns = financial_df.columns.str.strip()

# rename "EV / EBITDA" -> "EVEBITDA"
financial_df.rename(columns={"EV / EBITDA": "EVEBITDA"}, inplace=True)

# --------------------
# (1) เตรียม DataFrame sentiment_df
# --------------------
# ถ้าไฟล์ sentiment_df ไม่มีคอลัมน์ 'Sentiment Category' ให้สร้างเป็น 'Neutral'
if 'Sentiment Category' not in sentiment_df.columns:
    sentiment_df['Sentiment Category'] = 'Neutral'

# จากนั้น rename 'Sentiment Category' -> 'Sentiment'
sentiment_df.rename(columns={"Sentiment Category": "Sentiment"}, inplace=True)

# แปลงคอลัมน์ date เป็น datetime
sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])

# --------------------
# (2) Merge sentiment เข้ากับ stock
# --------------------
stock_df["Date"] = pd.to_datetime(stock_df["Date"])
merged_df = stock_df.merge(
    sentiment_df[['date', 'Sentiment']],  # ใช้เฉพาะสองคอลัมน์นี้พอ
    left_on='Date',
    right_on='date',
    how='left'
)
merged_df.drop(columns=['date'], inplace=True)  # ลบ date ซ้ำ

# ตรวจสอบว่าคอลัมน์ 'Quarter' มีไหม ถ้าไม่มีให้สร้าง
if 'Quarter' not in merged_df.columns:
    merged_df['Quarter'] = merged_df['Date'].dt.to_period('Q').astype(str)

# --------------------
# (3) เตรียม DataFrame financial_df ตามปกติ
# --------------------
financial_df.rename(columns={'Sentiment Category': 'Sentiment'}, inplace=True)  # <== ถ้าในไฟล์ finance มี "Sentiment Category" จริงค่อย rename
columns_to_keep = [
    'Stock', 'Quarter', 'QoQ Growth (%)', 'Total Revenue', 'YoY Growth (%)', 
    'Net Profit', 'Earnings Per Share (EPS)', 'ROA (%)', 'ROE (%)', 'Gross Margin (%)', 
    'Net Profit Margin (%)', 'Debt to Equity (x)', 'P/E Ratio (x)', 'P/BV Ratio (x)', 
    'Dividend Yield (%)', 'EVEBITDA', 'MarketCap'
]
financial_df = financial_df[columns_to_keep]

def fix_quarter_format(quarter_str):
    quarter_str = str(quarter_str).strip()
    if len(quarter_str) == 6 and quarter_str[0].isdigit() and quarter_str[1] == "Q":
        return quarter_str[-4:] + "Q" + quarter_str[0]
    return quarter_str

def quarter_to_announcement_date(quarter_str):
    try:
        year, q = int(quarter_str[:4]), int(quarter_str[-1])
        quarter_dates = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
        if q == 4:
            year += 1
        return pd.Timestamp(f"{year}-{quarter_dates[q]}")
    except:
        return pd.NaT

financial_df.dropna(axis=1, how='all', inplace=True)
financial_df['Quarter'] = financial_df['Quarter'].apply(fix_quarter_format)
financial_df['Quarter Date'] = financial_df['Quarter'].apply(quarter_to_announcement_date)

# --------------------
# (4) Merge financial_df
# --------------------
merged_df = merged_df.merge(
    financial_df,
    left_on=['Ticker', 'Quarter'],
    right_on=['Stock', 'Quarter'],
    how='left'
)

# --------------------
# (5) จัดการคอลัมน์ Sentiment ที่ยังเป็น NaN => Neutral
# --------------------
merged_df['Sentiment'] = merged_df['Sentiment'].fillna("Neutral")

# --------------------
# (6) ลบคอลัมน์ *Category* ทั้งหมด (ไม่มีใน merged_df แล้ว แต่หากยังมีเกิดจากกรณีอื่น ก็สามารถ drop ได้)
# --------------------
for col in ['Sentiment Category', 'Sentiment Category_x', 'Sentiment Category_y']:
    if col in merged_df.columns:
        merged_df.drop(columns=[col], inplace=True)

# --------------------
# (7) เติมค่าและ Clean อื่น ๆ ตามเดิม
# --------------------
# ตัวอย่างเช็ค *_x, *_y
for col in merged_df.columns:
    if col.endswith('_x'):
        y_col = col.replace('_x', '_y')
        if y_col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(merged_df[y_col])
            merged_df.drop(columns=[y_col], inplace=True)

merged_df.columns = merged_df.columns.str.replace(r'(_x|\(x\))', '', regex=True)

front_columns = [col for col in ['Date', 'Ticker', 'Quarter Date'] if col in merged_df.columns]
other_columns = [col for col in merged_df.columns if col not in front_columns]
merged_df = merged_df[front_columns + other_columns]

def clean_data_based_on_dates(df):
    columns_to_clean = [
        'QoQ Growth (%)', 'Total Revenue', 'YoY Growth (%)', 'Net Profit', 
        'Earnings Per Share (EPS)', 'ROA (%)', 'ROE (%)', 'Gross Margin (%)', 
        'Net Profit Margin (%)', 'Debt to Equity (x)', 'P/E Ratio (x)', 'P/BV Ratio (x)', 
        'Dividend Yield (%)', 'Debt to Equity ', 'P/E Ratio ', 'P/BV Ratio ',
        'EVEBITDA'
    ]
    for col in columns_to_clean:
        if col in df.columns:
            df.loc[df['Date'] != df['Quarter Date'], col] = None
    return df

merged_df = clean_data_based_on_dates(merged_df)

financial_columns = [
    'QoQ Growth (%)', 'Total Revenue', 'YoY Growth (%)', 'Net Profit', 
    'Earnings Per Share (EPS)', 'ROA (%)', 'ROE (%)', 'Gross Margin (%)', 
    'Net Profit Margin (%)', 'Debt to Equity (x)', 'P/E Ratio (x)', 'P/BV Ratio (x)', 
    'Dividend Yield (%)', 'Debt to Equity ', 'P/E Ratio ', 'P/BV Ratio '
]

# สร้างลิสต์ใหม่เก็บเฉพาะคอลัมน์ที่ merged_df มีอยู่จริง
valid_financial_cols = [col for col in financial_columns if col in merged_df.columns]

# จากนั้นจึงเรียกใช้งาน valid_financial_cols แทน financial_columns
merged_df = merged_df[
    ~(
        (merged_df['Close'] == 0)
        & (merged_df[valid_financial_cols].isna().all(axis=1))
    )
]

columns_to_remove = ["Debt to Equity (x)", "P/E Ratio (x)", "P/BV Ratio (x)", "Quarter"]
merged_df.drop(columns=columns_to_remove, errors='ignore', inplace=True)

# เติม MarketCap กรณี Date == Quarter Date
if 'MarketCap' in merged_df.columns:
    merged_df['MarketCap'] = merged_df.apply(
        lambda row: row['MarketCap'] if row['Date'] == row['Quarter Date'] else None, axis=1
    )

merged_df.to_csv("merged_stock_sentiment_financial.csv", index=False)
print(merged_df.head())

# เติม Quarter Date หรือ Stock หากขาด
merged_df['Quarter Date'] = merged_df['Quarter Date'].fillna(
    merged_df['Date'].dt.to_period('Q').astype(str)
)
if 'Stock' in merged_df.columns:
    merged_df['Stock'] = merged_df['Stock'].fillna(merged_df['Ticker'])

print("ข้อมูลหลังเติม Quarter Date และ Stock:")
print(merged_df[['Date', 'Ticker', 'Quarter Date', 'Stock']].head())

merged_df.to_csv("merged_stock_sentiment_financial.csv", index=False)
