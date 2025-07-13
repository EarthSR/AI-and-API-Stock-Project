import yfinance as yf
import pandas as pd
import datetime
import sys
import os
import mysql.connector
from dotenv import load_dotenv
from pandas_market_calendars import get_calendar

# ✅ ป้องกัน UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# ✅ ตรวจสอบระดับของโฟลเดอร์
CURRENT_DIR = os.getcwd()
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.env')
load_dotenv(path)

# ✅ ดึงตัวแปรสภาพแวดล้อม
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME]):
    raise ValueError("❌ ขาดค่าการตั้งค่าฐานข้อมูลในไฟล์ .env")

# ✅ เชื่อมต่อฐานข้อมูล
conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME,
    autocommit=True
)
cursor = conn.cursor()
print("✅ เชื่อมต่อฐานข้อมูลสำเร็จ!")

# ✅ กำหนดรายชื่อหุ้นไทย
tickers = ['ADVANC.BK', 'TRUE.BK', 'DITTO.BK', 'DIF.BK', 
           'INSET.BK', 'JMART.BK', 'INET.BK', 'JAS.BK', 'HUMAN.BK']

# ✅ ฟังก์ชันตรวจสอบวันหยุด
def is_holiday(date, trading_days):
    return pd.Timestamp(date) not in trading_days

# ✅ ฟังก์ชันเติมข้อมูลวันหยุดด้วย Rolling Mean
def impute_holiday_data(ticker_data, window=3):
    ticker_data = ticker_data.copy()
    ticker_data.index = pd.to_datetime(ticker_data.index)
    all_dates = pd.date_range(start=ticker_data.index.min(), end=ticker_data.index.max(), freq='D')
    ticker_data = ticker_data.reindex(all_dates)

    # ตรวจสอบข้อมูลที่ขาดหาย
    missing_percentage = ticker_data[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().mean() * 100
    if missing_percentage.sum() > 20:
        print(f"⚠️ Warning: Excessive missing data ({missing_percentage.sum():.2f}%).")

    # เติมข้อมูลด้วย Rolling Mean เฉพาะราคา
    ticker_data[['Open', 'High', 'Low', 'Close']] = (
        ticker_data[['Open', 'High', 'Low', 'Close']]
        .ffill(limit=2)  # Forward Fill สำหรับช่องว่างสั้น ๆ
        .rolling(window=window, min_periods=1).mean()
    )

    # ตั้ง Volume และ Changepercent สำหรับวันหยุด
    ticker_data['Volume'] = ticker_data['Volume'].fillna(0)
    ticker_data['Changepercent'] = (ticker_data['Close'] - ticker_data['Open']) / ticker_data['Open'] * 100
    ticker_data['Changepercent'] = ticker_data['Changepercent'].fillna(0)

    return ticker_data

# ✅ ตรวจสอบวันที่ล่าสุดจากฐานข้อมูล
latest_dates = {}
end_date = datetime.datetime.today().strftime('%Y-%m-%d')  # วันที่ปัจจุบัน

for ticker in tickers:
    stock_name = ticker.replace('.BK', '')
    cursor.execute("SELECT MAX(Date) FROM StockDetail WHERE StockSymbol = %s", (stock_name,))
    result = cursor.fetchone()[0]
    if result is None:
        latest_dates[ticker] = "2018-01-01"
    else:
        latest_dates[ticker] = (result + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

# ✅ ปิดการเชื่อมต่อฐานข้อมูล
cursor.close()
conn.close()
print("🔹 ปิดการเชื่อมต่อฐานข้อมูลแล้ว")

# ✅ กำหนดวันที่เริ่มต้น
start_date = min(latest_dates.values())

# ตรวจสอบว่ามีข้อมูลใหม่ให้ดึงหรือไม่
if start_date > end_date:
    print(f"❌ ไม่มีข้อมูลใหม่ให้ดึง (start_date: {start_date} > end_date: {end_date})")
    sys.exit(0)

print(f"🔹 ดึงข้อมูลจาก {start_date} ถึง {end_date}")

# ✅ ดึงปฏิทิน SET เพื่อระบุวันหยุด
set_calendar = get_calendar('XBKK')
trading_days = set_calendar.schedule(start_date=start_date, end_date=end_date).index

# ✅ ดึงข้อมูลด้วย yfinance
max_retries = 3
retry_count = 0
data_dict = {}

while retry_count < max_retries:
    try:
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            ticker_data = stock.history(start=start_date, end=end_date, interval='1d')
            if not ticker_data.empty:
                data_dict[ticker] = ticker_data
        if not data_dict:
            print("⚠️ ไม่มีข้อมูลใหม่จาก yfinance")
            sys.exit(0)
        break
    except Exception as e:
        retry_count += 1
        print(f"⚠️ Error: {e} (ลองใหม่ {retry_count}/{max_retries})")
        if retry_count == max_retries:
            print("❌ ไม่สามารถดึงข้อมูลจาก yfinance ได้")
            sys.exit(1)

# ✅ ประมวลผลข้อมูลและเติมวันหยุด
data_list = []
for ticker, ticker_data in data_dict.items():
    if ticker_data.empty:
        print(f"⚠️ ไม่มีข้อมูลสำหรับ {ticker}")
        continue
    
    # ✅ Reindex เป็น freq='D' และใช้ Rolling Mean
    ticker_data = impute_holiday_data(ticker_data, window=3)
    stock_name = ticker.replace('.BK', '')
    ticker_data['Ticker'] = stock_name
    
    # ✅ แจ้งเตือนสำหรับวันหยุด
    for date in ticker_data.index:
        if is_holiday(date, trading_days):
            print(f"⚠️ Note: Data for {ticker} on {date.strftime('%Y-%m-%d')} is imputed using Rolling Mean.")

    data_list.append(ticker_data)

# ตรวจสอบว่ามีข้อมูลหรือไม่
if not data_list:
    print("❌ ไม่มีข้อมูลใด ๆ ที่ดึงมาได้")
    sys.exit(1)

# รวมข้อมูลทั้งหมดเป็น DataFrame เดียว
cleaned_data = pd.concat(data_list).reset_index()

# ✅ Merge กับข้อมูล Sentiment (สมมติว่ามีไฟล์ daily_sentiment_summary.csv)
try:
    sentiment_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "thai", "Stock", "daily_sentiment_summary.csv"))
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    sentiment_df = sentiment_df.set_index(['Date', 'Ticker']).reindex(
        pd.MultiIndex.from_product([cleaned_data['Date'].unique(), [t.replace('.BK', '') for t in tickers]], 
                                  names=['Date', 'Ticker'])
    ).reset_index()
    sentiment_df[['net_sentiment_score', 'Sentiment']] = sentiment_df[['net_sentiment_score', 'Sentiment']].ffill()

    cleaned_data = cleaned_data.merge(
        sentiment_df[['Date', 'Ticker', 'net_sentiment_score', 'Sentiment']],
        on=['Date', 'Ticker'],
        how='left'
    )
    print("✅ Merge ข้อมูล Sentiment สำเร็จ")
except FileNotFoundError:
    print("⚠️ ไม่พบไฟล์ daily_sentiment_summary.csv ข้ามการ Merge ข้อมูล Sentiment")
    cleaned_data['net_sentiment_score'] = pd.NA
    cleaned_data['Sentiment'] = pd.NA

# ✅ เปลี่ยนชื่อคอลัมน์ให้ตรงกับฐานข้อมูล
cleaned_data = cleaned_data.rename(columns={
    'index': 'Date',
    'Ticker': 'Ticker',
    'Open': 'Open',
    'High': 'High',
    'Low': 'Low',
    'Close': 'Close',
    'Volume': 'Volume',
    'Changepercent': 'Changepercent'
})

cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date']).dt.strftime('%Y-%m-%d')

# ✅ จัดเรียงคอลัมน์
columns_to_keep = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Changepercent', 'net_sentiment_score', 'Sentiment']
cleaned_data = cleaned_data[columns_to_keep]

# ✅ กรองข้อมูลที่ไม่ถูกต้อง
print("🔹 กรองข้อมูลที่ไม่ถูกต้องออก...")
before_filter = len(cleaned_data)
cleaned_data = cleaned_data[
    (cleaned_data['Open'] > 0) &
    (cleaned_data['High'] > 0) &
    (cleaned_data['Low'] > 0) &
    (cleaned_data['Close'] > 0) &
    (cleaned_data['Open'].notna()) &
    (cleaned_data['High'].notna()) &
    (cleaned_data['Low'].notna()) &
    (cleaned_data['Close'].notna())
]
after_filter = len(cleaned_data)
print(f"🔹 กรองข้อมูลแล้ว: {before_filter} -> {after_filter} แถว")

# ✅ เรียงลำดับและลบข้อมูลซ้ำ
cleaned_data = cleaned_data.sort_values(['Date', 'Ticker']).reset_index(drop=True)
cleaned_data = cleaned_data.drop_duplicates(subset=['Date', 'Ticker'], keep='first')

# ✅ ลบข้อมูลของวันนี้ (ถ้าต้องการ)
today = datetime.datetime.today().strftime('%Y-%m-%d')
initial_rows = len(cleaned_data)
cleaned_data = cleaned_data[cleaned_data['Date'].astype(str) != today]
if len(cleaned_data) < initial_rows:
    print(f"🔹 ลบข้อมูลของวันนี้ ({today}) แล้ว")

# ✅ บันทึกข้อมูลเป็นไฟล์ CSV
csv_path = os.path.join(CURRENT_DIR, "thai", "Stock", "stock_data_thai.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
cleaned_data.to_csv(csv_path, index=False)
print(f"✅ บันทึกไฟล์ CSV สำเร็จที่: {csv_path}")

# ✅ แสดงสถิติข้อมูล
print(f"🔹 จำนวนข้อมูลทั้งหมด: {len(cleaned_data)} แถว")
print(f"🔹 วันที่ที่มีข้อมูล: {cleaned_data['Date'].nunique()} วัน")
if not cleaned_data.empty:
    print(f"🔹 ช่วงวันที่: {cleaned_data['Date'].min()} ถึง {cleaned_data['Date'].max()}")
    for ticker in cleaned_data['Ticker'].unique():
        ticker_data = cleaned_data[cleaned_data['Ticker'] == ticker]
        print(f"🔹 {ticker}: {len(ticker_data)} แถว, วันที่ล่าสุด {ticker_data['Date'].max()}")
    print("\n📋 ตัวอย่างข้อมูล:")
    print(cleaned_data.head(10))
else:
    print("❌ ไม่มีข้อมูลที่ถูกต้องในช่วงวันที่ที่กำหนด")

# ✅ เพิ่มการแจ้งเตือนสำหรับการใช้งานจริง
print("\n⚠️ Warning: This data includes imputed prices for non-trading days (e.g., weekends, holidays) using a 3-day Rolling Mean, as required by the LSTM/GRU model trained with freq='D'. Use with caution for real-world trading decisions.")