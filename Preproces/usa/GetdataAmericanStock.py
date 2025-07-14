import yfinance as yf
import pandas as pd
import datetime
import sys
import os
from dotenv import load_dotenv
import io
from pandas_market_calendars import get_calendar
try:
    import mysql.connector
except ImportError:
    print("⚠️ mysql-connector-python not installed. Skipping database operations.")
    mysql = None

# ✅ ป้องกัน UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ✅ ตรวจสอบระดับของโฟลเดอร์
CURRENT_DIR = os.getcwd()
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.env')
load_dotenv(path)

# ✅ ดึงตัวแปรสภาพแวดล้อม
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# ✅ ตรวจสอบโครงสร้างตารางฐานข้อมูล (เพื่อดึงวันที่ล่าสุด)
def check_table_structure():
    if not mysql:
        return False
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = conn.cursor()
        cursor.execute("SHOW COLUMNS FROM StockDetail")
        columns = [col[0] for col in cursor.fetchall()]
        cursor.close()
        conn.close()
        expected_columns = ['Date', 'StockSymbol']
        missing_columns = [col for col in expected_columns if col not in columns]
        if missing_columns:
            print(f"❌ Missing columns in StockDetail: {missing_columns}")
            print("⚠️ Using default start date (2024-01-01) due to table issues.")
            return False
        print("✅ Table structure is sufficient for date checking")
        return True
    except Exception as e:
        print(f"❌ Error checking table structure: {e}")
        print("⚠️ Using default start date (2024-01-01) due to table issues.")
        return False

# ✅ ตรวจสอบวันที่ล่าสุดจากฐานข้อมูล (รองรับ Mock)
latest_dates = {}
tickers = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'TSM', 'AMD']
has_valid_table = False
today = datetime.datetime.now()
current_date = (today - datetime.timedelta(days=1)).strftime('%Y-%m-%d')  # 13 กรกฎาคม 2025

# ตัวเลือก Mock (สำหรับทดสอบ)
MOCK_MODE = False  # เปลี่ยนเป็น True เพื่อทดสอบ
if MOCK_MODE:
    latest_dates = {ticker: "2025-07-11" for ticker in tickers}  # จำลองวันที่ล่าสุด
else:
    if all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME]) and mysql:
        try:
            conn = mysql.connector.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                autocommit=True
            )
            cursor = conn.cursor()
            print("✅ เชื่อมต่อฐานข้อมูลสำเร็จ!")
            has_valid_table = check_table_structure()
            
            if has_valid_table:
                for ticker in tickers:
                    try:
                        cursor.execute("SELECT MAX(Date) FROM StockDetail WHERE StockSymbol = %s", (ticker,))
                        result = cursor.fetchone()[0]
                        if result is None:
                            latest_dates[ticker] = "2024-01-01"
                        elif result > today.date():
                            print(f"⚠️ Future date found for {ticker}: {result}. Using default start date (2024-01-01)")
                            latest_dates[ticker] = "2024-01-01"
                        else:
                            latest_dates[ticker] = (result + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                    except Exception as e:
                        print(f"⚠️ Error fetching latest date for {ticker}: {e}")
                        latest_dates[ticker] = "2024-01-01"
            else:
                for ticker in tickers:
                    latest_dates[ticker] = "2024-01-01"
            
            cursor.close()
            conn.close()
            print("🔹 ปิดการเชื่อมต่อฐานข้อมูลแล้ว")
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            for ticker in tickers:
                latest_dates[ticker] = "2024-01-01"
    else:
        print("⚠️ Missing database configuration or mysql-connector-python, using default start date (2024-01-01)")
        for ticker in tickers:
            latest_dates[ticker] = "2024-01-01"

# ✅ กำหนดวันที่เริ่มต้นและสิ้นสุด
start_date_db = min(latest_dates.values())
start_date = min(latest_dates.values())  # วันที่ล่าสุดจากฐานข้อมูล
print(f"🔹 วันที่เริ่มต้นสำหรับดึงข้อมูล: {start_date}")
end_date = current_date  # 13 กรกฎาคม 2025
print(f"🔹 วันที่สิ้นสุดสำหรับดึงข้อมูล: {end_date}")

# ปรับ start_date ให้ย้อนกลับ 10 วัน และจำกัดไม่ให้เกิน end_date
start_date = (pd.to_datetime(start_date) - datetime.timedelta(days=10)).strftime('%Y-%m-%d')
if pd.to_datetime(start_date) > pd.to_datetime(end_date):
    start_date = (pd.to_datetime(end_date) - datetime.timedelta(days=10)).strftime('%Y-%m-%d')
    if pd.to_datetime(start_date) < pd.to_datetime("2024-01-01"):
        start_date = "2024-01-01"

# ตรวจสอบว่ามีข้อมูลใหม่ให้ดึงหรือไม่
if start_date >= end_date:
    print(f"❌ ไม่มีข้อมูลใหม่ให้ดึง (start_date: {start_date} >= end_date: {end_date})")
    sys.exit(0)

print(f"🔹 ดึงข้อมูลจาก {start_date} ถึง {end_date}")

# ✅ ดึงปฏิทิน NYSE เพื่อระบุวันซื้อขาย
nyse = get_calendar('NYSE')
trading_days = nyse.schedule(start_date=start_date, end_date=end_date).index

# ✅ ฟังก์ชันตรวจสอบว่าเป็นวันซื้อขายหรือไม่
def is_trading_day(date, trading_days):
    return pd.Timestamp(date) in trading_days

# ✅ ฟังก์ชันเติมข้อมูลวันหยุดด้วย Forward Fill และ Rolling Mean
def impute_holiday_data(ticker_data, all_dates, ticker, window=3):
    ticker_data = ticker_data.copy()
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in ticker_data.columns for col in required_columns):
        print(f"❌ Missing required columns for {ticker}: {required_columns}")
        return pd.DataFrame()
    
    ticker_data.index = pd.to_datetime(ticker_data.index).tz_localize(None)
    ticker_data = ticker_data.reindex(all_dates, method=None)
    
    missing_percentage = ticker_data[required_columns].isnull().mean() * 100
    print(f"🔍 Missing data for {ticker}: {missing_percentage.to_dict()}")
    if missing_percentage.sum() > 20:
        print(f"⚠️ Warning: Excessive missing data for {ticker} ({missing_percentage.sum():.2f}%).")

    ticker_data[['Open', 'High', 'Low', 'Close']] = (
        ticker_data[['Open', 'High', 'Low', 'Close']]
        .ffill(limit=2)
        .bfill(limit=2)
        .rolling(window=window, min_periods=1).mean()
    )
    ticker_data['Volume'] = ticker_data['Volume'].fillna(0)
    ticker_data['Changepercent'] = (ticker_data['Close'] - ticker_data['Open']) / ticker_data['Open'] * 100
    ticker_data['Changepercent'] = ticker_data['Changepercent'].fillna(0)

    return ticker_data

# ✅ สร้างช่วงวันที่ทั้งหมด (รวมวันหยุด)
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# ✅ ดึงข้อมูลด้วย yfinance
max_retries = 3
data_dict = {}

for ticker in tickers:
    retry_count = 0
    while retry_count < max_retries:
        try:
            stock = yf.Ticker(ticker)
            ticker_data = stock.history(start=start_date, end=end_date, interval='1d')
            if not ticker_data.empty:
                print(f"✅ Retrieved data for {ticker}: {len(ticker_data)} rows")
                print(f"📋 Sample data for {ticker}:\n{ticker_data.head()}")
                ticker_data = impute_holiday_data(ticker_data, all_dates, ticker, window=3)
                ticker_data['Ticker'] = ticker
                data_dict[ticker] = ticker_data
            else:
                print(f"⚠️ No data retrieved for {ticker}")
            break
        except Exception as e:
            retry_count += 1
            print(f"⚠️ Error for {ticker}: {e} (ลองใหม่ {retry_count}/{max_retries})")
            if retry_count == max_retries:
                print(f"❌ Failed to retrieve data for {ticker} after {max_retries} attempts")
                break

if not data_dict:
    print("⚠️ ไม่มีข้อมูลใหม่จาก yfinance")
    sys.exit(0)

# ✅ ประมวลผลข้อมูลและเติมวันหยุด
data_list = []
for ticker, ticker_data in data_dict.items():
    if ticker_data.empty:
        print(f"⚠️ ไม่มีข้อมูลสำหรับ {ticker}")
        continue
    for date in ticker_data.index:
        if not is_trading_day(date, trading_days):
            print(f"⚠️ Note: Data for {ticker} on {date.strftime('%Y-%m-%d')} is imputed using Rolling Mean.")
    data_list.append(ticker_data)

if not data_list:
    print("❌ ไม่มีข้อมูลใด ๆ ที่ดึงมาได้")
    sys.exit(1)

# รวมข้อมูลทั้งหมดเป็น DataFrame เดียว
cleaned_data = pd.concat(data_list).reset_index()

# ✅ เปลี่ยนชื่อคอลัมน์ให้ตรงกับ CSV
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
columns_to_keep = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Changepercent']
cleaned_data = cleaned_data[columns_to_keep]

# ✅ กรองข้อมูลที่ไม่ถูกต้องและตัดวันที่
print("🔹 กรองข้อมูลที่ไม่ถูกต้องออก...")
before_filter = len(cleaned_data)
cleaned_data = cleaned_data[
    (cleaned_data['Open'].notna()) &
    (cleaned_data['High'].notna()) &
    (cleaned_data['Low'].notna()) &
    (cleaned_data['Close'].notna()) &
    (cleaned_data['Date'] >= start_date_db) &    # เพิ่มเงื่อนไขนี้
    (cleaned_data['Date'] <= end_date)
]
after_filter = len(cleaned_data)
print(f"🔹 กรองข้อมูลแล้ว: {before_filter} -> {after_filter} แถว")

# ✅ เรียงลำดับและลบข้อมูลซ้ำ
cleaned_data = cleaned_data.sort_values(['Date', 'Ticker']).reset_index(drop=True)
cleaned_data = cleaned_data.drop_duplicates(subset=['Date', 'Ticker'], keep='first')

# ✅ บันทึกข้อมูลเป็นไฟล์ CSV
output_path = os.path.join(os.path.dirname(__file__), "Stock", "stock_data_usa.csv")
cleaned_data.to_csv(output_path, index=False)
print(f"✅ บันทึกข้อมูลลงไฟล์ CSV สำเร็จ: {output_path}")

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