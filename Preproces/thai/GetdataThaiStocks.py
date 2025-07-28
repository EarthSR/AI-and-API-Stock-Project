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
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')

# ✅ ตรวจสอบระดับของโฟลเดอร์
CURRENT_DIR = os.getcwd()
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.env')
load_dotenv(path)

# ✅ ดึงตัวแปรสภาพแวดล้อม
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME]) and mysql:
    raise ValueError("❌ ขาดค่าการตั้งค่าฐานข้อมูลในไฟล์ .env")

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
            print("⚠️ Using default start date (2018-01-01) due to table issues.")
            return False
        print("✅ Table structure is sufficient for date checking")
        return True
    except Exception as e:
        print(f"❌ Error checking table structure: {e}")
        print("⚠️ Using default start date (2018-01-01) due to table issues.")
        return False

# ✅ ตรวจสอบวันที่ล่าสุดจากฐานข้อมูล
latest_dates = {}
tickers = ['ADVANC.BK', 'TRUE.BK', 'DITTO.BK', 'DIF.BK', 
           'INSET.BK', 'JMART.BK', 'INET.BK', 'JAS.BK', 'HUMAN.BK']
has_valid_table = False
today = datetime.datetime.now()
current_date = (today - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

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
                stock_name = ticker.replace('.BK', '')
                try:
                    cursor.execute("SELECT MAX(Date) FROM StockDetail WHERE StockSymbol = %s", (stock_name,))
                    result = cursor.fetchone()[0]
                    if result is None:
                        latest_dates[ticker] = "2018-01-01"
                    elif result > today.date():
                        print(f"⚠️ Future date found for {ticker}: {result}. Using default start date (2018-01-01)")
                        latest_dates[ticker] = "2018-01-01"
                    else:
                        latest_dates[ticker] = (result + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                except Exception as e:
                    print(f"⚠️ Error fetching latest date for {ticker}: {e}")
                    latest_dates[ticker] = "2018-01-01"
        else:
            for ticker in tickers:
                latest_dates[ticker] = "2018-01-01"
        
        cursor.close()
        conn.close()
        print("🔹 ปิดการเชื่อมต่อฐานข้อมูลแล้ว")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        for ticker in tickers:
            latest_dates[ticker] = "2018-01-01"
else:
    print("⚠️ Missing database configuration or mysql-connector-python, using default start date (2018-01-01)")
    for ticker in tickers:
        latest_dates[ticker] = "2018-01-01"

# ✅ กำหนดวันที่เริ่มต้นและสิ้นสุด
start_date_db = min(latest_dates.values())
start_date = (pd.to_datetime(min(latest_dates.values())) + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
end_date = current_date

# ตรวจสอบว่ามีข้อมูลใหม่ให้ดึงหรือไม่
if start_date >= end_date:
    print(f"❌ ไม่มีข้อมูลใหม่ให้ดึง (start_date: {start_date} >= end_date: {end_date})")
    sys.exit(0)

print(f"🔹 ดึงข้อมูลจาก {start_date} ถึง {end_date}")

# ✅ ดึงปฏิทิน SET เพื่อระบุวันซื้อขาย
try:
    set_calendar = get_calendar('XBKK')
    trading_days = set_calendar.schedule(start_date=start_date, end_date=end_date).index
    print("✅ ใช้ปฏิทิน SET (XBKK) สำเร็จ")
except Exception as e:
    print(f"⚠️ ไม่สามารถใช้ปฏิทิน SET ได้: {e}")
    # ใช้วันทำการทั่วไปแทน (จันทร์-ศุกร์)
    all_dates_range = pd.date_range(start=start_date, end=end_date, freq='D')
    trading_days = all_dates_range[all_dates_range.weekday < 5]  # 0=จันทร์, 4=ศุกร์
    print("✅ ใช้วันทำการทั่วไป (จันทร์-ศุกร์) แทน")

# ✅ ฟังก์ชันตรวจสอบว่าเป็นวันซื้อขายหรือไม่
def is_trading_day(date, trading_days):
    return pd.Timestamp(date) in trading_days

# ✅ ฟังก์ชันเติมข้อมูลวันหยุดด้วย Forward Fill และ Rolling Mean (ปรับปรุงแล้ว)
def impute_holiday_data(ticker_data, all_dates, ticker, window=3):
    print(f"🔄 กำลังเติมข้อมูลวันหยุดสำหรับ {ticker}...")
    ticker_data = ticker_data.copy()
    
    # ตรวจสอบคอลัมน์ที่จำเป็น
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in ticker_data.columns for col in required_columns):
        print(f"❌ Missing required columns for {ticker}: {required_columns}")
        return pd.DataFrame()
    
    # แปลง index เป็น datetime และลบ timezone
    ticker_data.index = pd.to_datetime(ticker_data.index).tz_localize(None)
    
    print(f"📊 {ticker} - ข้อมูลดิบ: {len(ticker_data)} แถว")
    print(f"📊 {ticker} - ช่วงวันที่ดิบ: {ticker_data.index.min()} ถึง {ticker_data.index.max()}")
    
    # Reindex เพื่อรวมวันหยุด
    ticker_data = ticker_data.reindex(all_dates, method=None)
    print(f"📊 {ticker} - หลัง reindex: {len(ticker_data)} แถว")
    
    # คำนวณเปอร์เซ็นต์ข้อมูลที่หายไป
    missing_percentage = ticker_data[required_columns].isnull().mean() * 100
    print(f"🔍 {ticker} - ข้อมูลที่หายไป: Open: {missing_percentage['Open']:.1f}%, High: {missing_percentage['High']:.1f}%, Low: {missing_percentage['Low']:.1f}%, Close: {missing_percentage['Close']:.1f}%, Volume: {missing_percentage['Volume']:.1f}%")
    
    if missing_percentage.sum() > 50:
        print(f"⚠️ Warning: ข้อมูลหายไปมากสำหรับ {ticker} ({missing_percentage.sum():.2f}%)")
    
    # เติมข้อมูลราคาด้วย Forward Fill, Backward Fill และ Rolling Mean
    print(f"🔄 {ticker} - เติมข้อมูลราคาด้วย Forward Fill...")
    ticker_data[['Open', 'High', 'Low', 'Close']] = ticker_data[['Open', 'High', 'Low', 'Close']].ffill(limit=5)
    
    print(f"🔄 {ticker} - เติมข้อมูลราคาด้วย Backward Fill...")
    ticker_data[['Open', 'High', 'Low', 'Close']] = ticker_data[['Open', 'High', 'Low', 'Close']].bfill(limit=5)
    
    print(f"🔄 {ticker} - เติมข้อมูลราคาด้วย Rolling Mean (window={window})...")
    # ใช้ rolling mean สำหรับข้อมูลที่ยังหายไปอยู่
    for col in ['Open', 'High', 'Low', 'Close']:
        missing_mask = ticker_data[col].isnull()
        if missing_mask.any():
            # คำนวณ rolling mean และเติมเฉพาะตำแหน่งที่หายไป
            rolling_mean = ticker_data[col].rolling(window=window, min_periods=1, center=True).mean()
            ticker_data.loc[missing_mask, col] = rolling_mean.loc[missing_mask]
    
    # เติม Volume ด้วย 0 หรือค่าเฉลี่ย
    print(f"🔄 {ticker} - เติมข้อมูล Volume...")
    ticker_data['Volume'] = ticker_data['Volume'].fillna(0)
    
    # คำนวณ Changepercent
    print(f"🔄 {ticker} - คำนวณ Changepercent...")
    ticker_data['Changepercent'] = ((ticker_data['Close'] - ticker_data['Open']) / ticker_data['Open'] * 100).fillna(0)
    
    # ตรวจสอบผลลัพธ์หลังเติมข้อมูล
    final_missing = ticker_data[required_columns].isnull().mean() * 100
    print(f"✅ {ticker} - ข้อมูลหายไปหลังเติม: Open: {final_missing['Open']:.1f}%, High: {final_missing['High']:.1f}%, Low: {final_missing['Low']:.1f}%, Close: {final_missing['Close']:.1f}%, Volume: {final_missing['Volume']:.1f}%")
    
    # นับจำนวนวันที่เติมข้อมูล
    original_dates = set(ticker_data.dropna(subset=['Open']).index)
    all_dates_set = set(ticker_data.index)
    imputed_dates = all_dates_set - original_dates
    print(f"📈 {ticker} - เติมข้อมูลแล้ว {len(imputed_dates)} วัน จากทั้งหมด {len(all_dates_set)} วัน")

    return ticker_data

# ✅ สร้างช่วงวันที่ทั้งหมด (รวมวันหยุด)
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
print(f"📅 สร้างช่วงวันที่: {len(all_dates)} วัน (จาก {start_date} ถึง {end_date})")

# ✅ ดึงข้อมูลด้วย yfinance
max_retries = 3
data_dict = {}

print("🔄 เริ่มดึงข้อมูลจาก yfinance...")
for ticker in tickers:
    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"📡 กำลังดึงข้อมูล {ticker}...")
            stock = yf.Ticker(ticker)
            ticker_data = stock.history(start=start_date, end=end_date, interval='1d')
            if not ticker_data.empty:
                print(f"✅ ดึงข้อมูล {ticker} สำเร็จ: {len(ticker_data)} แถว")
                print(f"📋 ตัวอย่างข้อมูล {ticker}:")
                print(ticker_data.head(3).to_string())
                
                # เติมข้อมูลวันหยุด
                ticker_data = impute_holiday_data(ticker_data, all_dates, ticker, window=3)  
                stock_name = ticker.replace('.BK', '')
                ticker_data['Ticker'] = stock_name
                data_dict[ticker] = ticker_data
                print(f"✅ เติมข้อมูลวันหยุดสำหรับ {ticker} เสร็จสิ้น")
            else:
                print(f"⚠️ ไม่พบข้อมูลสำหรับ {ticker}")
            break
        except Exception as e:
            retry_count += 1
            print(f"⚠️ Error for {ticker}: {e} (ลองใหม่ {retry_count}/{max_retries})")
            if retry_count == max_retries:
                print(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้หลังจากลองแล้ว {max_retries} ครั้ง")
                break

if not data_dict:
    print("⚠️ ไม่มีข้อมูลใหม่จาก yfinance")
    sys.exit(0)

# ✅ ประมวลผลข้อมูลและแสดงสถิติการเติมข้อมูล
data_list = []
total_imputed_days = 0

print("\n🔍 ตรวจสอบข้อมูลที่เติมในวันหยุด:")
for ticker, ticker_data in data_dict.items():
    if ticker_data.empty:
        print(f"⚠️ ไม่มีข้อมูลสำหรับ {ticker}")
        continue
    
    # นับวันที่เติมข้อมูล
    imputed_count = 0
    for date in ticker_data.index:
        if not is_trading_day(date, trading_days):
            if not ticker_data.loc[date, ['Open', 'High', 'Low', 'Close']].isnull().any():
                imputed_count += 1
                if imputed_count <= 3:  # แสดงเฉพาะ 3 วันแรก
                    print(f"📝 {ticker} - {date.strftime('%Y-%m-%d')} (วันหยุด): เติมข้อมูลด้วยค่าเฉลี่ย")
    
    if imputed_count > 3:
        print(f"📝 {ticker} - เติมข้อมูลวันหยุดทั้งหมด {imputed_count} วัน")
    
    total_imputed_days += imputed_count
    data_list.append(ticker_data)

print(f"\n✅ เติมข้อมูลวันหยุดทั้งหมด: {total_imputed_days} วัน")

if not data_list:
    print("❌ ไม่มีข้อมูลใด ๆ ที่ดึงมาได้")
    sys.exit(1)

# รวมข้อมูลทั้งหมดเป็น DataFrame เดียว
print("🔄 รวมข้อมูลทั้งหมด...")
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
    (cleaned_data['Date'] >= start_date_db) &
    (cleaned_data['Date'] <= end_date)
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
csv_path = os.path.join(os.path.dirname(__file__), "Stock", "stock_data_thai.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
cleaned_data.to_csv(csv_path, index=False)
print(f"✅ บันทึกไฟล์ CSV สำเร็จที่: {csv_path}")

# ✅ แสดงสถิติข้อมูล
print(f"\n📊 สรุปผลลัพธ์:")
print(f"🔹 จำนวนข้อมูลทั้งหมด: {len(cleaned_data)} แถว")
print(f"🔹 วันที่ที่มีข้อมูล: {cleaned_data['Date'].nunique()} วัน")
if not cleaned_data.empty:
    print(f"🔹 ช่วงวันที่: {cleaned_data['Date'].min()} ถึง {cleaned_data['Date'].max()}")
    for ticker in cleaned_data['Ticker'].unique():
        ticker_data = cleaned_data[cleaned_data['Ticker'] == ticker]
        trading_days_count = len([d for d in ticker_data['Date'] if is_trading_day(pd.to_datetime(d), trading_days)])
        holiday_days_count = len(ticker_data) - trading_days_count
        print(f"🔹 {ticker}: {len(ticker_data)} แถว (วันซื้อขาย: {trading_days_count}, วันหยุดที่เติม: {holiday_days_count}), วันที่ล่าสุด {ticker_data['Date'].max()}")
    
    print("\n📋 ตัวอย่างข้อมูลที่เติมวันหยุด:")
    # แสดงตัวอย่างข้อมูลวันหยุดที่เติม
    sample_data = cleaned_data.head(15)
    for _, row in sample_data.iterrows():
        date_status = "วันซื้อขาย" if is_trading_day(pd.to_datetime(row['Date']), trading_days) else "วันหยุด(เติม)"
        print(f"  {row['Date']} | {row['Ticker']} | Close: {row['Close']:.2f} | ({date_status})")
else:
    print("❌ ไม่มีข้อมูลที่ถูกต้องในช่วงวันที่ที่กำหนด")