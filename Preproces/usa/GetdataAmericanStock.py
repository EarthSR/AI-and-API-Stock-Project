import yfinance as yf
import pandas as pd
import datetime
import sys
import os
import mysql.connector
from dotenv import load_dotenv
import io

# ✅ ป้องกัน UnicodeEncodeError (ข้ามอีโมจิที่ไม่รองรับ)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ✅ ตรวจสอบระดับของโฟลเดอร์ (ปรับ `..` ตามตำแหน่งของไฟล์)
CURRENT_DIR = os.getcwd()

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.env')
load_dotenv(path)

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

# ✅ กำหนดรายชื่อหุ้นอเมริกา
tickers = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'TSM', 'AMD']

# ✅ ตรวจสอบวันที่ล่าสุดจากฐานข้อมูล
latest_dates = {}
# ✅ แก้ไข: ใช้วันที่เมื่อวาน เป็น end_date
today = datetime.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
end_date = (today - datetime.timedelta(days=1)).strftime('%Y-%m-%d')  # เมื่อวาน

for ticker in tickers:
    cursor.execute("SELECT MAX(Date) FROM StockDetail WHERE StockSymbol = %s", (ticker,))
    result = cursor.fetchone()[0]

    if result is None:
        latest_dates[ticker] = "2018-01-01"  # ถ้าไม่มีข้อมูลให้เริ่มจาก 2018-01-01
    else:
        latest_dates[ticker] = (result + datetime.timedelta(days=1)).strftime('%Y-%m-%d')  # เริ่มจากวันถัดไป

# ✅ ปิดการเชื่อมต่อฐานข้อมูล
cursor.close()
conn.close()
print("🔹 ปิดการเชื่อมต่อฐานข้อมูลแล้ว")

# ✅ กำหนดวันที่เริ่มต้นและสิ้นสุด (ใช้วันที่ล่าสุดจากฐานข้อมูล)
start_date = min(latest_dates.values())  # ใช้วันที่ที่เก่าสุดของหุ้นทั้งหมด

# ตรวจสอบว่ามีข้อมูลใหม่ให้ดึงหรือไม่
if start_date > end_date:
    print(f"❌ ไม่มีข้อมูลใหม่ให้ดึง (start_date: {start_date} > end_date: {end_date})")
    sys.exit(0)

print(f"🔹 ดึงข้อมูลจาก {start_date} ถึง {end_date}")

# ดึงข้อมูลราคาหุ้นจาก yfinance
max_retries = 3  # ✅ ลองใหม่ได้ 3 ครั้ง
retry_count = 0

while retry_count < max_retries:
    try:
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        if data.empty:
            print("⚠️ ไม่มีข้อมูลใหม่จาก yfinance")
            sys.exit(0)
        break  # ✅ ถ้าดึงได้สำเร็จ ให้ออกจาก loop ทันที
    except Exception as e:
        retry_count += 1
        print(f"⚠️ Error: {e} (ลองใหม่ {retry_count}/{max_retries})")
        if retry_count == max_retries:
            sys.exit(1)  # ❌ ถ้าลองครบแล้วยังไม่ได้ หยุดโปรแกรม

# สร้าง DataFrame สำหรับแต่ละหุ้น
data_list = []

for ticker in tickers:
    # ดึงข้อมูลราคาหุ้น และใช้ .copy() ป้องกัน SettingWithCopyWarning
    ticker_data = data[ticker].copy()
    
    # ✅ ตรวจสอบว่ามีข้อมูลหรือไม่
    if ticker_data.empty:
        print(f"⚠️ ไม่มีข้อมูลสำหรับ {ticker}")
        continue
    
    ticker_data['Ticker'] = ticker  # กำหนดค่า Ticker
    
    # รีอินเด็กซ์ให้มีทุกวัน (รวมเสาร์-อาทิตย์)
    ticker_data.index = pd.to_datetime(ticker_data.index)  # แปลงเป็น datetime index
    all_dates = pd.date_range(start=latest_dates[ticker], end=end_date, freq='D')  # ใช้ start_date ที่อัปเดตจากฐานข้อมูล
    ticker_data = ticker_data.reindex(all_dates)

    # 🔹 ใช้ค่า **วันก่อนหน้า** แทน NaN ก่อนเติม 0
    if ticker_data[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().sum().sum() > 0:
        print(f"⚠️ พบค่า NaN ในข้อมูลของ {ticker}, ใช้ค่าเฉลี่ยย้อนหลังเติมแทน")

    ticker_data[['Open', 'High', 'Low', 'Close', 'Volume']] = (
        ticker_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        .ffill()
        .rolling(window=3, min_periods=1).mean()
        .fillna(0)  # ✅ ถ้า ffill() ยังมีค่า NaN ให้เติม 0
    )

    # ✅ เพิ่มการคำนวณ Changepercent
    ticker_data['Changepercent'] = ((ticker_data['Close'] - ticker_data['Open']) / ticker_data['Open'] * 100).round(6)

    ticker_data['Ticker'] = ticker  # คงค่า Ticker
    
    data_list.append(ticker_data)

# ตรวจสอบว่ามีข้อมูลหรือไม่
if not data_list:
    print("❌ ไม่มีข้อมูลใด ๆ ที่ดึงมาได้")
    sys.exit(1)

# รวมข้อมูลทั้งหมดเป็น DataFrame เดียว
cleaned_data = pd.concat(data_list).reset_index().rename(columns={'index': 'Date'})

# ✅ กรองเฉพาะข้อมูลที่ไม่เป็น 0 (วันที่มีการซื้อขายจริง)
print("🔹 กรองข้อมูลที่เป็น 0 ออก...")
before_filter = len(cleaned_data)
cleaned_data = cleaned_data[
    (cleaned_data['Open'] != 0) & 
    (cleaned_data['High'] != 0) & 
    (cleaned_data['Low'] != 0) & 
    (cleaned_data['Close'] != 0) &
    (cleaned_data['Volume'] != 0)
]
after_filter = len(cleaned_data)
print(f"🔹 กรองข้อมูลแล้ว: {before_filter} -> {after_filter} แถว")

# ตั้งลำดับคอลัมน์ให้ถูกต้อง
cleaned_data = cleaned_data[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Changepercent']]

# ✅ เรียงลำดับตามวันที่และ Ticker
cleaned_data = cleaned_data.sort_values(['Date', 'Ticker']).reset_index(drop=True)

# บันทึกข้อมูลเป็นไฟล์ CSV
output_path = os.path.join(os.path.dirname(__file__), "Stock", "stock_data_usa.csv")
cleaned_data.to_csv(output_path, index=False)
print(f"✅ บันทึกข้อมูลลงไฟล์ CSV สำเร็จ: {output_path}")

# ✅ แสดงสถิติข้อมูล
print(f"🔹 จำนวนข้อมูลทั้งหมด: {len(cleaned_data)} แถว")
print(f"🔹 วันที่ที่มีข้อมูล: {cleaned_data['Date'].nunique()} วัน")
print(f"🔹 ช่วงวันที่: {cleaned_data['Date'].min()} ถึง {cleaned_data['Date'].max()}")

# แสดงตัวอย่างข้อมูลแต่ละหุ้น
for ticker in cleaned_data['Ticker'].unique():
    ticker_data = cleaned_data[cleaned_data['Ticker'] == ticker]
    print(f"🔹 {ticker}: {len(ticker_data)} แถว, วันที่ล่าสุด {ticker_data['Date'].max()}")

# แสดงตัวอย่างข้อมูล
print("\n📋 ตัวอย่างข้อมูล:")
print(cleaned_data.head(10))