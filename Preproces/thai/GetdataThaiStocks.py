import yfinance as yf
import pandas as pd
import datetime
import sys
import os
import mysql.connector
from dotenv import load_dotenv
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ✅ ตรวจสอบระดับของโฟลเดอร์
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

# ✅ กำหนดรายชื่อหุ้นไทย
tickers = ['ADVANC.BK', 'TRUE.BK', 'DITTO.BK', 'DIF.BK', 
           'INSET.BK', 'JMART.BK', 'INET.BK', 'JAS.BK', 'HUMAN.BK']

# ✅ ตรวจสอบวันที่ล่าสุดจากฐานข้อมูล
latest_dates = {}
today = datetime.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
end_date = (today - datetime.timedelta(days=1)).date()  # ใช้เมื่อวานเป็นวันที่สิ้นสุด

for ticker in tickers:
    stock_name = ticker.replace('.BK', '')  # เอา .BK ออกก่อนเช็คในฐานข้อมูล
    cursor.execute("SELECT MAX(Date) FROM StockDetail WHERE StockSymbol = %s", (stock_name,))
    result = cursor.fetchone()[0]
    if result is None:
        latest_dates[ticker] = "2018-01-01"  # Default start date if no data
    else:
        next_day = result + datetime.timedelta(days=1)  # วันที่ถัดไป
        # ใช้ค่าวันที่ที่เร็วกว่าระหว่าง next_day และ end_date
        latest_dates[ticker] = min(next_day, end_date).strftime('%Y-%m-%d')

# ✅ ปิดการเชื่อมต่อฐานข้อมูล (ชั่วคราว)
cursor.close()
conn.close()
print("🔹 ปิดการเชื่อมต่อฐานข้อมูลแล้ว")

# ✅ ตรวจสอบวันที่เริ่มต้นสำหรับแต่ละหุ้น
valid_tickers = []
for ticker in tickers:
    if latest_dates[ticker] < end_date.strftime('%Y-%m-%d'):
        valid_tickers.append(ticker)
    else:
        print(f"⚠️ ข้าม {ticker}: ไม่มีวันที่ใหม่ให้ดึงข้อมูล (start_date {latest_dates[ticker]} >= end_date {end_date})")

if not valid_tickers:
    print("❌ ไม่มีหุ้นใดที่มีวันที่ใหม่ให้ดึงข้อมูล")
    sys.exit(0)

# ดึงข้อมูลราคาหุ้นจาก yfinance
max_retries = 3
data_list = []

for ticker in valid_tickers:
    retry_count = 0
    while retry_count < max_retries:
        try:
            # ดึงข้อมูลสำหรับแต่ละหุ้นแยกกัน
            data = yf.download(ticker, start=latest_dates[ticker], end=end_date.strftime('%Y-%m-%d'), interval='1d')
            if data.empty:
                print(f"⚠️ ไม่มีข้อมูลสำหรับ {ticker} ในช่วงวันที่ {latest_dates[ticker]} ถึง {end_date}")
                break
            data['Ticker'] = ticker.replace('.BK', '')  # เอา .BK ออกก่อนบันทึก
            data_list.append(data)
            break
        except Exception as e:
            retry_count += 1
            print(f"⚠️ Error สำหรับ {ticker}: {e} (ลองใหม่ {retry_count}/{max_retries})")
            if retry_count == max_retries:
                print(f"❌ ล้มเหลวในการดึงข้อมูลสำหรับ {ticker}")
                break
            time.sleep(1)  # รอเพื่อป้องกัน rate limit

# รวมข้อมูลทั้งหมด
if not data_list:
    print("❌ ไม่มีข้อมูลใด ๆ ที่ดึงมาได้")
    sys.exit(1)

cleaned_data = pd.concat(data_list).reset_index()

# รีอินเด็กซ์และจัดการ NaN
data_list = []  # รีเซ็ท data_list สำหรับข้อมูลที่รีอินเด็กซ์
for ticker in valid_tickers:
    stock_name = ticker.replace('.BK', '')
    ticker_data = cleaned_data[cleaned_data['Ticker'] == stock_name].copy()
    if ticker_data.empty:
        continue
    ticker_data.index = pd.to_datetime(ticker_data['Date'])
    all_dates = pd.date_range(start=latest_dates[ticker], end=end_date, freq='D')
    ticker_data = ticker_data.reindex(all_dates, method='ffill')  # Forward fill เพื่อหลีกเลี่ยงการเติมศูนย์
    ticker_data['Changepercen'] = (ticker_data['Close'] - ticker_data['Open']) / ticker_data['Open'] * 100
    ticker_data['Ticker'] = stock_name

    # เติม NaN ด้วยค่าเฉลี่ยย้อนหลัง (ถ้ายังมี NaN หลัง ffill)
    if ticker_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Changepercen']].isnull().any().any():
        print(f"⚠️ พบค่า NaN ในข้อมูลของ {stock_name}, ใช้ค่าเฉลี่ยย้อนหลังเติม")
        ticker_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Changepercen']] = (
            ticker_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Changepercen']]
            .rolling(window=3, min_periods=1).mean()
            .fillna(0)  # เติม 0 เฉพาะเมื่อจำเป็น
        )
    data_list.append(ticker_data.reset_index().rename(columns={'index': 'Date'}))

# รวมข้อมูลทั้งหมดเป็น DataFrame เดียว
cleaned_data = pd.concat(data_list).reset_index(drop=True)

# ตั้งลำดับคอลัมน์ให้ถูกต้อง
cleaned_data = cleaned_data[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Changepercen']]

# บันทึกข้อมูลเป็นไฟล์ CSV
output_path = os.path.join(CURRENT_DIR, "thai", "Stock", "stock_data_thai.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # สร้างโฟลเดอร์ถ้ายังไม่มี
cleaned_data.to_csv(output_path, index=False)
print(f"✅ บันทึกข้อมูลลงไฟล์ CSV สำเร็จ: {output_path}")

# แสดงตัวอย่างข้อมูล
print(cleaned_data.head())