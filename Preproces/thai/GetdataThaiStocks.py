import yfinance as yf
import pandas as pd
import datetime
import sys
import os
import mysql.connector
from dotenv import load_dotenv
import io

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

# ✅ กำหนดรายชื่อหุ้นไทย
tickers = ['ADVANC.BK', 'INTUCH.BK', 'TRUE.BK', 'DITTO.BK', 'DIF.BK', 
           'INSET.BK', 'JMART.BK', 'INET.BK', 'JAS.BK', 'HUMAN.BK']

# ✅ ตรวจสอบวันที่ล่าสุดจากฐานข้อมูล
latest_dates = {}
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

for ticker in tickers:
    stock_name = ticker.replace('.BK', '')  # เอา .BK ออกก่อนเช็คในฐานข้อมูล
    cursor.execute("SELECT MAX(Date) FROM StockDetail WHERE StockSymbol = %s", (stock_name,))
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

# ดึงข้อมูลราคาหุ้นจากวันที่เริ่มต้นถึงวันที่สิ้นสุด
try:
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    if data.empty:
        raise ValueError("❌ ไม่สามารถดึงข้อมูลจาก yfinance ได้")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)  # ❌ หยุดการทำงานทันที

# สร้าง DataFrame สำหรับแต่ละหุ้น
data_list = []

for ticker in tickers:
    # ดึงข้อมูลราคาหุ้น และใช้ .copy() เพื่อป้องกัน SettingWithCopyWarning
    ticker_data = data[ticker].copy()
    stock_name = ticker.replace('.BK', '')  # เอา .BK ออกก่อนบันทึกลง DataFrame
    ticker_data['Ticker'] = stock_name

    # รีอินเด็กซ์ให้มีทุกวัน (รวมเสาร์-อาทิตย์)
    ticker_data.index = pd.to_datetime(ticker_data.index)  # แปลงเป็น datetime index
    all_dates = pd.date_range(start=latest_dates[ticker], end=end_date, freq='D')  # ใช้ start_date ที่อัปเดตจากฐานข้อมูล
    ticker_data = ticker_data.reindex(all_dates)
    ticker_data['Changepercen'] = (ticker_data['Close'] - ticker_data['Open']) / ticker_data['Open'] * 100
    # 🔹 ใช้ค่า **วันก่อนหน้า** แทน NaN ก่อนเติม 0
    if ticker_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Changepercen']].isnull().sum().sum() > 0:
        print(f"⚠️ พบค่า NaN ในข้อมูลของ {stock_name}, ใช้ค่าเฉลี่ยย้อนหลังเติมแทน")

    ticker_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Changepercen']] = (
        ticker_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Changepercen']]
        .ffill()
        .rolling(window=3, min_periods=1).mean()
        .fillna(0)  # ✅ ถ้า ffill() ยังมีค่า NaN ให้เติม 0
    )

    # เติมค่าที่ขาด
    ticker_data['Ticker'] = stock_name  # คงค่า Ticker ในวันที่เพิ่ม
    
    data_list.append(ticker_data)

# รวมข้อมูลทั้งหมดเป็น DataFrame เดียว
cleaned_data = pd.concat(data_list).reset_index().rename(columns={'index': 'Date'})

# ตั้งลำดับคอลัมน์ให้ถูกต้อง
cleaned_data = cleaned_data[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume' , 
                             'Changepercen']]

# บันทึกข้อมูลเป็นไฟล์ CSV
csv_path = os.path.join(CURRENT_DIR,"thai", "Stock", "stock_data_thai.csv")
cleaned_data.to_csv(csv_path, index=False)

# แสดงตัวอย่างข้อมูล
print(f"✅ บันทึกไฟล์ CSV สำเร็จที่: {csv_path}")
print(cleaned_data.head())
