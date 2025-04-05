import yfinance as yf
import pandas as pd
import datetime
import sys
import os
import mysql.connector
from dotenv import load_dotenv

# ✅ ป้องกัน UnicodeEncodeError (ข้ามอีโมจิที่ไม่รองรับ)
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# ✅ กำหนดรายชื่อหุ้นไทย
tickers = ['ADVANC.BK', 'INTUCH.BK', 'TRUE.BK', 'DITTO.BK', 'DIF.BK', 
           'INSET.BK', 'JMART.BK', 'INET.BK', 'JAS.BK', 'HUMAN.BK']

# ✅ ใช้วันที่เริ่มต้นเดียวกันทุกหุ้น
start_date = '2017-12-20'
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

# ✅ สร้างโฟลเดอร์ Stock ถ้ายังไม่มี
CURRENT_DIR = os.getcwd()
os.makedirs(os.path.join(CURRENT_DIR, "Stock"), exist_ok=True)

# ✅ ดึงข้อมูลจาก yfinance
try:
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    if data.empty:
        raise ValueError("❌ ไม่สามารถดึงข้อมูลจาก yfinance ได้")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# ✅ สร้าง DataFrame สำหรับแต่ละหุ้น
data_list = []

for ticker in tickers:
    ticker_data = data[ticker].copy()
    stock_name = ticker.replace('.BK', '')  # เอา .BK ออก
    ticker_data['Ticker'] = stock_name

    # ✅ รีอินเด็กซ์ให้มีทุกวัน (รวมเสาร์-อาทิตย์)
    ticker_data.index = pd.to_datetime(ticker_data.index)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    ticker_data = ticker_data.reindex(all_dates)

    # ✅ เติม NaN ด้วย rolling average
    if ticker_data[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().sum().sum() > 0:
        print(f"⚠️ พบค่า NaN ในข้อมูลของ {stock_name}, ใช้ค่าเฉลี่ยย้อนหลังเติมแทน")

    ticker_data[['Open', 'High', 'Low', 'Close', 'Volume']] = (
        ticker_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        .ffill()
        .rolling(window=3, min_periods=1).mean()
        .fillna(0)
    )

    ticker_data['Ticker'] = stock_name
    data_list.append(ticker_data)

# ✅ รวมข้อมูลทั้งหมด
cleaned_data = pd.concat(data_list).reset_index().rename(columns={'index': 'Date'})
cleaned_data = cleaned_data[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]

# ✅ บันทึกเป็น CSV
csv_path = os.path.join(CURRENT_DIR, "Stock", "stock_data_thai.csv")
cleaned_data.to_csv(csv_path, index=False)

# ✅ แสดงผล
print(f"✅ บันทึกไฟล์ CSV สำเร็จที่: {csv_path}")
print(cleaned_data.head())
