import yfinance as yf
import pandas as pd
import datetime
import sys
import os
import mysql.connector
from dotenv import load_dotenv

# ✅ ป้องกัน UnicodeEncodeError (ข้ามอีโมจิที่ไม่รองรับ)
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# ✅ กำหนดรายชื่อหุ้นอเมริกา (Top 10)
tickers = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'TSM', 'AMD']

# ✅ กำหนดวันเริ่มต้นเดียวกันสำหรับทุกหุ้น
start_date = '2017-12-20'
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

# ✅ เตรียมโฟลเดอร์สำหรับบันทึกข้อมูล
CURRENT_DIR = os.getcwd()
os.makedirs(os.path.join(CURRENT_DIR, "Stock"), exist_ok=True)

# ✅ ดาวน์โหลดข้อมูลจาก yfinance พร้อม Retry
max_retries = 3
retry_count = 0

while retry_count < max_retries:
    try:
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        if data.empty:
            raise ValueError("❌ ไม่สามารถดึงข้อมูลจาก yfinance ได้")
        break
    except Exception as e:
        retry_count += 1
        print(f"⚠️ Error: {e} (ลองใหม่ {retry_count}/{max_retries})")
        if retry_count == max_retries:
            sys.exit(1)

# ✅ รวมข้อมูลหุ้นแต่ละตัว
data_list = []

for ticker in tickers:
    ticker_data = data[ticker].copy()
    ticker_data['Ticker'] = ticker

    # ✅ สร้างช่วงวันที่ทุกวัน (รวมเสาร์-อาทิตย์)
    ticker_data.index = pd.to_datetime(ticker_data.index)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    ticker_data = ticker_data.reindex(all_dates)

    # ✅ เติม NaN ด้วย rolling mean จากค่าเดิม
    if ticker_data[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().sum().sum() > 0:
        print(f"⚠️ พบค่า NaN ในข้อมูลของ {ticker}, ใช้ค่าเฉลี่ยย้อนหลังเติมแทน")

    ticker_data[['Open', 'High', 'Low', 'Close', 'Volume']] = (
        ticker_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        .ffill()
        .rolling(window=3, min_periods=1).mean()
        .fillna(0)
    )

    ticker_data['Ticker'] = ticker
    data_list.append(ticker_data)

# ✅ รวมเป็น DataFrame เดียว
cleaned_data = pd.concat(data_list).reset_index().rename(columns={'index': 'Date'})
cleaned_data = cleaned_data[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]

# ✅ บันทึกเป็น CSV
output_path = os.path.join(CURRENT_DIR, "Stock", "stock_data_usa.csv")
cleaned_data.to_csv(output_path, index=False)

# ✅ แสดงตัวอย่างข้อมูล
print(cleaned_data.head())
