import yfinance as yf
import pandas as pd
import datetime
import sys
import os

# ✅ ป้องกัน UnicodeEncodeError (ข้ามอีโมจิที่ไม่รองรับ)
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# ✅ ตรวจสอบระดับของโฟลเดอร์ (ปรับ `..` ตามตำแหน่งของไฟล์)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 

# กำหนดรายชื่อหุ้นไทย
tickers = ['ADVANC.BK', 'INTUCH.BK', 'TRUE.BK', 'DITTO.BK', 'DIF.BK', 
           'INSET.BK', 'JMART.BK', 'INET.BK', 'JAS.BK', 'HUMAN.BK']

# กำหนดวันที่เริ่มต้นและวันที่สิ้นสุด
start_date = '2018-01-01'
end_date = datetime.datetime.today().strftime('%Y-%m-%d')  # ได้วันที่ปัจจุบันในรูปแบบ YYYY-MM-DD

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
    ticker_data['Ticker'] = ticker.replace('.BK', '')  # ลบ .BK ออก

    # รีอินเด็กซ์ให้มีทุกวัน (รวมเสาร์-อาทิตย์)
    ticker_data.index = pd.to_datetime(ticker_data.index)  # แปลงเป็น datetime index
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')  # ใช้ end_date ที่อัปเดตอัตโนมัติ
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

    # เติมค่าที่ขาด
    ticker_data['Ticker'] = ticker.replace('.BK', '')  # คงค่า Ticker ในวันที่เพิ่ม
    
    data_list.append(ticker_data)

# รวมข้อมูลทั้งหมดเป็น DataFrame เดียว
cleaned_data = pd.concat(data_list).reset_index().rename(columns={'index': 'Date'})

# ตั้งลำดับคอลัมน์ให้ถูกต้อง
cleaned_data = cleaned_data[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]

# บันทึกข้อมูลเป็นไฟล์ CSV
cleaned_data.to_csv(os.path.join(BASE_DIR, "Finbert", "stock_data_thai.csv"), index=False)

# แสดงตัวอย่างข้อมูล
print(cleaned_data.head())
