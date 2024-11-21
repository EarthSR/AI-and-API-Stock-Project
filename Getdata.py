import os
import yfinance as yf
import pandas as pd

def fetch_nasdaq_tickers():
    # ดึงข้อมูลหุ้นทั้งหมดในตลาด NASDAQ
    url = "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed-symbols.csv"
    nasdaq_data = pd.read_csv(url)

    # ดึงเฉพาะชื่อหุ้น (Ticker)
    tickers = nasdaq_data['Symbol'].tolist()
    return tickers

def fetch_historical_data(tickers):
    # สร้างโฟลเดอร์ NASDAQ_Historical ถ้ายังไม่มี
    folder_name = "NASDAQ_Historical"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for ticker in tickers[:100]:  # ตัวอย่างดึงข้อมูล 100 หุ้นแรก (ปรับจำนวนตามต้องการ)
        try:
            print(f"Fetching data for {ticker}...")
            stock = yf.Ticker(ticker)

            # ดึงข้อมูลประวัติย้อนหลังทั้งหมด
            hist = stock.history(period="max")  # period="max" ดึงข้อมูลตั้งแต่เริ่มต้น

            # ตรวจสอบว่ามีข้อมูลหรือไม่
            if not hist.empty:
                # บันทึกข้อมูลเป็นไฟล์ CSV ในโฟลเดอร์ NASDAQ_Historical
                filename = os.path.join(folder_name, f"{ticker}_historical_data.csv")
                hist.to_csv(filename)
                print(f"Data for {ticker} saved to {filename}")
            else:
                print(f"No data found for {ticker}")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

# ดึงข้อมูล Ticker
tickers = fetch_nasdaq_tickers()

# ดึงข้อมูลประวัติย้อนหลัง
fetch_historical_data(tickers)
