import yfinance as yf
import pandas as pd

# กำหนดรายชื่อหุ้น
tickers = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL','META','TSLA', 'AVGO', 'TSM', 'AMD']

# กำหนดวันที่เริ่มต้นและวันที่สิ้นสุด
start_date = '2018-01-01'  # กำหนดวันที่เริ่มต้น
end_date = '2025-02-03'    # กำหนดวันที่สิ้นสุด

# ดึงข้อมูลราคาหุ้นจากวันที่เริ่มต้นถึงวันที่สิ้นสุด
data = yf.download(tickers, start=start_date, end=end_date)

# กำหนดชื่อคอลัมน์ที่ต้องการ
required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# สร้าง DataFrame สำหรับแต่ละหุ้น
data_list = []

for ticker in tickers:
    ticker_data = data.xs(ticker, axis=1, level=1)  # ดึงข้อมูลของหุ้นแต่ละตัว
    ticker_data['Ticker'] = ticker  # เพิ่มคอลัมน์ Ticker เพื่อระบุชื่อหุ้น
    data_list.append(ticker_data)

# รวมข้อมูลทั้งหมดเป็น DataFrame เดียว
cleaned_data = pd.concat(data_list, axis=0)

# ตั้งชื่อคอลัมน์ใหม่
cleaned_data = cleaned_data[['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]

# บันทึกข้อมูลเป็นไฟล์ CSV
cleaned_data.to_csv('stock_data_from_dates.csv')

print("ข้อมูลราคาหุ้น (Open, High, Low, Close, Volume) ถูกบันทึกลงในไฟล์ stock_data_from_dates.csv")
