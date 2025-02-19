import yfinance as yf
import pandas as pd

# กำหนดรายชื่อหุ้นไทย
tickers = ['ADVANC.BK', 'INTUCH.BK', 'TRUE.BK', 'DITTO.BK', 'DIF.BK', 'INSET.BK', 'JMART.BK', 'INET.BK', 'JAS.BK', 'HUMAN.BK']
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
    # ดึงข้อมูลราคาหุ้น
    ticker_data = data.xs(ticker, axis=1, level=1)  # ดึงข้อมูลของหุ้นแต่ละตัว
    ticker_data['Ticker'] = ticker  # เพิ่มคอลัมน์ Ticker เพื่อระบุชื่อหุ้น
    
    # ดึงข้อมูล market cap
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
        market_cap = info.get('marketCap', 'N/A')  # ถ้าไม่ได้รับข้อมูลจะเป็น 'N/A'
    except Exception as e:
        market_cap = 'N/A'  # ถ้าเกิดข้อผิดพลาดให้แสดงเป็น 'N/A'
    
    ticker_data['Market Cap'] = market_cap  # เพิ่มข้อมูล Market Cap ใน DataFrame

    data_list.append(ticker_data)

# รวมข้อมูลทั้งหมดเป็น DataFrame เดียว
cleaned_data = pd.concat(data_list, axis=0)

# ตั้งชื่อคอลัมน์ใหม่
cleaned_data = cleaned_data[['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']]

# บันทึกข้อมูลเป็นไฟล์ CSV
cleaned_data.to_csv('stock_data_with_marketcap_thai.csv')

print("ข้อมูลราคาหุ้น (Open, High, Low, Close, Volume, Market Cap) ของหุ้นไทยถูกบันทึกลงในไฟล์ stock_data_with_marketcap_thai.csv")
