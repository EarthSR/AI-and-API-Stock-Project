import yfinance as yf
import pandas as pd

# กำหนดรายชื่อหุ้นอเมริกา
tickers = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL','META','TSLA', 'AVGO', 'TSM', 'AMD']

# กำหนดวันที่เริ่มต้นและวันที่สิ้นสุด
start_date = '2018-01-01'  
end_date = '2025-02-28'    

# ดึงข้อมูลราคาหุ้นจากวันที่เริ่มต้นถึงวันที่สิ้นสุด
data = yf.download(tickers, start=start_date, end=end_date)

# สร้าง DataFrame สำหรับแต่ละหุ้น
data_list = []

for ticker in tickers:
    # ดึงข้อมูลราคาหุ้น และใช้ .copy() เพื่อป้องกัน SettingWithCopyWarning
    ticker_data = data.xs(ticker, axis=1, level=1).copy()  
    ticker_data['Ticker'] = ticker.replace('.BK', '')  # ลบ .BK ออก  
    
    # ดึงข้อมูล Market Cap
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
        market_cap = info.get('marketCap', 'N/A')  
    except Exception:
        market_cap = 'N/A'  
    
    ticker_data['Market Cap'] = market_cap  

    # รีอินเด็กซ์เพื่อให้มีทุกวัน (รวมเสาร์-อาทิตย์)
    ticker_data.index = pd.to_datetime(ticker_data.index)  # แปลงเป็น datetime index
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')  # ทุกวัน
    ticker_data = ticker_data.reindex(all_dates)  

    # เปลี่ยนค่าในวันเสาร์-อาทิตย์เป็น 0
    ticker_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']] = ticker_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']].fillna(0)

    data_list.append(ticker_data)

# รวมข้อมูลทั้งหมดเป็น DataFrame เดียว
cleaned_data = pd.concat(data_list).reset_index().rename(columns={'index': 'Date'})

# ตั้งชื่อคอลัมน์ใหม่
cleaned_data = cleaned_data[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']]

# บันทึกข้อมูลเป็นไฟล์ CSV
cleaned_data.to_csv('stock_data_with_marketcap_usa.csv', index=False)
