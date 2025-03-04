import yfinance as yf
import pandas as pd

# กำหนดรายชื่อหุ้นอเมริกา
tickers = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'TSM', 'AMD']

# กำหนดวันที่เริ่มต้นและสิ้นสุด
start_date = '2018-01-01'
end_date = '2025-02-28'

# ดึงข้อมูลราคาหุ้นจาก yfinance
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

# สร้าง DataFrame สำหรับแต่ละหุ้น
data_list = []

for ticker in tickers:
    # ดึงข้อมูลราคาหุ้น และใช้ .copy() ป้องกัน SettingWithCopyWarning
    ticker_data = data[ticker].copy()
    ticker_data['Ticker'] = ticker  # กำหนดค่า Ticker
    
    # ดึงข้อมูล Market Cap
    stock = yf.Ticker(ticker)
    try:
        market_cap = stock.info.get('marketCap', None)
        if market_cap is None:
            market_cap = 'N/A'
    except Exception:
        market_cap = 'N/A'
    
    ticker_data['Market Cap'] = market_cap

    # รีอินเด็กซ์ให้มีทุกวัน (รวมเสาร์-อาทิตย์)
    ticker_data.index = pd.to_datetime(ticker_data.index)  # แปลงเป็น datetime index
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')  # ทุกวัน
    ticker_data = ticker_data.reindex(all_dates)

    # 🔹 ใช้ค่า **วันก่อนหน้า** แทน NaN ก่อนเติม 0
    ticker_data[['Open', 'High', 'Low', 'Close', 'Volume']] = (
        ticker_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        .fillna(method='ffill')
        .rolling(window=3, min_periods=1).mean()  # ใช้ค่าเฉลี่ย 3 วันก่อนหน้า
    )


    ticker_data['Ticker'] = ticker  # คงค่า Ticker
    ticker_data['Market Cap'] = market_cap  # คงค่า Market Cap
    
    data_list.append(ticker_data)

# รวมข้อมูลทั้งหมดเป็น DataFrame เดียว
cleaned_data = pd.concat(data_list).reset_index().rename(columns={'index': 'Date'})

# ตั้งลำดับคอลัมน์ให้ถูกต้อง
cleaned_data = cleaned_data[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']]

# บันทึกข้อมูลเป็นไฟล์ CSV
cleaned_data.to_csv('stock_data_with_marketcap_usa.csv', index=False)

# แสดงตัวอย่างข้อมูล
print(cleaned_data.head())
