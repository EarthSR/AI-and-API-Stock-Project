import pymysql
import requests
from settrade_v2 import Investor

# 📌 ตั้งค่าพอร์ตโบรกเกอร์ Alpaca (หุ้นต่างประเทศ)
ALPACA_API_KEY = "CK7TZZC4V4NIBB43FPKD"
ALPACA_SECRET_KEY = "CqHdNXFXJllrU6Rq9aIgCUERNSWdw2tq1cwagSwe"
ALPACA_BASE_URL = "https://broker-api.sandbox.alpaca.markets"

HEADERS_ALPACA = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json"
}

# 📌 ตั้งค่า API สำหรับตลาดหุ้นไทย (Settrade v2)
SETTRADE_APP_ID = "9hULPQzhsHUwnWtA"
SETTRADE_APP_SECRET = "FYbYY4VaqUh1V0rj97sCKOJkoSNqqK8TZhaS6lT9WpE="
SETTRADE_BROKER_ID = "SANDBOX"  # ✅ เปลี่ยนจาก "SANDBOX" เป็นโบรกเกอร์จริง เช่น "SIT", "KSEC", "BLS"
SETTRADE_APP_CODE = "SANDBOX"
SETTRADE_ACC_NO = "earthsr-E"

# 📌 เชื่อมต่อกับ Settrade API
try:
    investor = Investor(
        app_id=SETTRADE_APP_ID,
        app_secret=SETTRADE_APP_SECRET,
        broker_id=SETTRADE_BROKER_ID,
        app_code=SETTRADE_APP_CODE,
        is_auto_queue=False
    )
    thai_account = investor.Equity(account_no=SETTRADE_ACC_NO)
    
    # ✅ ตรวจสอบข้อมูลบัญชี Settrade API ก่อนทำงาน
    account_info = thai_account.get_account_info()
    print("✅ ข้อมูลบัญชี Settrade API:", account_info)

except Exception as e:
    print("❌ ไม่สามารถเชื่อมต่อ Settrade API:", str(e))
    exit()  # ❌ ถ้าบัญชีมีปัญหา ให้หยุดโปรแกรม

# 📌 ตั้งค่าการเชื่อมต่อกับ MySQL
conn = pymysql.connect(
    host="10.10.50.62",
    user="trademine",
    password="trade789",
    database="TradeMine",
    port=3306
)
cursor = conn.cursor()

# 📌 ดึงข้อมูลพยากรณ์จากวันที่ล่าสุดของแต่ละหุ้น พร้อมตรวจสอบตลาด (`Market`)
query = """
SELECT s.StockDetailID, s.StockSymbol, COALESCE(st.Market, 'UNKNOWN') AS Market, 
       s.PredictionTrend, s.PredictionClose 
FROM StockDetail s
JOIN Stock st ON s.StockSymbol = st.StockSymbol  -- เชื่อมกับ Stock เพื่อดึง Market
INNER JOIN (
    SELECT StockSymbol, MAX(Date) AS LatestDate 
    FROM StockDetail 
    GROUP BY StockSymbol
) latest ON s.StockSymbol = latest.StockSymbol AND s.Date = latest.LatestDate
WHERE s.PredictionTrend IS NOT NULL AND s.PredictionClose IS NOT NULL 
LIMIT 10;
"""
cursor.execute(query)
stocks = cursor.fetchall()

# 📌 ฟังก์ชันดึงราคาตลาดหุ้นไทยจาก Settrade
def get_settrade_price(stock_symbol):
    stock_symbol_api = f"{stock_symbol}.BK"
    try:
        quote = thai_account.get_quote(stock_symbol_api)
        return quote["last_price"]
    except Exception as e:
        print(f"❌ ไม่สามารถดึงราคาหุ้นไทย {stock_symbol}: {str(e)}")
        return None

# 📌 ฟังก์ชันดึงราคาตลาดหุ้นต่างประเทศจาก Alpaca
def get_alpaca_price(stock_symbol):
    url = f"{ALPACA_BASE_URL}/v1/market_data/stocks/{stock_symbol}/quotes/latest"
    response = requests.get(url, headers=HEADERS_ALPACA)

    if response.status_code == 200:
        data = response.json()
        return data.get("quote", {}).get("ap", None)  # "ap" คือ Ask Price (ราคาขายล่าสุด)
    
    print(f"❌ ไม่สามารถดึงราคาหุ้น {stock_symbol}: {response.json()}")
    return None

# 📌 ฟังก์ชันส่งคำสั่งซื้อขายหุ้นไทยผ่าน Settrade
def execute_settrade_trade(stock_symbol, trade_type, quantity):
    try:
        order_result = thai_account.place_order(
            symbol=f"{stock_symbol}.BK",
            side=trade_type.lower(),
            quantity=quantity,
            price="MP",  # ✅ ใช้ราคาตลาด (Market Price) แทน 0
            validity="GTC"
        )
        return order_result
    except Exception as e:
        print(f"❌ คำสั่ง {trade_type} หุ้นไทย {stock_symbol} ล้มเหลว: {str(e)}")
        return None

# 📌 ฟังก์ชันส่งคำสั่งซื้อขายหุ้นต่างประเทศผ่าน Alpaca API
def execute_alpaca_trade(account_id, stock_symbol, trade_type, quantity):
    trade_payload = {
        "symbol": stock_symbol,
        "qty": quantity,
        "side": trade_type.lower(),
        "type": "market",
        "time_in_force": "gtc"
    }
    url = f"{ALPACA_BASE_URL}/v1/trading/accounts/{account_id}/orders"
    response = requests.post(url, json=trade_payload, headers=HEADERS_ALPACA)

    if response.status_code == 200:
        return response.json()
    
    print(f"❌ คำสั่ง {trade_type} {stock_symbol} ล้มเหลว: {response.json()}")
    return None

# 📌 ดึง account_id จาก Alpaca
def get_alpaca_account_id():
    url = f"{ALPACA_BASE_URL}/v1/accounts"
    response = requests.get(url, headers=HEADERS_ALPACA)

    if response.status_code == 200:
        accounts = response.json()
        if accounts:
            return accounts[0]["id"]  # ใช้บัญชีแรก
    
    print(f"❌ ไม่สามารถดึง account_id: {response.json()}")
    return None

# 📌 ดึง account_id จาก Alpaca (ใช้เฉพาะหุ้นต่างประเทศ)
account_id = get_alpaca_account_id()
if not account_id:
    print("❌ ไม่สามารถดึง account_id, ยกเลิกการเทรดหุ้นต่างประเทศ")

# 📌 เริ่ม Trade Logic
for stock in stocks:
    stock_id, stock_symbol, market, prediction_trend, prediction_close = stock

    # ตรวจสอบว่า Market เป็น "Thailand" หรือไม่
    is_thai_stock = market.lower() == "thailand"

    # แปลงค่าจาก String เป็น Float
    prediction_trend = float(prediction_trend)
    prediction_close = float(prediction_close)

    # ดึงราคาตลาดปัจจุบัน
    market_price = None
    if is_thai_stock:
        market_price = get_settrade_price(stock_symbol)
    else:
        market_price = get_alpaca_price(stock_symbol)

    if market_price is None:
        print(f"❌ ไม่พบราคาตลาดของ {stock_symbol}")
        continue

    # 📌 ตัดสินใจซื้อขาย
    trade_decision = "HOLD"
    trade_quantity = 10

    if prediction_trend > 0 and prediction_close > market_price:
        trade_decision = "BUY"
    elif prediction_trend < 0 and prediction_close < market_price:
        trade_decision = "SELL"

    # 📌 ส่งคำสั่งซื้อขาย
    if trade_decision in ["BUY", "SELL"]:
        if is_thai_stock:
            trade_response = execute_settrade_trade(stock_symbol, trade_decision, trade_quantity)
        else:
            trade_response = execute_alpaca_trade(account_id, stock_symbol, trade_decision, trade_quantity)

        print(f"📈 {trade_decision} {stock_symbol} ({market}) @ {market_price} → {trade_response}")

conn.close()
print("✅ Auto Trade Logic ดำเนินการเสร็จสิ้น!")
