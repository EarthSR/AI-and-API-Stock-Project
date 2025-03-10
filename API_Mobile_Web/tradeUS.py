import pymysql
import requests

# 📌 ตั้งค่าพอร์ตโบรกเกอร์ Alpaca (หุ้นอเมริกา)
ALPACA_API_KEY = "CKCC9EEJJO0B1WAHEOZN"
ALPACA_SECRET_KEY = "n7SJWh4eysMZKBoefJ5xhtnawtDH2nXoadcDweFS"
ALPACA_BASE_URL = "https://broker-api.sandbox.alpaca.markets"

HEADERS_ALPACA = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json"
}

# 📌 ตั้งค่าการเชื่อมต่อกับ MySQL
conn = pymysql.connect(
    host="10.10.50.62",
    user="trademine",
    password="trade789",
    database="TradeMine",
    port=3306
)
cursor = conn.cursor()

# 📌 ดึงข้อมูลพยากรณ์จากวันที่ล่าสุดของแต่ละหุ้น (เฉพาะหุ้นอเมริกา)
query = """
SELECT s.StockDetailID, s.StockSymbol, s.PredictionTrend, s.PredictionClose 
FROM StockDetail s
INNER JOIN (
    SELECT StockSymbol, MAX(Date) AS LatestDate 
    FROM StockDetail 
    GROUP BY StockSymbol
) latest ON s.StockSymbol = latest.StockSymbol AND s.Date = latest.LatestDate
JOIN Stock st ON s.StockSymbol = st.StockSymbol
WHERE st.Market = 'USA'
AND s.PredictionTrend IS NOT NULL 
AND s.PredictionClose IS NOT NULL 
LIMIT 10;
"""
cursor.execute(query)
stocks = cursor.fetchall()

# 📌 ฟังก์ชันดึงราคาตลาดหุ้นอเมริกาจาก Alpaca
def get_alpaca_price(stock_symbol):
    url = f"{ALPACA_BASE_URL}/v1/market_data/stocks/{stock_symbol}/quotes/latest"
    response = requests.get(url, headers=HEADERS_ALPACA)

    if response.status_code == 200:
        data = response.json()
        return data.get("quote", {}).get("ap", None)  # "ap" คือ Ask Price (ราคาขายล่าสุด)
    
    print(f"❌ ไม่สามารถดึงราคาหุ้น {stock_symbol}: {response.json()}")
    return None

# 📌 ฟังก์ชันส่งคำสั่งซื้อขายหุ้นอเมริกาผ่าน Alpaca API
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

# 📌 ดึง account_id จาก Alpaca (ใช้เฉพาะหุ้นอเมริกา)
account_id = get_alpaca_account_id()
if not account_id:
    print("❌ ไม่สามารถดึง account_id, ยกเลิกการเทรดหุ้นอเมริกา")
    conn.close()
    exit()

# 📌 เริ่ม Trade Logic
for stock in stocks:
    stock_id, stock_symbol, prediction_trend, prediction_close = stock

    # แปลงค่าจาก String เป็น Float
    prediction_trend = float(prediction_trend)
    prediction_close = float(prediction_close)

    # ดึงราคาตลาดปัจจุบัน
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
        trade_response = execute_alpaca_trade(account_id, stock_symbol, trade_decision, trade_quantity)
        print(f"📈 {trade_decision} {stock_symbol} @ {market_price} → {trade_response}")

conn.close()
print("✅ Auto Trade Logic (เฉพาะหุ้นอเมริกา) ดำเนินการเสร็จสิ้น!")
