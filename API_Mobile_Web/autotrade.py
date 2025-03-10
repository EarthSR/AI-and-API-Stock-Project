import pymysql
import requests
import json
from settrade_v2 import Investor

# 📌 ตั้งค่า API สำหรับตลาดหุ้นไทย (Settrade v2)
SETTRADE_APP_ID = "2DaswIgiipmhvNAf"
SETTRADE_APP_SECRET = "PFjxcryxZol0qmuCCGslg60D/H1djOOrooqw0vmgLMs="
SETTRADE_BROKER_ID = "SANDBOX"  # ✅ เปลี่ยนเป็นโบรกเกอร์จริง เช่น "SIT", "KSEC", "BLS"
SETTRADE_APP_CODE = "SANDBOX"
SETTRADE_ACC_NO = "earthsr-E"

# 📌 เชื่อมต่อกับ Settrade API
try:
    investor = Investor(
        app_id=SETTRADE_APP_ID,
        app_secret=SETTRADE_APP_SECRET,
        broker_id=SETTRADE_BROKER_ID,
        app_code=SETTRADE_APP_CODE,
        is_auto_queue=True  # ✅ เปิดใช้งาน Auto Queue
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

# 📌 ดึงข้อมูลพยากรณ์จากวันที่ล่าสุดของแต่ละหุ้น
query = """
SELECT s.StockDetailID, s.StockSymbol, COALESCE(st.Market, 'UNKNOWN') AS Market, 
       s.PredictionTrend, s.PredictionClose 
FROM StockDetail s
JOIN Stock st ON s.StockSymbol = st.StockSymbol
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

def get_settrade_price(stock_symbol):
    try:
        market_data = investor.MarketData()
        print(f"📌 ตรวจสอบ API: {market_data}")

        # ✅ ดึงข้อมูลจาก API โดยไม่ต้องเติม ".BK"
        quote = market_data.get_quote_symbol(stock_symbol)

        # ✅ Debug Data
        print(f"📌 Debug Data สำหรับ {stock_symbol}: {repr(quote)}")

        # ✅ เช็คว่าข้อมูลมีอยู่หรือไม่
        if isinstance(quote, dict):
            if quote.get("marketStatus") == "Close":
                print(f"⚠️ ตลาดปิดอยู่ ไม่สามารถดึงราคาสำหรับ {stock_symbol}")
                return None
            if "last" in quote and quote["last"] is not None:
                return float(quote["last"])
            else:
                print(f"⚠️ ไม่พบราคาล่าสุดของ {stock_symbol}, อาจไม่มีการซื้อขาย")
                return None
        else:
            print(f"❌ API คืนค่าที่ไม่ใช่ JSON: {repr(quote)}")
            return None

    except Exception as e:
        print(f"❌ ไม่สามารถดึงราคาหุ้นไทย {stock_symbol}: {str(e)}")
        return None

# 📌 ฟังก์ชันส่งคำสั่งซื้อขายหุ้นไทยผ่าน Settrade
def execute_settrade_trade(stock_symbol, trade_type, quantity):
    try:
        order_result = thai_account.place_order(
            symbol=f"{stock_symbol}.BK",
            side=trade_type.lower(),
            quantity=quantity,
            price="MP",  # ✅ ใช้ราคาตลาด (Market Price)
            validity="GTC"
        )
        return order_result
    except Exception as e:
        print(f"❌ คำสั่ง {trade_type} หุ้นไทย {stock_symbol} ล้มเหลว: {str(e)}")
        return None

# 📌 ฟังก์ชันบันทึกคำสั่งซื้อขายลง `TradeHistory`
def log_trade(stock_symbol, trade_type, quantity, predicted_price, actual_price, status):
    total_trade_value = quantity * actual_price if actual_price else None
    profit_loss = (actual_price - predicted_price) * quantity if actual_price else None
    trade_fee = total_trade_value * 0.001 if total_trade_value else None  # ✅ ค่าธรรมเนียม 0.1%
    vat = trade_fee * 0.07 if trade_fee else None  # ✅ ภาษี 7% ของค่าธรรมเนียม

    query = """
    INSERT INTO TradeHistory (TradeType, Quantity, Price, ActualPrice, 
                              TotalTradeValue, ProfitLoss, TradeFee, VAT, TradeStatus)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
    """
    cursor.execute(query, (trade_type, quantity, predicted_price, actual_price, 
                           total_trade_value, profit_loss, trade_fee, vat, status))
    conn.commit()
    print(f"✅ บันทึกคำสั่ง {trade_type} {stock_symbol} จำนวน {quantity} หุ้น @ {actual_price} สำเร็จ")

# 📌 เริ่ม Trade Logic
STOP_LOSS_THRESHOLD = 0.98  # 2% loss
TAKE_PROFIT_THRESHOLD = 1.05  # 5% gain

for stock in stocks:
    stock_id, stock_symbol, market, prediction_trend, prediction_close = stock

    # ตรวจสอบว่าเป็นหุ้นไทย
    if market.lower() != "thailand":
        continue

    # ดึงราคาตลาดปัจจุบัน
    market_price = get_settrade_price(stock_symbol)

    if market_price is None:
        continue

    # ตัดสินใจซื้อขาย
    trade_decision = "HOLD"
    trade_quantity = 10

    if prediction_trend > 0 and prediction_close > market_price:
        if market_price <= prediction_close * STOP_LOSS_THRESHOLD:
            trade_decision = "SELL"  # Stop Loss
        else:
            trade_decision = "BUY"
    elif prediction_trend < 0 and prediction_close < market_price:
        if market_price >= prediction_close * TAKE_PROFIT_THRESHOLD:
            trade_decision = "BUY"  # Take Profit
        else:
            trade_decision = "SELL"

    # 📌 ส่งคำสั่งซื้อขาย
    if trade_decision in ["BUY", "SELL"]:
        trade_response = execute_settrade_trade(stock_symbol, trade_decision, trade_quantity)

        # ✅ บันทึกข้อมูลคำสั่งซื้อขายลง `TradeHistory`
        if trade_response:
            log_trade(stock_symbol, trade_decision, trade_quantity, prediction_close, market_price, "SUCCESS")
        else:
            log_trade(stock_symbol, trade_decision, trade_quantity, prediction_close, None, "FAILED")

# ปิดการเชื่อมต่อ Database
conn.close()
print("✅ Auto Trade Logic สำหรับตลาดหุ้นไทยดำเนินการเสร็จสิ้น!")
