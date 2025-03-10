import requests

# 🔹 ตั้งค่า API Key สำหรับ Sandbox
ALPACA_API_KEY = "CK4IA1P74HGAOCJ7S6IT"
ALPACA_SECRET_KEY = "bRoiiplob9Toute5pia1pL6pd8aPbSCvm7Qq1uBj"
ALPACA_BASE_URL = "https://broker-api.sandbox.alpaca.markets"

HEADERS_ALPACA = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json"
}

# 📌 ✅ 1. ตรวจสอบ API Key ว่าใช้งานได้หรือไม่
def test_sandbox_api():
    url = f"{ALPACA_BASE_URL}/v1/accounts"
    response = requests.get(url, headers=HEADERS_ALPACA)
    
    if response.status_code == 200:
        print("✅ API Key ใช้งานได้:", response.json())
    else:
        print(f"❌ ไม่สามารถดึงข้อมูลบัญชี: {response.status_code}, {response.json()}")

# 📌 ✅ 2. สร้างบัญชีลูกค้าใหม่ใน Sandbox
def create_sandbox_account():
    url = f"{ALPACA_BASE_URL}/v1/accounts"
    
    account_data = {
        "contact": {
            "email_address": "testuser@example.com",
            "phone_number": "+1234567890",
            "street_address": ["123 Test Street"],
            "city": "Testville",
            "state": "CA",
            "postal_code": "12345",
            "country": "USA"
        },
        "identity": {
            "given_name": "John",
            "family_name": "Doe",
            "date_of_birth": "1990-01-01",
            "tax_id": "123-45-6789",
            "tax_id_type": "USA_SSN",
            "country_of_citizenship": "USA",
            "country_of_birth": "USA",
            "country_of_tax_residence": "USA",
            "funding_source": ["employment_income"]
        },
        "disclosures": {
            "is_control_person": False,
            "is_affiliated_exchange_or_finra": False,
            "is_politically_exposed": False,
            "immediate_family_exposed": False
        },
        "agreements": [
            {"agreement": "customer_agreement", "signed_at": "2025-03-10T00:00:00Z", "ip_address": "127.0.0.1"},
            {"agreement": "margin_agreement", "signed_at": "2025-03-10T00:00:00Z", "ip_address": "127.0.0.1"},
            {"agreement": "option_agreement", "signed_at": "2025-03-10T00:00:00Z", "ip_address": "127.0.0.1"}
        ]
    }

    response = requests.post(url, json=account_data, headers=HEADERS_ALPACA)
    
    if response.status_code == 201:
        print("✅ บัญชี Sandbox ถูกสร้างเรียบร้อย:", response.json())
    else:
        print(f"❌ ไม่สามารถสร้างบัญชี Sandbox: {response.status_code}, {response.json()}")

# 📌 ✅ 3. ดึงรายการบัญชีทั้งหมดที่สร้างขึ้นใน Sandbox
def get_all_accounts():
    url = f"{ALPACA_BASE_URL}/v1/accounts"
    response = requests.get(url, headers=HEADERS_ALPACA)

    if response.status_code == 200:
        print("✅ รายการบัญชีทั้งหมด:", response.json())
    else:
        print(f"❌ ไม่สามารถดึงข้อมูลบัญชี: {response.status_code}, {response.json()}")

# 📌 ✅ 4. เติมเงินเข้าบัญชี Sandbox
def deposit_funds(account_id, amount):
    url = f"{ALPACA_BASE_URL}/v1/accounts/{account_id}/transfers"
    
    deposit_data = {
        "amount": str(amount),
        "direction": "INCOMING",
        "timing": "IMMEDIATE"
    }

    response = requests.post(url, json=deposit_data, headers=HEADERS_ALPACA)

    if response.status_code == 201:
        print(f"✅ เติมเงิน {amount} USD เข้าบัญชี {account_id} สำเร็จ:", response.json())
    else:
        print(f"❌ ไม่สามารถเติมเงิน: {response.status_code}, {response.json()}")

# 📌 ✅ 5. ดึงสินทรัพย์ที่สามารถเทรดได้
def get_assets():
    url = f"{ALPACA_BASE_URL}/v1/assets"
    response = requests.get(url, headers=HEADERS_ALPACA)

    if response.status_code == 200:
        print("✅ รายชื่อสินทรัพย์ที่เทรดได้:", response.json())
    else:
        print(f"❌ ไม่สามารถดึงข้อมูลสินทรัพย์: {response.status_code}, {response.json()}")

# 📌 ✅ 6. ส่งคำสั่งซื้อขายหุ้นใน Sandbox
def place_trade(account_id, stock_symbol, trade_type, quantity):
    url = f"{ALPACA_BASE_URL}/v1/trading/accounts/{account_id}/orders"

    trade_payload = {
        "symbol": stock_symbol,
        "qty": quantity,
        "side": trade_type.lower(),
        "type": "market",
        "time_in_force": "gtc"
    }

    response = requests.post(url, json=trade_payload, headers=HEADERS_ALPACA)

    if response.status_code == 201:
        print(f"✅ คำสั่ง {trade_type} {stock_symbol} สำเร็จ:", response.json())
    else:
        print(f"❌ คำสั่ง {trade_type} {stock_symbol} ล้มเหลว: {response.status_code}, {response.json()}")

# 📌 ✅ 7. ดึงข้อมูลคำสั่งซื้อขาย
def get_orders(account_id):
    url = f"{ALPACA_BASE_URL}/v1/trading/accounts/{account_id}/orders"
    response = requests.get(url, headers=HEADERS_ALPACA)

    if response.status_code == 200:
        print("✅ คำสั่งซื้อขายที่ดำเนินการแล้ว:", response.json())
    else:
        print(f"❌ ไม่สามารถดึงข้อมูลคำสั่งซื้อขาย: {response.status_code}, {response.json()}")

# 📌 🔥 เรียกใช้งานฟังก์ชัน
if __name__ == "__main__":
    # 🔹 1. ทดสอบว่า API Key ใช้งานได้
    test_sandbox_api()
    
    # 🔹 2. สร้างบัญชีใหม่ใน Sandbox
    create_sandbox_account()
    
    # 🔹 3. ดึงบัญชีทั้งหมดที่มีอยู่
    get_all_accounts()

    # 🔹 4. เติมเงินเข้า Sandbox (ใส่ account_id ที่ได้จากการสร้างบัญชี)
    # deposit_funds("ACCOUNT_ID_HERE", 100000)

    # 🔹 5. ดึงสินทรัพย์ที่สามารถเทรดได้
    # get_assets()

    # 🔹 6. ส่งคำสั่งซื้อหุ้น
    # place_trade("ACCOUNT_ID_HERE", "AAPL", "BUY", 10)

    # 🔹 7. ดึงคำสั่งซื้อขาย
    # get_orders("ACCOUNT_ID_HERE")
