from bs4 import BeautifulSoup
import requests
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer
from dotenv import load_dotenv
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
import mysql.connector
from datetime import datetime, timedelta

# โหลดไฟล์ .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.env')
load_dotenv(env_path)

# ดึงค่าจาก .env
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
SECRET_KEY = os.getenv("JWT_SECRET", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME]):
    raise ValueError("❌ ขาดค่าการตั้งค่าฐานข้อมูลในไฟล์ .env")

# -------------------- Database --------------------
def get_db_connection():
    try:
        return mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# -------------------- JWT --------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True, buffered=True)

    try:
        cursor.execute("SELECT * FROM user WHERE UserID = %s", (user_id,))
        user = cursor.fetchone()
        if user is None:
            raise credentials_exception
        return user
    finally:
        cursor.close()
        conn.close()

# -------------------- FastAPI Init --------------------
app = FastAPI()

# -------------------- Endpoints --------------------

@app.post("/create-user")
async def create_user(request: Request):
    data = await request.json()
    username = data.get("username")
    password = data.get("password")

    conn = get_db_connection()
    cursor = conn.cursor()
    hashed_password = pwd_context.hash(password)
    try:
        cursor.execute(
            "INSERT INTO user (Username, Password, Balance) VALUES (%s, %s, %s)",
            (username, hashed_password, 100000)
        )
        conn.commit()
        return {"username": username, "password": "********"}
    except mysql.connector.Error as e:
        raise HTTPException(status_code=400, detail=f"Error creating user: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.post("/login")
async def login(request: Request):
    data = await request.json()
    username = data.get("username")
    password = data.get("password")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True, buffered=True)

    try:
        cursor.execute("SELECT * FROM user WHERE Username = %s", (username,))
        user = cursor.fetchone()
        if not user or not pwd_context.verify(password, user['Password']):
            raise HTTPException(status_code=401, detail="Incorrect username or password")

        access_token = create_access_token(data={"sub": str(user['UserID'])})
        return {"access_token": access_token, "token_type": "bearer"}
    finally:
        cursor.close()
        conn.close()

@app.get("/price/yahoo/{symbol}")
def get_price_yahoo(symbol: str):
    url = f"https://finance.yahoo.com/quote/{symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    try:
        span = soup.find("fin-streamer", {"data-field": "regularMarketPrice"})
        price = float(span.text.replace(',', ''))
        return {"symbol": symbol.upper(), "price": price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch price from Yahoo: {str(e)}")

@app.post("/trade")
async def trade_stock(request: Request, current_user: dict = Depends(get_current_user)):
    data = await request.json()
    user_id = data.get("user_id")
    stock_symbol = data.get("stock_symbol")
    quantity = data.get("quantity")
    trade_type = data.get("trade_type")

    if current_user['UserID'] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to perform trade for this user")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True, buffered=True)

    try:
        cursor.execute("SELECT * FROM user WHERE UserID = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        price_data = get_price_yahoo(stock_symbol)
        price = price_data["price"]
        total_price = quantity * price

        cursor.execute("SELECT * FROM portfolio WHERE UserID = %s", (user_id,))
        portfolio = cursor.fetchone()
        if not portfolio:
            cursor.execute("INSERT INTO portfolio (UserID, TotalValue, ProfitLoss) VALUES (%s, %s, %s)", (user_id, 0.0, 0.0))
            conn.commit()
            cursor.execute("SELECT * FROM portfolio WHERE UserID = %s", (user_id,))
            portfolio = cursor.fetchone()

        cursor.execute("SELECT * FROM portfolioholdings WHERE PortfolioID = %s AND StockSymbol = %s",
                       (portfolio['PortfolioID'], stock_symbol))
        holding = cursor.fetchone()

        if trade_type.upper() == "BUY":
            if user['Balance'] < total_price:
                raise HTTPException(status_code=400, detail="Insufficient balance")

            if holding:
                new_total = holding['Quantity'] + quantity
                avg_buy_price = ((holding['AvgBuyPrice'] * holding['Quantity'] + total_price) / new_total)
                profit_loss = (price - avg_buy_price) * new_total
                cursor.execute("""
                    UPDATE portfolioholdings
                    SET Quantity = %s, AvgBuyPrice = %s, CurrentPrice = %s, ProfitLoss = %s
                    WHERE HoldingID = %s
                """, (new_total, avg_buy_price, price, profit_loss, holding['HoldingID']))
            else:
                cursor.execute("""
                    INSERT INTO portfolioholdings (PortfolioID, StockSymbol, Quantity, AvgBuyPrice, CurrentPrice, ProfitLoss)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (portfolio['PortfolioID'], stock_symbol, quantity, price, price, 0.0))

            cursor.execute("UPDATE user SET Balance = %s WHERE UserID = %s",
                           (user['Balance'] - total_price, user_id))

        elif trade_type.upper() == "SELL":
            if not holding or holding['Quantity'] < quantity:
                raise HTTPException(status_code=400, detail="Not enough shares to sell")

            new_quantity = holding['Quantity'] - quantity
            profit_loss = (price - holding['AvgBuyPrice']) * new_quantity if new_quantity > 0 else 0.0
            if new_quantity == 0:
                cursor.execute("DELETE FROM portfolioholdings WHERE HoldingID = %s", (holding['HoldingID'],))
            else:
                cursor.execute("""
                    UPDATE portfolioholdings
                    SET Quantity = %s, CurrentPrice = %s, ProfitLoss = %s
                    WHERE HoldingID = %s
                """, (new_quantity, price, profit_loss, holding['HoldingID']))

            cursor.execute("UPDATE user SET Balance = %s WHERE UserID = %s",
                           (user['Balance'] + total_price, user_id))
        else:
            raise HTTPException(status_code=400, detail="Invalid trade type")

        cursor.execute("""
            INSERT INTO tradehistory (UserID, PortfolioID, StockSymbol, TradeType, Quantity, Price, TotalTradeValue)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (user_id, portfolio['PortfolioID'], stock_symbol, trade_type.upper(), quantity, price, total_price))

        cursor.execute("SELECT * FROM portfolioholdings WHERE PortfolioID = %s", (portfolio['PortfolioID'],))
        holdings = cursor.fetchall()
        total_value = sum(h['Quantity'] * h['CurrentPrice'] for h in holdings)
        total_profit_loss = sum(h['ProfitLoss'] for h in holdings)

        cursor.execute("UPDATE portfolio SET TotalValue = %s, ProfitLoss = %s WHERE PortfolioID = %s",
                       (total_value, total_profit_loss, portfolio['PortfolioID']))
        conn.commit()

        cursor.execute("SELECT Balance FROM user WHERE UserID = %s", (user_id,))
        updated_balance = cursor.fetchone()['Balance']

        return {
            "message": f"{trade_type} successful",
            "balance": updated_balance,
            "trade_price": price
        }
    finally:
        cursor.close()
        conn.close()

@app.get("/portfolio/{user_id}")
def get_portfolio(user_id: int, current_user: dict = Depends(get_current_user)):
    if current_user['UserID'] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to view this portfolio")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True, buffered=True)

    try:
        cursor.execute("SELECT * FROM user WHERE UserID = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        cursor.execute("SELECT * FROM portfolio WHERE UserID = %s", (user_id,))
        portfolio = cursor.fetchone()
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        cursor.execute("SELECT * FROM portfolioholdings WHERE PortfolioID = %s", (portfolio['PortfolioID'],))
        holdings = cursor.fetchall()
        holdings_data = [{
            "stock_symbol": h['StockSymbol'],
            "quantity": h['Quantity'],
            "avg_buy_price": h['AvgBuyPrice'],
            "current_price": h['CurrentPrice'],
            "profit_loss": h['ProfitLoss'],
            "total_value": h['Quantity'] * h['CurrentPrice']
        } for h in holdings]

        return {
            "user_id": user_id,
            "username": user['Username'],
            "balance": user['Balance'],
            "portfolio_id": portfolio['PortfolioID'],
            "total_value": portfolio['TotalValue'],
            "total_profit_loss": portfolio['ProfitLoss'],
            "holdings": holdings_data
        }
    finally:
        cursor.close()
        conn.close()

@app.get("/trade-history/{user_id}")
def get_trade_history(user_id: int, current_user: dict = Depends(get_current_user)):
    if current_user['UserID'] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to view this trade history")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True, buffered=True)

    try:
        cursor.execute("SELECT * FROM user WHERE UserID = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        cursor.execute("SELECT * FROM tradehistory WHERE UserID = %s", (user_id,))
        trades = cursor.fetchall()
        return {
            "user_id": user_id,
            "username": user['Username'],
            "trade_history": [{
                "trade_id": t['TradeID'],
                "portfolio_id": t['PortfolioID'],
                "stock_symbol": t['StockSymbol'],
                "trade_type": t['TradeType'],
                "quantity": t['Quantity'],
                "price": t['Price'],
                "total_trade_value": t['TotalTradeValue']
            } for t in trades]
        }
    finally:
        cursor.close()
        conn.close()

# -------------------- Main --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API_Papertrade:app", host="127.0.0.1", port=8000, reload=True)
