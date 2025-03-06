import os
import csv
import mysql.connector
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import sys

# ✅ ป้องกัน UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# ✅ โหลดตัวแปรจาก .env
load_dotenv()

# ✅ กำหนดพาธของไฟล์ CSV
MERGED_CSV_PATH = "D:\\Stock_Project\\AI-and-API-Stock-Project\\merged_stock_sentiment_financial.csv"
MARKETCAP_THAI_CSV = "D:\\Stock_Project\\AI-and-API-Stock-Project\\Finbert\\stock_data_with_marketcap_thai.csv"
MARKETCAP_USA_CSV = "D:\\Stock_Project\\AI-and-API-Stock-Project\\Finbert\\stock_data_with_marketcap_usa.csv"
STOCK_CSV_PATH = "D:\\Stock_Project\\AI-and-API-Stock-Project\\Stock.csv"
STOCK_DETAIL_CSV_PATH = "D:\\Stock_Project\\AI-and-API-Stock-Project\\StockDetail.csv"

# ✅ **Dictionary ของ CompanyName และ Market**
company_dict = {
    "ADVANC": ("Advanced Info Service Public Company Limited", "Thailand"),
    "INTUCH": ("Intouch Holdings Public Company Limited", "Thailand"),
    "TRUE": ("True Corporation Public Company Limited", "Thailand"),
    "DITTO": ("DITTO (Thailand) Public Company Limited", "Thailand"),
    "DIF": ("Digital Telecommunications Infrastructure Fund", "Thailand"),
    "INSET": ("Infraset Public Company Limited", "Thailand"),
    "JMART": ("Jay Mart Public Company Limited", "Thailand"),
    "INET": ("Internet Thailand Public Company Limited", "Thailand"),
    "JAS": ("Jasmine International Public Company Limited", "Thailand"),
    "HUMAN": ("Humanica Public Company Limited", "Thailand"),
    "AMD": ("Advanced Micro Devices Inc.", "America"),
    "TSM": ("Taiwan Semiconductor Manufacturing Company", "America"),
    "AVGO": ("Broadcom Inc.", "America"),
    "TSLA": ("Tesla Inc.", "America"),
    "META": ("Meta Platforms Inc.", "America"),
    "GOOGL": ("Alphabet Inc. (Google)", "America"),
    "AMZN": ("Amazon.com Inc.", "America"),
    "NVDA": ("NVIDIA Corporation", "America"),
    "MSFT": ("Microsoft Corporation", "America"),
    "AAPL": ("Apple Inc.", "America"),
}

# ✅ โหลดข้อมูลจาก MarketCap CSV
def load_marketcap_data(filepath):
    df = pd.read_csv(filepath)
    df = df.rename(columns={"Ticker": "StockSymbol", "Market Cap": "MarketCap"})
    df = df[["StockSymbol", "Date", "MarketCap"]]
    return df

marketcap_thai_df = load_marketcap_data(MARKETCAP_THAI_CSV)
marketcap_usa_df = load_marketcap_data(MARKETCAP_USA_CSV)

# ✅ รวม MarketCap จากไทยและอเมริกา
marketcap_df = pd.concat([marketcap_thai_df, marketcap_usa_df])

# ✅ โหลดข้อมูลจาก merged_stock_sentiment_financial.csv
print("📥 กำลังโหลดไฟล์ merged_stock_sentiment_financial.csv ...")
df = pd.read_csv(MERGED_CSV_PATH)

# ✅ แปลงชื่อ Column ให้ตรงกับ Database
df = df.rename(columns={
    "Ticker": "StockSymbol",
    "Open": "OpenPrice",
    "High": "HighPrice",
    "Low": "LowPrice",
    "Close": "ClosePrice",
    "P/E Ratio ": "PERatio",
    "ROE (%)": "ROE",
    "Dividend Yield (%)": "DividendYield",
    "QoQ Growth (%)": "QoQGrowth",
    "YoY Growth (%)": "YoYGrowth",
    "Total Revenue": "TotalRevenue",
    "Net Profit": "NetProfit",
    "Earnings Per Share (EPS)": "EPS",
    "Gross Margin (%)": "GrossMargin",
    "Net Profit Margin (%)": "NetProfitMargin",
    "Debt to Equity ": "DebtToEquity"
})

# ✅ เติม CompanyName และ Market
df["CompanyName"] = df["StockSymbol"].map(lambda x: company_dict.get(x, ("Unknown", "Unknown"))[0])
df["Market"] = df["StockSymbol"].map(lambda x: company_dict.get(x, ("Unknown", "Unknown"))[1])

# ✅ ผสม MarketCap ให้ตรงกับ StockSymbol และ Date
df = df.merge(marketcap_df, on=["StockSymbol", "Date"], how="left")

# ✅ จัดการค่า NaN เพื่อให้สอดคล้องกับ Database
df = df.where(pd.notna(df), None)  # แปลง NaN เป็น None เพื่อใช้กับ MySQL

# ✅ แยกข้อมูลสำหรับ Stock และ StockDetail
stock__data = df[["StockSymbol", "Market", "MarketCap", "CompanyName"]].drop_duplicates(subset=["StockSymbol"], keep="last")
stock_detail_data = df[[
    "Date", "StockSymbol", "OpenPrice", "HighPrice", "LowPrice", "ClosePrice", "PERatio", "ROE", "DividendYield",
    "QoQGrowth", "YoYGrowth", "TotalRevenue", "NetProfit", "EPS", "GrossMargin", "NetProfitMargin", "DebtToEquity"
]]
stock_detail_data["PredictionTrend"] = None
stock_detail_data["PredictionClose"] = None

# ✅ บันทึกไฟล์ CSV แยกกัน
print(f"💾 กำลังบันทึกไฟล์ {STOCK_CSV_PATH} ...")
stock__data.to_csv(STOCK_CSV_PATH, index=False, na_rep="NULL")
print(f"✅ บันทึกไฟล์ {STOCK_CSV_PATH} สำเร็จ!")

print(f"💾 กำลังบันทึกไฟล์ {STOCK_DETAIL_CSV_PATH} ...")
stock_detail_data.to_csv(STOCK_DETAIL_CSV_PATH, index=False, na_rep="NULL")
print(f"✅ บันทึกไฟล์ {STOCK_DETAIL_CSV_PATH} สำเร็จ!")

# ✅ เชื่อมต่อฐานข้อมูล
try:
    print("🔗 กำลังเชื่อมต่อกับฐานข้อมูล ...")
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        autocommit=True
    )
    cursor = conn.cursor()
    print("✅ เชื่อมต่อฐานข้อมูลสำเร็จ!")

    # ✅ ฟังก์ชันแปลง NaN เป็น None ก่อน Insert
    def convert_nan_to_none(data_list):
        return [[None if (isinstance(x, float) and np.isnan(x)) else x for x in row] for row in data_list]

    # ✅ **บันทึกข้อมูลลง Stock**
    insert_stock_query = """
    INSERT INTO Stock (StockSymbol, Market, MarketCap, CompanyName)
    VALUES (%s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE Market=VALUES(Market), MarketCap=VALUES(MarketCap), CompanyName=VALUES(CompanyName)
    """

    stock_values = convert_nan_to_none(stock__data.values.tolist())
    cursor.executemany(insert_stock_query, stock_values)
    print(f"✅ บันทึกข้อมูลลง Stock: {len(stock_values)} รายการ")

    # ✅ **บันทึกข้อมูลลง StockDetail**
    insert_stock_detail_query = """
    INSERT INTO StockDetail (
        Date, StockSymbol, OpenPrice, HighPrice, LowPrice, ClosePrice, PERatio, ROE, DividendYield,
        QoQGrowth, YoYGrowth, TotalRevenue, NetProfit, EPS, GrossMargin, NetProfitMargin, DebtToEquity, PredictionTrend, PredictionClose
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE 
        OpenPrice=VALUES(OpenPrice), HighPrice=VALUES(HighPrice), LowPrice=VALUES(LowPrice), 
        ClosePrice=VALUES(ClosePrice), PERatio=VALUES(PERatio), ROE=VALUES(ROE), DividendYield=VALUES(DividendYield),
        QoQGrowth=VALUES(QoQGrowth), YoYGrowth=VALUES(YoYGrowth), TotalRevenue=VALUES(TotalRevenue), 
        NetProfit=VALUES(NetProfit), EPS=VALUES(EPS), GrossMargin=VALUES(GrossMargin), 
        NetProfitMargin=VALUES(NetProfitMargin), DebtToEquity=VALUES(DebtToEquity),
        PredictionTrend=VALUES(PredictionTrend), PredictionClose=VALUES(PredictionClose);
    """
    
    stock_detail_values = convert_nan_to_none(stock_detail_data.values.tolist())
    cursor.executemany(insert_stock_detail_query, stock_detail_values)
    print(f"✅ บันทึกข้อมูลลง StockDetail: {len(stock_detail_values)} รายการ")

except mysql.connector.Error as err:
    print(f"❌ เกิดข้อผิดพลาด: {err}")

finally:
    cursor.close()
    conn.close()
    print("🔹 ปิดการเชื่อมต่อฐานข้อมูลแล้ว")
