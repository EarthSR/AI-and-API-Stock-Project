import os
import csv
import mysql.connector
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# โหลด env
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.env')
load_dotenv(path)

# พาธ
MERGED_CSV_PATH = os.path.join(os.path.dirname(__file__), "Stock", "merged_stock_sentiment_financial.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "Stock", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    print("📥 กำลังโหลดไฟล์ merged_stock_sentiment_financial.csv ...")
    df = pd.read_csv(MERGED_CSV_PATH)
    print("🧾 คอลัมน์ทั้งหมดในไฟล์:")
    print(df.columns.tolist())
    print(f"✅ โหลดข้อมูลสำเร็จ: {len(df)} แถว")

    df = df.rename(columns={
        "Ticker": "StockSymbol",
        "Open": "OpenPrice",
        "High": "HighPrice",
        "Low": "LowPrice",
        "Close": "ClosePrice",
        'P/E Ratio': 'PERatio',
        "ROE (%)": "ROE",
        "QoQ Growth (%)": "QoQGrowth",
        "YoY Growth (%)": "YoYGrowth",
        "Total Revenue": "TotalRevenue",
        "Net Profit": "NetProfit",
        "Earnings Per Share (EPS)": "EPS",
        "Gross Margin (%)": "GrossMargin",
        "Net Profit Margin (%)": "NetProfitMargin",
        "EVEBITDA": "EVEBITDA",
        'Debt to Equity': 'DebtToEquity',
        "MarketCap": "MarketCap",
        'P/BV Ratio': 'PBVRatio',
        "Dividend Yield (%)": "Dividend_Yield",
        "Sentiment": "Sentiment"
    })

    company_dict = {
        "ADVANC": ("Advanced Info Service Public Company Limited", "Thailand", "Communication", "Telecom Services", "Advanced Info Service Public Company Limited operates as a telecommunications and technology company primarily in Thailand..."),
        # เพิ่มบริษัทอื่น ๆ ตามต้องการ
    }

    df["Change"] = df["ClosePrice"] - df["OpenPrice"]
    df["Changepercen"] = (df["Change"] / df["OpenPrice"]) * 100

    df["CompanyName"] = df["StockSymbol"].map(lambda x: company_dict.get(x, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))[0])
    df["Market"] = df["StockSymbol"].map(lambda x: company_dict.get(x, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))[1])
    df["Sector"] = df["StockSymbol"].map(lambda x: company_dict.get(x, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))[2])
    df["Industry"] = df["StockSymbol"].map(lambda x: company_dict.get(x, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))[3])
    df["Description"] = df["StockSymbol"].map(lambda x: company_dict.get(x, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))[4])
    df["Sentiment"] = df["Sentiment"].fillna("Neutral")
    df = df.where(pd.notna(df), None)

    stock_data = df[["StockSymbol", "Market", "CompanyName", "Sector", "Industry", "Description"]].drop_duplicates(subset=["StockSymbol"], keep="last")

    stock_detail_data = df[[
        "Date", "StockSymbol", "OpenPrice", "HighPrice", "LowPrice", "ClosePrice", "PERatio", "ROE",
        "QoQGrowth", "YoYGrowth", "TotalRevenue", "NetProfit", "EPS", "GrossMargin", "NetProfitMargin", "DebtToEquity",
        "Changepercen", "Volume", "EVEBITDA", "MarketCap", "PBVRatio", "Dividend_Yield", "Sentiment"
    ]]

    stock_detail_data["PredictionTrend"] = None
    stock_detail_data["PredictionClose"] = None
    stock_detail_data["YoYGrowth"] = pd.to_numeric(stock_detail_data["YoYGrowth"], errors='coerce').fillna(0)
    stock_detail_data["YoYGrowth"] = np.clip(stock_detail_data["YoYGrowth"], -100, 100)
    stock_detail_data["Volume"] = stock_detail_data["Volume"].replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)
    stock_detail_data["EVEBITDA"] = stock_detail_data["EVEBITDA"].replace([np.inf, -np.inf], np.nan)

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

        def convert_nan_to_none(data_list):
            return [[None if (isinstance(x, float) and np.isnan(x)) else x for x in row] for row in data_list]

        insert_stock_query = """
        INSERT INTO Stock (StockSymbol, Market, CompanyName, Sector, Industry, Description)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            Market=COALESCE(VALUES(Market), Market),
            CompanyName=COALESCE(VALUES(CompanyName), CompanyName),
            Sector=COALESCE(VALUES(Sector), Sector),
            Industry=COALESCE(VALUES(Industry), Industry),
            Description=COALESCE(VALUES(Description), Description);
        """

        stock_values = convert_nan_to_none(stock_data.values.tolist())
        cursor.executemany(insert_stock_query, stock_values)
        print(f"✅ บันทึกข้อมูลลง Stock: {len(stock_values)} รายการ")

        insert_stock_detail_query = """
        INSERT INTO StockDetail (
            Date, StockSymbol, OpenPrice, HighPrice, LowPrice, ClosePrice, PERatio, ROE, QoQGrowth, YoYGrowth,
            TotalRevenue, NetProfit, EPS, GrossMargin, NetProfitMargin, DebtToEquity, Changepercen, Volume,
            EVEBITDA, MarketCap, P_BV_Ratio, Dividend_Yield, Sentiment, PredictionTrend, PredictionClose
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE  
            OpenPrice=VALUES(OpenPrice), HighPrice=VALUES(HighPrice), LowPrice=VALUES(LowPrice), 
            ClosePrice=VALUES(ClosePrice), PERatio=VALUES(PERatio), ROE=VALUES(ROE), 
            QoQGrowth=VALUES(QoQGrowth), YoYGrowth=VALUES(YoYGrowth), TotalRevenue=VALUES(TotalRevenue), 
            NetProfit=VALUES(NetProfit), EPS=VALUES(EPS), GrossMargin=VALUES(GrossMargin), 
            NetProfitMargin=VALUES(NetProfitMargin), DebtToEquity=VALUES(DebtToEquity),
            Changepercen=VALUES(Changepercen), Volume=VALUES(Volume), EVEBITDA=VALUES(EVEBITDA), 
            MarketCap=VALUES(MarketCap), P_BV_Ratio =VALUES(P_BV_Ratio), Dividend_Yield=VALUES(Dividend_Yield), 
            Sentiment=VALUES(Sentiment), PredictionTrend=COALESCE(VALUES(PredictionTrend), PredictionTrend), 
            PredictionClose=COALESCE(VALUES(PredictionClose), PredictionClose);
        """

        stock_detail_values = convert_nan_to_none(stock_detail_data.values.tolist())
        batch_size = 1000
        total_rows = len(stock_detail_values)

        for i in range(0, total_rows, batch_size):
            batch = stock_detail_values[i:i+batch_size]
            cursor.executemany(insert_stock_detail_query, batch)
            print(f"✅ บันทึกข้อมูลลง StockDetail batch {i//batch_size + 1}: {len(batch)} รายการ")
        
        print(f"✅ บันทึกข้อมูลลง StockDetail ทั้งหมด: {total_rows} รายการ")

    except mysql.connector.Error as err:
        print(f"❌ เกิดข้อผิดพลาดกับฐานข้อมูล: {err}")

    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
            print("🔹 ปิดการเชื่อมต่อฐานข้อมูลแล้ว")

except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการประมวลผลข้อมูล: {e}")
