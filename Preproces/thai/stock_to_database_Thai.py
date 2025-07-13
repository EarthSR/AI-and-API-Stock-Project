import os
import csv
import mysql.connector
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import sys
import io

# Set UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.env')
load_dotenv(path)

# File paths
MERGED_CSV_PATH = os.path.join(os.path.dirname(__file__), "Stock", "merged_stock_sentiment_financial.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "Stock", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    print("📥 กำลังโหลดไฟล์ merged_stock_sentiment_financial.csv ...")
    df = pd.read_csv(MERGED_CSV_PATH)
    print("🧾 คอลัมน์ทั้งหมดในไฟล์:")
    print(df.columns.tolist())
    print(f"✅ โหลดข้อมูลสำเร็จ: {len(df)} แถว")

    # Rename columns to match database schema
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

    # Company dictionary
    company_dict = {
        "ADVANC": ("Advanced Info Service Public Company Limited", "Thailand", "Communication", "Telecom Services", "Advanced Info Service Public Company Limited operates as a telecommunications and technology company primarily in Thailand..."),
        "INTUCH": ("Intouch Holdings Public Company Limited", "Thailand", "Communication", "Telecom Holding", "Intouch Holdings Public Company Limited, through its subsidiaries, engages in the telecommunications and other businesses in Thailand..."),
        "TRUE": ("True Corporation Public Company Limited", "Thailand", "Communication", "Telecom Services", "True Corporation Public Company Limited, together with its subsidiaries, provides telecommunications and value-added services in Thailand..."),
        "DITTO": ("DITTO (Thailand) Public Company Limited", "Thailand", "Technology", "IT Solutions", "Ditto (Thailand) Public Company Limited distributes data and document management solutions in Thailand..."),
        "DIF": ("Digital Telecommunications Infrastructure Fund", "Thailand", "Real Estate", "Infrastructure Fund", "We own or are entitled to the net revenues generated from a portfolio of 16,059 telecommunications towers..."),
        "INSET": ("Infraset Public Company Limited", "Thailand", "Technology", "IT Infrastructure", "Infraset Public Company Limited constructs data centers, information technology system, infrastructure, and telecommunication transportation infrastructure in Thailand..."),
        "JMART": ("Jay Mart Public Company Limited", "Thailand", "Consumer", "Retail", "Jaymart Group Holdings Public Company Limited, through its subsidiaries, engages in the wholesale and retail of mobile phones, accessories, and gadgets in Thailand..."),
        "INET": ("Internet Thailand Public Company Limited", "Thailand", "Technology", "Cloud Computing", "Internet Thailand Public Company Limited, together with its subsidiaries, provides Internet access, and information and communication technology services..."),
        "JAS": ("Jasmine International Public Company Limited", "Thailand", "Communication", "Broadband Services", "Jasmine International Public Company Limited engages in the telecommunications business in Thailand..."),
        "HUMAN": ("Humanica Public Company Limited", "Thailand", "Technology", "HR Software", "Humanica Public Company Limited provides human resource services and solutions in Thailand, Singapore, Japan, Malaysia, Indonesia, Vietnam, Philippines, and internationally..."),
        "AMD": ("Advanced Micro Devices Inc.", "America", "Technology", "Semiconductors", "Advanced Micro Devices, Inc. operates as a semiconductor company worldwide..."),
        "TSM": ("Taiwan Semiconductor Manufacturing Company", "America", "Technology", "Semiconductors", "Taiwan Semiconductor Manufacturing Company Limited, together with its subsidiaries, manufactures, packages, tests, and sells integrated circuits..."),
        "AVGO": ("Broadcom Inc.", "America", "Technology", "Semiconductors", "Broadcom Inc. designs, develops, and supplies various semiconductor devices..."),
        "TSLA": ("Tesla Inc.", "America", "Consumer", "Electric Vehicles", "Tesla, Inc. designs, develops, manufactures, leases, and sells electric vehicles, and energy generation and storage systems..."),
        "META": ("Meta Platforms Inc.", "America", "Technology", "Social Media", "Meta Platforms, Inc. engages in the development of products that enable people to connect and share with friends and family..."),
        "GOOGL": ("Alphabet Inc. (Google)", "America", "Technology", "Internet Services", "Alphabet Inc. offers various products and platforms in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America..."),
        "AMZN": ("Amazon.com Inc.", "America", "Consumer", "E-Commerce", "Amazon.com, Inc. engages in the retail sale of consumer products, advertising, and subscriptions service through online and physical stores..."),
        "NVDA": ("NVIDIA Corporation", "America", "Technology", "Semiconductors", "NVIDIA Corporation, a computing infrastructure company, provides graphics and compute and networking solutions..."),
        "MSFT": ("Microsoft Corporation", "America", "Technology", "Software", "Microsoft Corporation develops and supports software, services, devices and solutions worldwide..."),
        "AAPL": ("Apple Inc.", "America", "Technology", "Consumer Electronics", "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide...")
    }

    # Calculate Change and Changepercen
    df["Change"] = df["ClosePrice"] - df["OpenPrice"]
    df["Changepercen"] = (df["Change"] / df["OpenPrice"]) * 100

    # Map company information
    df["CompanyName"] = df["StockSymbol"].map(lambda x: company_dict.get(x, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))[0])
    df["Market"] = df["StockSymbol"].map(lambda x: company_dict.get(x, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))[1])
    df["Sector"] = df["StockSymbol"].map(lambda x: company_dict.get(x, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))[2])
    df["Industry"] = df["StockSymbol"].map(lambda x: company_dict.get(x, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))[3])
    df["Description"] = df["StockSymbol"].map(lambda x: company_dict.get(x, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))[4])
    df["Sentiment"] = df["Sentiment"].fillna("Neutral")
    if df["Sentiment"] == 0:
        df["Sentiment"] = "Neutral"
    if df["Sentiment"] == 1:
        df["Sentiment"] = "Positive"
    if df["Sentiment"] == -1:
        df["Sentiment"] = "Negative"
    df = df.where(pd.notna(df), None)

    # Prepare stock data for Stock table
    stock_data = df[["StockSymbol", "Market", "CompanyName", "Sector", "Industry", "Description"]].drop_duplicates(subset=["StockSymbol"], keep="last")

    # Prepare stock detail data for StockDetail table
    stock_detail_data = df[[
        "Date", "StockSymbol", "OpenPrice", "HighPrice", "LowPrice", "ClosePrice",
        "PERatio", "ROE", "QoQGrowth", "YoYGrowth", "TotalRevenue", "NetProfit",
        "EPS", "GrossMargin", "NetProfitMargin", "DebtToEquity", "Changepercen",
        "Volume", "EVEBITDA", "MarketCap", "PBVRatio", "Dividend_Yield", "Sentiment"
    ]]

    # Add news columns
    stock_detail_data["positive_news"] = df["positive_news"].fillna(0)
    stock_detail_data["negative_news"] = df["negative_news"].fillna(0)
    stock_detail_data["neutral_news"] = df["neutral_news"].fillna(0)

    # Data cleaning
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
            charset='utf8mb4',
            autocommit=True
        )
        cursor = conn.cursor()
        print("✅ เชื่อมต่อฐานข้อมูลสำเร็จ!")

        def convert_nan_to_none(data_list):
            return [[None if (isinstance(x, float) and np.isnan(x)) else x for x in row] for row in data_list]

        # Insert into Stock table
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

        # Insert into StockDetail table
        insert_stock_detail_query = """
        INSERT INTO StockDetail (
            Date, StockSymbol, OpenPrice, HighPrice, LowPrice, ClosePrice,
            PERatio, ROE, QoQGrowth, YoYGrowth, TotalRevenue, NetProfit, EPS,
            GrossMargin, NetProfitMargin, DebtToEquity, Changepercen, Volume,
            EVEBITDA, MarketCap, P_BV_Ratio, Dividend_Yield, Sentiment,
            positive_news, negative_news, neutral_news
        )
        VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            OpenPrice = VALUES(OpenPrice),
            HighPrice = VALUES(HighPrice),
            LowPrice = VALUES(LowPrice),
            ClosePrice = VALUES(ClosePrice),
            PERatio = VALUES(PERatio),
            ROE = VALUES(ROE),
            QoQGrowth = VALUES(QoQGrowth),
            YoYGrowth = VALUES(YoYGrowth),
            TotalRevenue = VALUES(TotalRevenue),
            NetProfit = VALUES(NetProfit),
            EPS = VALUES(EPS),
            GrossMargin = VALUES(GrossMargin),
            NetProfitMargin = VALUES(NetProfitMargin),
            DebtToEquity = VALUES(DebtToEquity),
            Changepercen = VALUES(Changepercen),
            Volume = VALUES(Volume),
            EVEBITDA = VALUES(EVEBITDA),
            MarketCap = VALUES(MarketCap),
            P_BV_Ratio = VALUES(P_BV_Ratio),
            Dividend_Yield = VALUES(Dividend_Yield),
            Sentiment = VALUES(Sentiment),
            positive_news = COALESCE(VALUES(positive_news), positive_news),
            negative_news = COALESCE(VALUES(negative_news), negative_news),
            neutral_news = COALESCE(VALUES(neutral_news), neutral_news);
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