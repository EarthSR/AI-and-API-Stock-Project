import pandas as pd
from datetime import datetime
import os

# ---------------------------
# 1) ฟังก์ชัน merge หลัก
# ---------------------------
def merge_stock_financial_sentiment(
    stock_filepath: str,
    financial_filepath: str,
    sentiment_df: pd.DataFrame,
    country_name: str = "thai"
):
    """
    อ่านไฟล์ stock + financial แล้ว merge เข้ากับ sentiment (ที่เป็น DataFrame)
    เพื่อให้ได้ข้อมูล (Date, Ticker, ...) + sentiment + financial.
    country_name แค่ใส่ไว้เผื่อ debug แยกว่ามาจากประเทศไหน
    """
    
    # 1. อ่านไฟล์หลัก
    stock_df = pd.read_csv(stock_filepath)
    financial_df = pd.read_csv(financial_filepath)
    
    # --------------------
    # (a) เตรียม DataFrame stock
    # --------------------
    # ให้ Date เป็น datetime
    stock_df["Date"] = pd.to_datetime(stock_df["Date"], errors="coerce")
    
    # ถ้ายังไม่มีคอลัมน์ Quarter ก็สร้างจาก Date
    if "Quarter" not in stock_df.columns:
        stock_df["Quarter"] = stock_df["Date"].dt.to_period("Q").astype(str)
    
    # --------------------
    # (b) เตรียม DataFrame financial
    # --------------------
    # ลบช่องว่างหัวท้ายชื่อคอลัมน์
    financial_df.columns = financial_df.columns.str.strip()
    
    # เปลี่ยนชื่อ "EV / EBITDA" -> "EVEBITDA" (ถ้ามี)
    if "EV / EBITDA" in financial_df.columns:
        financial_df.rename(columns={"EV / EBITDA": "EVEBITDA"}, inplace=True)
    
    # ตัวอย่างลิสต์คอลัมน์สำคัญที่ต้องการเก็บ
    columns_to_keep = [
        "Stock", "Quarter", "QoQ Growth (%)", "Total Revenue", "YoY Growth (%)",
        "Net Profit", "Earnings Per Share (EPS)", "ROA (%)", "ROE (%)", 
        "Gross Margin (%)", "Net Profit Margin (%)", "Debt to Equity (x)",
        "P/E Ratio (x)", "P/BV Ratio (x)", "Dividend Yield (%)", "EVEBITDA", 
        "MarketCap"
    ]
    # ตัดเฉพาะคอลัมน์ที่มีจริงในไฟล์
    financial_df = financial_df[[c for c in columns_to_keep if c in financial_df.columns]]
    
    # ฟังก์ชันแก้ format ของ Quarter เช่น 3Q2022 -> 2022Q3
    def fix_quarter_format(q_str):
        q_str = str(q_str).strip()
        if len(q_str) == 6 and q_str[0].isdigit() and q_str[1] == "Q":
            return q_str[-4:] + "Q" + q_str[0]  # 3Q2022 -> 2022Q3
        return q_str
    
    # ฟังก์ชันคำนวณวันที่สมมติสำหรับ Quarter Date
    def quarter_to_announcement_date(q_str):
        try:
            year, q = int(q_str[:4]), int(q_str[-1])
            quarter_dates = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
            if q == 4:
                year += 1
            return pd.Timestamp(f"{year}-{quarter_dates[q]}")
        except:
            return pd.NaT
    
    financial_df["Quarter"] = financial_df["Quarter"].apply(fix_quarter_format)
    financial_df["Quarter Date"] = financial_df["Quarter"].apply(quarter_to_announcement_date)
    
    # --------------------
    # (c) รวม sentiment เข้ากับ stock
    # --------------------
    # ที่นี่เราจะ Merge ด้วยคีย์ ["Ticker","Date"] แทน 
    # (ต้องมั่นใจว่า sentiment_df ก็มี Ticker และ Date ในรูปแบบเดียวกัน)
    
    merged_df = stock_df.merge(
        sentiment_df,   # เอาคอลัมน์ sentiment ทั้งหมดติดมาได้เลย
        on=["Ticker","Date"],  
        how="left"
    )
    
    # ถ้าไม่มีคอลัมน์ "Sentiment" แต่มี net_sentiment_score ก็สร้างเพิ่ม
    if "Sentiment" not in merged_df.columns and "net_sentiment_score" in merged_df.columns:
        merged_df["Sentiment"] = merged_df["net_sentiment_score"].apply(
            lambda x: "Positive" if x > 0.2 else "Negative" if x < -0.2 else "Neutral"
        )
    
    # เติม NaN ใน Sentiment เป็น Neutral ถ้าต้องการ
    if "Sentiment" in merged_df.columns:
        merged_df["Sentiment"] = merged_df["Sentiment"].fillna("Neutral")
    
    # --------------------
    # (d) merge financial เข้ากับ merged_df
    # --------------------
    # โดยใช้ [Ticker, Quarter] เทียบกับ [Stock, Quarter]
    # ต้องมั่นใจว่า Ticker ใน stock_df กับ Stock ใน financial_df สะกดตรงกัน
    merged_df = merged_df.merge(
        financial_df,
        left_on=["Ticker", "Quarter"],
        right_on=["Stock", "Quarter"],
        how="left"
    )
    
    # ถ้าไม่ต้องการคอลัมน์ Stock ซ้ำ ให้ลบออก
    if "Stock" in merged_df.columns:
        merged_df.drop(columns=["Stock"], inplace=True)
    
    # --------------------
    # (e) ตัวเลือก: ลบคอลัมน์ที่ไม่ใช้
    # --------------------
    columns_to_remove = ["Quarter"]  # ตัวอย่างลบ Quarter ทิ้ง
    merged_df.drop(columns=columns_to_remove, errors="ignore", inplace=True)
    
    # --------------------
    # (f) ตัวเลือก: ถ้าต้องการให้ข้อมูล Financial โผล่เฉพาะวันที่ = Quarter Date เท่านั้น
    # --------------------
    def clean_data_based_on_dates(df):
        columns_to_clean = [
            "QoQ Growth (%)", "Total Revenue", "YoY Growth (%)", "Net Profit",
            "Earnings Per Share (EPS)", "ROA (%)", "ROE (%)", "Gross Margin (%)",
            "Net Profit Margin (%)", "Debt to Equity (x)", "P/E Ratio (x)",
            "P/BV Ratio (x)", "Dividend Yield (%)", "EVEBITDA", "MarketCap"
        ]
        for col in columns_to_clean:
            if col in df.columns:
                df.loc[df["Date"] != df["Quarter Date"], col] = None
        return df
    
    merged_df = clean_data_based_on_dates(merged_df)
    
    # --------------------
    # (g) ตัวเลือก: ลบแถวที่ Close == 0 แต่ไม่มีข้อมูลการเงิน
    # --------------------
    financial_cols = [
        "QoQ Growth (%)", "Total Revenue", "YoY Growth (%)", "Net Profit",
        "Earnings Per Share (EPS)", "ROA (%)", "ROE (%)", "Gross Margin (%)",
        "Net Profit Margin (%)", "Debt to Equity (x)", "P/E Ratio (x)",
        "P/BV Ratio (x)", "Dividend Yield (%)", "EVEBITDA", "MarketCap"
    ]
    valid_financial_cols = [col for col in financial_cols if col in merged_df.columns]
    merged_df = merged_df[
        ~(
            (merged_df["Close"] == 0)
            & (merged_df[valid_financial_cols].isna().all(axis=1))
        )
    ]
    
    # --------------------
    # (h) เติม MarketCap เฉพาะแถวที่ Date == Quarter Date
    # --------------------
    if "MarketCap" in merged_df.columns:
        merged_df["MarketCap"] = merged_df.apply(
            lambda row: row["MarketCap"] if row["Date"] == row["Quarter Date"] else None,
            axis=1
        )
    
    # --------------------
    # (i) จัดการ Quarter Date ที่ยังเป็น NaT
    # --------------------
    merged_df.rename(
        columns={
            "Debt to Equity (x)": "Debt to Equity",
            "P/E Ratio (x)": "P/E Ratio",
            "P/BV Ratio (x)": "P/BV Ratio"
        },
        inplace=True
    )
    
    us_stock = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'TSM', 'AMD']
    thai_stock = ['ADVANC', 'INTUCH', 'TRUE', 'DITTO', 'DIF', 
           'INSET', 'JMART', 'INET', 'JAS', 'HUMAN']
    
    merged_df["Market_ID"] = merged_df["Ticker"].apply(
        lambda x: "US" if x in us_stock else "TH" if x in thai_stock else None
    )
    
    # --------------------
    # เสร็จแล้ว return ออกไป
    # --------------------
    return merged_df


# ---------------------------
# 2) เรียกใช้ฟังก์ชัน merge กับ (ไทย + อเมริกา) แล้ว concat
# ---------------------------
if __name__ == "__main__":

    # 1) อ่าน sentiment ใหม่ ที่มีคอลัมน์ date, Stock, positive_news, negative_news, …
    sentiment_path = os.path.join(os.path.dirname(__file__), "News", "daily_sentiment_summary_USA.csv")
    sentiment_df = pd.read_csv(sentiment_path)
    
    # 2) เปลี่ยนชื่อคอลัมน์ให้ตรงกับ stock
    #    คือ Stock -> Ticker, date -> Date
    sentiment_df.rename(columns={
        "Stock": "Ticker",
        "date": "Date"
    }, inplace=True)
    
    # แปลง Date เป็น datetime
    sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"], errors="coerce")
    
    # 4) Merge ฝั่งอเมริกา
    merged_df_us = merge_stock_financial_sentiment(
        stock_filepath= os.path.join(os.path.dirname(__file__), "Stock", "stock_data_usa.csv"),
        financial_filepath= os.path.join(os.path.dirname(__file__), "Stock", "Financial_America_Quarter.csv"),
        sentiment_df=sentiment_df,
        country_name="usa"
    )
    
    # 6) บันทึกเป็น CSV ไฟล์สุดท้าย
    merged_path = os.path.join(os.path.dirname(__file__),"stock", "merged_stock_sentiment_financial.csv")
    merged_df_us.to_csv(merged_path, index=False)
    
    # 7) ตรวจสอบตัวอย่าง 10 แถว
    print(merged_df_us.head(10))
