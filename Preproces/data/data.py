import yfinance as yf
import pandas as pd
import datetime
import sys
import os
import mysql.connector
from dotenv import load_dotenv
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.service import Service  
from selenium.webdriver.firefox.options import Options 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.firefox import GeckoDriverManager  # ใช้ WebDriverManager
import re
import sys
import os
import io
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ✅ กำหนดรายชื่อหุ้นอเมริกา (Top 10)
tickers = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'TSM', 'AMD','ADVANC.BK', 'TRUE.BK', 'DITTO.BK', 'DIF.BK', 
           'INSET.BK', 'JMART.BK', 'INET.BK', 'JAS.BK', 'HUMAN.BK']

# ✅ กำหนดวันเริ่มต้นเดียวกันสำหรับทุกหุ้น
start_date = '2017-12-20'
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

# ✅ เตรียมโฟลเดอร์สำหรับบันทึกข้อมูล
CURRENT_DIR = os.getcwd()
os.makedirs(os.path.join(CURRENT_DIR, "Stock"), exist_ok=True)

# ✅ ดาวน์โหลดข้อมูลจาก yfinance พร้อม Retry
max_retries = 3
retry_count = 0

while retry_count < max_retries:
    try:
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        if data.empty:
            raise ValueError("❌ ไม่สามารถดึงข้อมูลจาก yfinance ได้")
        break
    except Exception as e:
        retry_count += 1
        print(f"⚠️ Error: {e} (ลองใหม่ {retry_count}/{max_retries})")
        if retry_count == max_retries:
            sys.exit(1)

# ✅ รวมข้อมูลหุ้นแต่ละตัว
data_list = []

# เติมข้อมูลด้วย Rolling Mean พร้อมการตรวจสอบ
# สำหรับทั้งชุดข้อมูลฝึกและทดสอบ
for ticker in tickers:
    ticker_data = data[ticker].copy()
    stock_name = ticker.replace('.BK', '')
    ticker_data.index = pd.to_datetime(ticker_data.index)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    ticker_data = ticker_data.reindex(all_dates)

    # ตรวจสอบข้อมูลที่ขาดหาย
    missing_percentage = ticker_data[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().mean() * 100
    if missing_percentage.sum() > 20:
        print(f"⚠️ Warning: {ticker} has excessive missing data ({missing_percentage.sum():.2f}%).")

    # เติมข้อมูลด้วย Rolling Mean เฉพาะราคา
    ticker_data[['Open', 'High', 'Low', 'Close']] = (
        ticker_data[['Open', 'High', 'Low', 'Close']]
        .ffill(limit=2)  # Forward Fill สำหรับช่องว่างสั้น ๆ
        .rolling(window=3, min_periods=1).mean()
    )

    # ตั้ง Volume เป็น 0 ในวันหยุด
    ticker_data['Volume'] = ticker_data['Volume'].fillna(0)

    # คำนวณ Changepercen ใหม่
    ticker_data['Changepercen'] = (ticker_data['Close'] - ticker_data['Open']) / ticker_data['Open'] * 100
    ticker_data['Changepercen'] = ticker_data['Changepercen'].fillna(0)

    ticker_data['Ticker'] = stock_name
    data_list.append(ticker_data)

# ✅ รวมเป็น DataFrame เดียว
cleaned_data = pd.concat(data_list).reset_index().rename(columns={'index': 'Date'})
cleaned_data = cleaned_data[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume' , 
                             'Changepercen']]
cleaned_data = cleaned_data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

# ✅ บันทึกเป็น CSV
output_path = os.path.join(CURRENT_DIR, "Stock", "stock_data.csv")
cleaned_data.to_csv(output_path, index=False)

# ✅ แสดงตัวอย่างข้อมูล
print(cleaned_data.head())

# ✅ ตรวจสอบระดับของโฟลเดอร์ (ปรับ `..` ตามตำแหน่งของไฟล์)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔹 ตั้งค่า Chrome options
options = Options()
options.add_argument('--headless')  # ทำงานแบบไม่มี UI
options.add_argument('--disable-gpu')
options.add_argument('--ignore-certificate-errors')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--blink-settings=imagesEnabled=false')  # ปิดการโหลดรูปภาพ

# 🔹 เริ่มต้น Firefox driver อัตโนมัติ
print("🚀 กำลังเปิด WebDriver...")
service = Service(GeckoDriverManager().install())
driver = webdriver.Firefox(service=service, options=options)
print("✅ WebDriver เปิดสำเร็จ!")

# ฟังก์ชันสำหรับแปลงปีจาก พ.ศ. เป็น ค.ศ.
def clean_year(value):
    if isinstance(value, str):
        match = re.search(r"\b(\d{2,4})\b", value)
        if match:
            year = int(match.group())
            if year > 2500:
                return str(year - 543)  # แปลงจาก พ.ศ. เป็น ค.ศ.
    return value

# ฟังก์ชันสำหรับแปลงชื่อคอลัมน์จากภาษาไทยเป็นภาษาอังกฤษ
column_translation = {
        "รายได้รวม": "Total Revenue",
        "การเติบโตต่อไตรมาส (%)": "QoQ Growth (%)",
        "การเติบโตเทียบปีก่อนหน้า (%)": "YoY Growth (%)",
        "กำไรสุทธิ": "Net Profit",
        "กำไรต่อหุ้น (EPS)": "Earnings Per Share (EPS)",
        "สินทรัพย์รวม": "Total Assets",
        "หนี้สินรวม": "Total Liabilities",
        "ส่วนของผู้ถือหุ้น": "Shareholder Equity",
        "กำไรขั้นต้น": "Gross Profit",
        "ค่าใช้จ่ายในการขายและบริหาร": "Selling & Admin Expenses",
        "ค่าเสื่อมราคาและค่าตัดจำหน่าย": "Depreciation & Amortization",
        "กระแสเงินสดจากการดำเนินงาน": "Operating Cash Flow",
        "กระแสเงินสดจากการลงทุน": "Investing Cash Flow",
        "กระแสเงินสดจากกิจกรรมทางการเงิน": "Financing Cash Flow",
        "ROA (%)": "ROA (%)",
        "ROE (%)": "ROE (%)",
        "อัตรากำไรขั้นต้น (%)": "Gross Margin (%)",
        "อัตราส่วนการขายและบริหารต่อรายได้ (%)": "Selling & Admin Expense to Revenue (%)",
        "อัตรากำไรสุทธิ (%)": "Net Profit Margin (%)",
        "หนี้สิน/ทุน (เท่า)": "Debt to Equity (x)",
        "วงจรเงินสด (วัน)": "Cash Cycle (Days)",
        "P/E (เท่า)": "P/E Ratio (x)",
        "P/BV (เท่า)": "P/BV Ratio (x)",
        "อัตราส่วนเงินปันผลตอบแทน(%)": "Dividend Yield (%)",
        "EV / EBITDA": "EV / EBITDA"
    }

# ฟังก์ชันที่แปลงชื่อคอลัมน์
def translate_columns(df, translation_dict):
    df.columns = [translation_dict.get(col, col) for col in df.columns]
    return df

# ฟังก์ชันดึงข้อมูลงบการเงินทั้งหมด
def fetch_full_financial_data(stock):
    url = f"https://www.finnomena.com/stock/{stock}"

    print(f"🌍 เปิดเว็บ: {url}")
    driver.get(url)

    try:
        # ✅ รอให้หน้าโหลด
        print("⏳ กำลังรอให้หน้าโหลด...")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "a-toggle-switchtext"))
        )
        print("✅ หน้าโหลดเสร็จแล้ว!")

        # ✅ รอให้ข้อมูลปีโหลด
        print("⏳ รอให้ข้อมูลปีโหลด...")
        time.sleep(3)
        print("✅ ข้อมูลปีโหลดเสร็จแล้ว!")

        # ✅ ดึง HTML ของหน้า
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # ✅ ค้นหาตารางงบการเงินทั้งหมด
        tables = soup.find_all("table")

        if not tables:
            print(f"❌ ไม่พบตารางข้อมูลทั้งหมดของ {stock}!")
            return None

        print(f"✅ พบ {len(tables)} ตารางข้อมูล!")

        all_data = []

        # 🔹 ดึงข้อมูลจากแต่ละตาราง
        for table in tables:
            rows = table.find_all("tr")
            quarters = [th.text.strip() for th in rows[0].find_all("th")[1:] if "Q" in th.text]
            values_dict = {quarter: [] for quarter in quarters}

            for row in rows[1:]:
                cols = row.find_all("td")
                metric_name = cols[0].text.strip()
                for year, col in zip(quarters, cols[1:]):
                    value = col.text.strip().replace(",", "")
                    try:
                        values_dict[year].append(float(value))  # แปลงเป็น float ถ้าเป็นตัวเลข
                    except ValueError:
                        values_dict[year].append(value)  # ถ้าไม่ใช่ตัวเลข ให้เก็บเป็น string

            # ✅ สร้าง DataFrame
            df = pd.DataFrame(values_dict, index=[row.find("td").text.strip() for row in rows[1:]]).T
            df.insert(0, "Stock", stock)
            # ✅ แปลง Quarter ให้เป็น "4Q2024" แทน "4Q2567"
            df.insert(1, "Quarter", df.index.map(lambda x: x[:2] + clean_year(x[2:])))

            # ✅ ดึงค่า 'Year' ออกจาก 'Quarter'
            df["Year"] = df["Quarter"].apply(lambda x: int(x[2:]))

            # ✅ สร้างตัวเลขลำดับของ Quarter เพื่อช่วยเรียงให้ถูกต้อง
            quarter_map = {"4Q": 4, "3Q": 3, "2Q": 2, "1Q": 1}
            df["Quarter_Order"] = df["Quarter"].apply(lambda x: quarter_map[x[:2]])
            
            # ✅ เรียงลำดับข้อมูลตาม Year ก่อน แล้วตามลำดับ Quarter
            df = df.sort_values(by=["Year", "Quarter_Order"], ascending=[False, False])

            # ✅ ลบคอลัมน์ที่ใช้ช่วยเรียง
            df = df.drop(columns=["Year", "Quarter_Order"])

            # แปลงปีเป็น ค.ศ.
            df['Quarter'] = df['Quarter'].apply(clean_year)
            all_data.append(df)

        # ✅ รวมทุกตารางเข้าด้วยกัน
        full_df = pd.concat(all_data, axis=1).loc[:, ~pd.concat(all_data, axis=1).columns.duplicated()]

        # ✅ ลบคอลัมน์ที่ซ้ำกัน
        full_df = full_df.loc[:, ~full_df.columns.duplicated()]

        # ✅ กรองคอลัมน์จนถึง "EV / EBITDA"
        columns_to_keep = []
        keep = False
        for col in full_df.columns:
            columns_to_keep.append(col)
            if "EV / EBITDA" in col:
                break

        columns_to_keep = ['Stock', 'Quarter'] + columns_to_keep[2:]  # กรองให้ไม่เพิ่ม 'Year' ซ้ำ
        full_df = full_df[columns_to_keep]

        # ✅ แทนที่ "N/A" ด้วยค่าว่าง (null)
        full_df = full_df.replace("N/A", "").infer_objects(copy=False)

        # ✅ แปลงชื่อคอลัมน์เป็นภาษาอังกฤษ
        full_df = translate_columns(full_df, column_translation)

        # ✅ เรียงปีจากใหม่ไปเก่า
        full_df = full_df.sort_values(by="Quarter", ascending=False)

        # ✅ จัดเรียงคอลัมน์ให้ Stock & Quarter อยู่ข้างหน้า
        columns_order = ["Stock", "Quarter"] + [col for col in full_df.columns if col not in ["Stock", "Quarter"]]
        full_df = full_df[columns_order]

        print("✅ ข้อมูลทั้งหมดรวมกันสำเร็จ!")
        return full_df

    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดขณะดึงข้อมูล {stock}: {e}")
        return None

# ✅ ดึงข้อมูลของหุ้นทั้งหมด
stocks = ["ADVANC", "INTUCH", "TRUE", "DITTO", "DIF", "INSET", "JMART", "INET", "JAS", "HUMAN",'AAPL.us', 'NVDA.us', 'MSFT.us', 'AMZN.us', 'GOOGL.us', 'META.us', 'TSLA.us', 'AVGO.us', 'TSM.us', 'AMD.us']
all_dfs = []

for stock in stocks:
    print(f"📊 กำลังดึงข้อมูลของ {stock}...")
    df = fetch_full_financial_data(stock)
    if df is not None:
        all_dfs.append(df)

# ✅ รวมข้อมูลทุกหุ้น
final_df = pd.concat(all_dfs, ignore_index=True)

# ✅ ลบ .us ออกจากชื่อหุ้นในคอลัมน์ 'Stock'
final_df['Stock'] = final_df['Stock'].str.replace('.us', '', regex=False)

# ✅ บันทึกข้อมูลลง CSV
output_path = os.path.join(SCRIPT_DIR, "Stock", "Financial_Quarter.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_df.to_csv(output_path, index=False)
print(f"✅ บันทึกข้อมูลลง '{os.path.basename(output_path)}' สำเร็จ!")

# ✅ ปิด WebDriver
driver.quit()
print("🛑 ปิด WebDriver เรียบร้อย!")


# ---------------------------
# 1) ฟังก์ชัน merge หลัก
# ---------------------------
def merge_stock_financial_sentiment(
    stock_filepath: str,
    financial_filepath: str,
    sentiment_df: pd.DataFrame,
):
    
    # 1. อ่านไฟล์หลัก
    stock_df = pd.read_csv(stock_filepath)
    financial_df = pd.read_csv(financial_filepath)
    
    # --------------------
    # (a) เตรียม DataFrame stock
    # --------------------
    # ให้ Date เป็น datetime และลบ timezone ถ้ามี
    stock_df["Date"] = pd.to_datetime(stock_df["Date"], errors="coerce").dt.tz_localize(None)
    
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
    # Merge ด้วยคีย์ ["Ticker","Date"]
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
    sentiment_path = os.path.join(os.path.dirname(__file__), "..", "sentimentnews","daily_sentiment_summary.csv")
    sentiment_df = pd.read_csv(sentiment_path)
    
    # 2) เปลี่ยนชื่อคอลัมน์ให้ตรงกับ stock
    #    คือ Stock -> Ticker, date -> Date
    sentiment_df.rename(columns={
        "Stock": "Ticker",
        "date": "Date"
    }, inplace=True)
    
    # แปลง Date เป็น datetime และลบ timezone ถ้ามี
    sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"], errors="coerce").dt.tz_localize(None)

    # 3) Merge ฝั่งไทย
    merged_df_th = merge_stock_financial_sentiment(
        stock_filepath=os.path.join(os.path.dirname(__file__), "Stock", "stock_data.csv"),
        financial_filepath=os.path.join(os.path.dirname(__file__), "Stock", "Financial_Quarter.csv"),
        sentiment_df=sentiment_df
    )
    
    # 6) บันทึกเป็น CSV ไฟล์สุดท้าย
    merged_df_th.to_csv(os.path.join(os.path.dirname(__file__), "Stock", "merged_stock_sentiment_financial.csv"), index=False)
    
    # 7) ตรวจสอบตัวอย่าง 10 แถว
    print(merged_df_th.head(10))