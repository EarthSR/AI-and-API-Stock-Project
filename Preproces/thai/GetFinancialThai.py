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
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ✅ ตรวจสอบระดับของโฟลเดอร์ (ปรับ `..` ตามตำแหน่งของไฟล์)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DASH_DIR = Path(__file__).resolve().parent
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
driver = webdriver.Firefox(options=options)
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
    "กำไรต่อหุ้น (บาท)": "EPS (THB)",
    "ROA (%)": "ROA (%)",
    "ROE (%)": "ROE (%)",
    "อัตรากำไรขั้นต้น (%)": "Gross Margin (%)",
    "อัตราส่วนการขายและบริหารต่อรายได้ (%)": "Selling & Admin Expense to Revenue (%)",
    "อัตรากำไรสุทธิ (%)": "Net Profit Margin (%)",
    "หนี้สิน/ทุน (เท่า)": "Debt to Equity (x)",
    "วงจรเงินสด (วัน)": "Cash Cycle (Days)",
    "ราคาล่าสุด (บาท)": "Last Price (THB)",
    "มูลค่าหลักทรัพย์ตามราคาตลาด (ล้านบาท)": "MarketCap",
    "P/E (เท่า)": "P/E Ratio (x)",
    "P/BV (เท่า)": "P/BV Ratio (x)",
    "มูลค่าหุ้นทางบัญชีต่อหุ้น (บาท)": "Book Value Per Share (THB)",
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
stocks = ["ADVANC", "INTUCH", "TRUE", "DITTO", "DIF", "INSET", "JMART", "INET", "JAS", "HUMAN"]
all_dfs = []

for stock in stocks:
    print(f"📊 กำลังดึงข้อมูลของ {stock}...")
    df = fetch_full_financial_data(stock)
    if df is not None:
        all_dfs.append(df)

# ✅ รวมข้อมูลทุกหุ้น
final_df = pd.concat(all_dfs, ignore_index=True)

# ✅ บันทึกข้อมูลลง CSV
output_path = os.path.join(SCRIPT_DIR, "Stock", "Financial_Thai_Quarter.csv")
output_path_dashboard = os.path.join(DASH_DIR, "..","..","API_Mobile_Web", "data","Financial_Thai_Quarter.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_df.to_csv(output_path, index=False)
final_df.to_csv(output_path_dashboard, index=False)
print(f"✅ บันทึกข้อมูลลง '{os.path.basename(output_path)}' สำเร็จ!")
print(f"✅ บันทึกข้อมูลลง '{os.path.basename(output_path_dashboard)}' สำเร็จ!")


# ✅ ปิด WebDriver
driver.quit()
print("🛑 ปิด WebDriver เรียบร้อย!")
