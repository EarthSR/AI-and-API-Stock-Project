import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager  # ใช้ WebDriverManager
import re

# 🔹 ตั้งค่า Chrome options
options = Options()
options.add_argument('--headless')  # ทำงานแบบไม่มี UI
options.add_argument('--disable-gpu')
options.add_argument('--ignore-certificate-errors')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--blink-settings=imagesEnabled=false')  # ปิดการโหลดรูปภาพ

# 🔹 เริ่มต้น Chrome driver อัตโนมัติ
print("🚀 กำลังเปิด WebDriver...")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)
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

# ฟังก์ชันแปลงชื่อคอลัมน์จากภาษาไทยเป็นภาษาอังกฤษ
column_translation = {
    "รายได้รวม": "Total Revenue",
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
    "กำไรต่อหุ้น (ดอลลาร์สหรัฐฯ)": "EPS (USD)",
    "ROA (%)": "ROA (%)",
    "ROE (%)": "ROE (%)",
    "อัตรากำไรขั้นต้น (%)": "Gross Margin (%)",
    "อัตราส่วนการขายและบริหารต่อรายได้ (%)": "Selling & Admin Expense to Revenue (%)",
    "อัตรากำไรสุทธิ (%)": "Net Profit Margin (%)",
    "หนี้สิน/ทุน (เท่า)": "Debt to Equity (x)",
    "วงจรเงินสด (วัน)": "Cash Cycle (Days)",
    "ราคาล่าสุด (ดอลลาร์สหรัฐฯ)": "Last Price (USD)",
    "มูลค่าหลักทรัพย์ตามราคาตลาด (ล้านดอลลาร์สหรัฐฯ)": "Market Cap (Million USD)",
    "P/E (เท่า)": "P/E Ratio (x)",
    "P/BV (เท่า)": "P/BV Ratio (x)",
    "มูลค่าหุ้นทางบัญชีต่อหุ้น (ดอลลาร์สหรัฐฯ)": "Book Value Per Share (USD)",
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

        # ✅ คลิกปุ่มเปลี่ยนจาก "ไตรมาส" เป็น "ปี"
        print("🔄 กำลังคลิกปุ่มเปลี่ยนเป็น 'ปี' ...")
        toggle_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//div[@data-alias="btn_growth_summary_year"]'))
        )
        driver.execute_script("arguments[0].click();", toggle_button)
        print("✅ คลิกเปลี่ยนเป็น 'ปี' สำเร็จ!")

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
            years = [th.text.strip() for th in rows[0].find_all("th")[1:] if "256" in th.text]
            values_dict = {year: [] for year in years}

            for row in rows[1:]:
                cols = row.find_all("td")
                metric_name = cols[0].text.strip()
                for year, col in zip(years, cols[1:]):
                    value = col.text.strip().replace(",", "")
                    try:
                        values_dict[year].append(float(value))  # แปลงเป็น float ถ้าเป็นตัวเลข
                    except ValueError:
                        values_dict[year].append(value)  # ถ้าไม่ใช่ตัวเลข ให้เก็บเป็น string

            # ✅ สร้าง DataFrame
            df = pd.DataFrame(values_dict, index=[row.find("td").text.strip() for row in rows[1:]]).T
            df.insert(0, "Stock", stock)
            df.insert(1, "Year", df.index)
            df.reset_index(drop=True, inplace=True)

            # แปลงปีเป็น ค.ศ.
            df['Year'] = df['Year'].apply(clean_year)
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

        columns_to_keep = ['Stock', 'Year'] + columns_to_keep[2:]  # กรองให้ไม่เพิ่ม 'Year' ซ้ำ
        full_df = full_df[columns_to_keep]

        # ✅ แทนที่ "N/A" ด้วยค่าว่าง (null)
        full_df = full_df.replace("N/A", "")

        # ✅ แปลงชื่อคอลัมน์เป็นภาษาอังกฤษ
        full_df = translate_columns(full_df, column_translation)

        # ✅ เรียงปีจากใหม่ไปเก่า
        full_df = full_df.sort_values(by="Year", ascending=False)

        # ✅ จัดเรียงคอลัมน์ให้ Stock & Year อยู่ข้างหน้า
        columns_order = ["Stock", "Year"] + [col for col in full_df.columns if col not in ["Stock", "Year"]]
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
final_df.to_csv("financial_thai_data.csv", index=False, encoding="utf-8-sig")
print("✅ บันทึกข้อมูลลง 'financial_thai_data.csv' สำเร็จ!")

# ✅ ปิด WebDriver
driver.quit()
print("🛑 ปิด WebDriver เรียบร้อย!")
