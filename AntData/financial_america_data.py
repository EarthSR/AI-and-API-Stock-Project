import time
import pandas as pd
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

# 🔹 ตั้งค่า Chrome options
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--ignore-certificate-errors')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--blink-settings=imagesEnabled=false')

# 🔹 เริ่มต้น Chrome driver
print("🚀 กำลังเปิด WebDriver...")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)
print("✅ WebDriver เปิดสำเร็จ!")

# 🔹 ฟังก์ชันแปลงค่าปีให้ถูกต้อง
def clean_year(value):
    if isinstance(value, str):
        # 🔹 ถ้ามีวันที่/เดือน เช่น "11 ก.พ. 68" → ดึงเฉพาะปี
        match = re.search(r"\b(\d{2,4})\b", value)
        if match:
            year = int(match.group())
            
            # 🔹 ถ้าเป็น พ.ศ. (มากกว่า 2500) → แปลงเป็น ค.ศ.
            if year > 2500:
                return str(year - 543)

            # 🔹 ถ้าเป็นเลข 2 หลัก และไม่มี พ.ศ.
            elif 50 <= year <= 99:  # 68 → 2568 → 2025
                return str(1900 + year)

            elif 0 <= year <= 49:  # 25 → 2025, 30 → 2030
                return str(2000 + year)

        return None  # ถ้าหาปีไม่ได้
    return value

# 🔹 ฟังก์ชันดึงข้อมูล
def fetch_full_financial_data(stock):
    url = f"https://www.finnomena.com/stock/{stock}.US"
    print(f"🌍 เปิดเว็บ: {url}")
    driver.get(url)

    try:
        # ✅ รอให้หน้าโหลด
        print("⏳ กำลังรอให้หน้าโหลด...")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "a-toggle-switchtext"))
        )
        print("✅ หน้าโหลดเสร็จแล้ว!")

        # ✅ กดปุ่มเปลี่ยนเป็น "ปี"
        try:
            print("🔄 กำลังคลิกปุ่มเปลี่ยนเป็น 'ปี' ...")
            toggle_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//div[@data-alias="btn_growth_summary_year"]'))
            )
            driver.execute_script("arguments[0].click();", toggle_button)
            print("✅ คลิกเปลี่ยนเป็น 'ปี' สำเร็จ!")
            time.sleep(3)
        except:
            print(f"⚠️ หุ้น {stock} ไม่มีปุ่มเปลี่ยนเป็น 'ปี' หรือเกิดข้อผิดพลาด")

        # ✅ ดึง HTML ของหน้า
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # ✅ ค้นหาตารางงบการเงินทั้งหมด
        tables = soup.find_all("table")

        if not tables:
            print(f"❌ ไม่พบตารางข้อมูลของ {stock}!")
            return None

        print(f"✅ พบ {len(tables)} ตารางข้อมูล!")

        all_data = []

        # 🔹 ดึงข้อมูลจากแต่ละตาราง
        for table in tables:
            rows = table.find_all("tr")
            headers = [th.text.strip() for th in rows[0].find_all("th")[1:]]
            if not any("256" in h or "20" in h for h in headers):
                continue

            years = []
            for header in headers:
                clean_header = clean_year(header)  # 🔹 แปลงปีให้ถูกต้อง
                if clean_header:
                    years.append(clean_header)

            values_dict = {year: [] for year in years}

            for row in rows[1:]:
                cols = row.find_all("td")
                metric_name = cols[0].text.strip()
                for year, col in zip(years, cols[1:]):
                    value = col.text.strip().replace(",", "")

                    try:
                        values_dict[year].append(float(value)) if value else values_dict[year].append(None)
                    except ValueError:
                        values_dict[year].append(value)

            # ✅ สร้าง DataFrame
            df = pd.DataFrame(values_dict, index=[row.find("td").text.strip() for row in rows[1:]]).T
            df.insert(0, "Stock", stock)
            df.insert(1, "Year", df.index)
            df.reset_index(drop=True, inplace=True)

            all_data.append(df)

        # ✅ รวมทุกตารางเข้าด้วยกัน
        if all_data:
            full_df = pd.concat(all_data, axis=1).loc[:, ~pd.concat(all_data, axis=1).columns.duplicated()]
            full_df = full_df.sort_values(by="Year", ascending=False)

            # ✅ จัดเรียงคอลัมน์ให้ Stock & Year อยู่ข้างหน้า
            columns_order = ["Stock", "Year"] + [col for col in full_df.columns if col not in ["Stock", "Year"]]
            full_df = full_df[columns_order]

            print(f"✅ ดึงข้อมูลของ {stock} สำเร็จ!")
            return full_df

    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดขณะดึงข้อมูล {stock}: {e}")
        return None

# ✅ ดึงข้อมูลของหุ้นทั้งหมด
stocks = ["AAPL", "MSTF", "NVDA", "AMZN", "GOOG", "META", "TSLA", "AVGO", "TSM", "AMD "]
all_dfs = []

for stock in stocks:
    print(f"📊 กำลังดึงข้อมูลของ {stock}...")
    df = fetch_full_financial_data(stock)
    if df is not None:
        all_dfs.append(df)

# ✅ รวมข้อมูลทุกหุ้น
final_df = pd.concat(all_dfs, ignore_index=True)

# ✅ บันทึกข้อมูลลง CSV
final_df.to_csv("financial_america_data.csv", index=False, encoding="utf-8-sig")
print("✅ บันทึกข้อมูลลง 'financial_america_data.csv' สำเร็จ!")

# ✅ ปิด WebDriver
driver.quit()
print("🛑 ปิด WebDriver เรียบร้อย!")
