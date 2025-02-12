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

# ฟังก์ชันดึงข้อมูลจากไตรมาสของปี 2567
def fetch_quarterly_data_2567(stock):
    print(f"🔄 ดึงข้อมูลไตรมาสของปี 2567 สำหรับ {stock}...")
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # ✅ ค้นหาตารางงบการเงินไตรมาส
    tables = soup.find_all("table")

    if not tables:
        print(f"❌ ไม่พบตารางไตรมาสของ {stock}!")
        return None

    print(f"✅ พบ {len(tables)} ตารางไตรมาส!")

    quarterly_data = {}

    # 🔹 ดึงข้อมูลจากตาราง
    for table in tables:
        rows = table.find_all("tr")
        headers = [th.text.strip() for th in rows[0].find_all("th")[1:]]

        if not any("2567" in h for h in headers):
            continue

        metrics = [row.find("td").text.strip() for row in rows[1:]]
        q_values = {metric: [] for metric in metrics}

        for row in rows[1:]:
            cols = row.find_all("td")[1:]
            metric_name = row.find("td").text.strip()
            for col, header in zip(cols, headers):
                value = col.text.strip().replace(",", "")
                if "2567" in header:
                    q_values[metric_name].append(float(value) if value.replace(".", "", 1).isdigit() else 0)

        # ✅ รวม 4 ไตรมาสให้เป็นข้อมูลรายปี
        for metric, values in q_values.items():
            if len(values) == 4:
                quarterly_data[metric] = sum(values)

    print(f"✅ ดึงข้อมูลไตรมาสของปี 2567 สำหรับ {stock} สำเร็จ!")
    return quarterly_data

# ฟังก์ชันดึงข้อมูลงบการเงินทั้งหมด
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

        # ✅ ดึงข้อมูลไตรมาสของปี 2567
        data_2567 = fetch_quarterly_data_2567(stock)

        # ✅ คลิกปุ่มเปลี่ยนจาก "ไตรมาส" เป็น "ปี"
        print("🔄 กำลังคลิกปุ่มเปลี่ยนเป็น 'ปี' ...")
        toggle_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//div[@data-alias="btn_growth_summary_year"]'))
        )
        driver.execute_script("arguments[0].click();", toggle_button)
        print("✅ คลิกเปลี่ยนเป็น 'ปี' สำเร็จ!")

        # ✅ รอให้ข้อมูลโหลด
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
            all_data.append(df)

        # ✅ รวมทุกตารางเข้าด้วยกัน
        full_df = pd.concat(all_data, axis=1).loc[:, ~pd.concat(all_data, axis=1).columns.duplicated()]

        # ✅ เพิ่มข้อมูลของปี 2567 จากไตรมาส
        if data_2567:
            data_2567["Stock"] = stock
            data_2567["Year"] = "2567"
            df_2567 = pd.DataFrame([data_2567])
            full_df = pd.concat([df_2567, full_df], ignore_index=True)

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
