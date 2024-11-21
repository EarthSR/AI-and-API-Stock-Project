import requests
import pandas as pd
import os
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
import yfinance as yf

# ฟังก์ชันดึงรายชื่อหุ้นในดัชนี SET
def get_set_stocks():
    url = 'https://www.settrade.com/th/equities/market-data/overview?category=Index&index'

    # เปิดเบราว์เซอร์
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')  # ข้ามการตรวจสอบ SSL
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    try:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        # คลิก Dropdown เพื่อเปิดตัวเลือก
        print("Attempting to click dropdown...")
        dropdown = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "multiselect__select"))
        )
        dropdown.click()
        print("Dropdown clicked successfully!")

        # เลือกตัวเลือก "ทั้งหมด"
        print("Selecting 'ทั้งหมด' option...")
        option_all = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//li[contains(., "ทั้งหมด")]'))
        )
        option_all.click()
        print("'ทั้งหมด' option selected!")

        # รอให้หน้าโหลดข้อมูลทั้งหมด
        print("Waiting for all elements to load...")
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "symbol"))
        )
        print("All elements loaded successfully!")

        # ดึง HTML หลังจากเลือกตัวเลือก
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        driver.quit()

        # ดึง ticker symbols จาก HTML
        print("Extracting stock symbols...")
        stock_elements = soup.find_all('a', {'class': 'symbol'})
        stocks = [stock.text.strip() for stock in stock_elements]
        print(f"Found {len(stocks)} stock symbols.")
        return stocks

    except Exception as e:
        print(f"Error: {e}")
        driver.quit()
        return []

# ฟังก์ชันดึงงบการเงินจาก Yahoo Finance
def get_financials_yfinance(ticker):
    try:
        stock = yf.Ticker(ticker + ".BK")  # เพิ่ม .BK สำหรับหุ้นไทย
        financials = stock.financials  # ดึงงบการเงิน
        return financials
    except Exception as e:
        print(f"Error fetching financials for {ticker}: {e}")
        return None

# ฟังก์ชันรวมข้อมูลการเงินของหุ้นทั้งหมด
def fetch_and_save_all_financials():
    stocks = get_set_stocks()  # ดึงรายชื่อหุ้น
    all_financials = []  # เก็บข้อมูลทั้งหมด

    for ticker in stocks:
        print(f"Fetching financial data for: {ticker}")
        financials_df = get_financials_yfinance(ticker)
        if financials_df is not None:
        # ทรานสโพสข้อมูลและเพิ่มคอลัมน์ชื่อหุ้น
            financials_df = financials_df.T
            financials_df['Stock'] = ticker

        # รีเซ็ตและจัดเรียงคอลัมน์
            financials_df = financials_df.reset_index().rename(columns={'index': 'Date'})
            financials_df = financials_df[['Stock', 'Date'] + [col for col in financials_df.columns if col not in ['Stock', 'Date']]]

        # เพิ่ม DataFrame เข้ารายการ
            all_financials.append(financials_df)


    # รวมข้อมูลทั้งหมดใน DataFrame เดียว
    if all_financials:
        combined_financials = pd.concat(all_financials, ignore_index=True)

        # สร้างโฟลเดอร์สำหรับบันทึก
        folder = "Financials"
        if not os.path.exists(folder):
            os.makedirs(folder)

        # บันทึกข้อมูลลงไฟล์ CSV
        filename = os.path.join(folder, "Combined_Financials.csv")
        combined_financials.to_csv(filename, index=False)
        print(f"ข้อมูลการเงินทั้งหมดถูกบันทึกในไฟล์ {filename}")
    else:
        print("ไม่พบข้อมูลสำหรับบันทึก")

if __name__ == "__main__":
    fetch_and_save_all_financials()
