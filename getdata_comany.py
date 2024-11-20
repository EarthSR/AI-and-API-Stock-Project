import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
import yfinance as yf

# ฟังก์ชันดึงรายชื่อหุ้นในดัชนี SET100
def get_set_stocks():
    url = 'https://www.settrade.com/th/equities/market-data/overview?category=Index&index'

    # เปิดเบราว์เซอร์
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')  # ข้ามการตรวจสอบ SSL
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    try:
        # คลิก Dropdown เพื่อเปิดตัวเลือก
        print("Attempting to click dropdown...")
        dropdown = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "multiselect__select"))
        )
        dropdown.click()
        print("Dropdown clicked successfully!")

        # เลือกตัวเลือก "100"
        print("Selecting 'ทั้งหมด' option...")
        option_all = WebDriverWait(driver, 1).until(
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

# ฟังก์ชันดึงข้อมูลงบการเงินของหุ้นใน SET
def fetch_financials_for_set_yfinance():
    stocks = get_set_stocks()  # ฟังก์ชันที่ดึงรายชื่อหุ้นใน SET (คุณต้องกำหนดเอง)
    all_financials = []

    for ticker in stocks:
        print(f"Fetching financial data for: {ticker}")
        financials_df = get_financials_yfinance(ticker)
        if financials_df is not None:
            financials_df["Ticker"] = ticker  # เพิ่มคอลัมน์ Ticker
            all_financials.append(financials_df.T)  # ทรานสโพสเพื่อให้เป็น DataFrame

    # รวมข้อมูลทั้งหมดเป็น DataFrame
    if all_financials:
        combined_df = pd.concat(all_financials, ignore_index=True)
        combined_df.to_csv("SET_Financials_Yahoo.csv", index=False)
        print("Financial data saved to SET_Financials_Yahoo.csv")
    else:
        print("No financial data found.")

# ใส่ API Key ของคุณ
fetch_financials_for_set_yfinance()

