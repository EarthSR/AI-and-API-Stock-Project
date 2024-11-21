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
        # คลิก Dropdown เพื่อเปิดตัวเลือก
        print("Attempting to click dropdown...")
        dropdown = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "multiselect__select"))
        )
        dropdown.click()
        print("Dropdown clicked successfully!")

        # เลือกตัวเลือก "100"
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

def save_financials_to_csv(financials_df, ticker, folder="Financials"):
    """
    บันทึกข้อมูลการเงินของหุ้นลงในไฟล์ CSV ในโฟลเดอร์ที่กำหนด
    
    Parameters:
        financials_df (DataFrame): ข้อมูลการเงินของหุ้น
        ticker (str): ชื่อย่อหุ้น
        folder (str): ชื่อโฟลเดอร์ที่ต้องการบันทึกไฟล์
    """
    try:
        # ตรวจสอบว่าโฟลเดอร์มีอยู่หรือไม่ หากไม่มีก็สร้างใหม่
        if not os.path.exists(folder):
            os.makedirs(folder)

        # กำหนดเส้นทางไฟล์
        filename = os.path.join(folder, f"{ticker}_Financials.csv")
        
        # บันทึกไฟล์ CSV
        financials_df.to_csv(filename, index=True)
        print(f"ข้อมูลการเงินของ {ticker} ถูกบันทึกในไฟล์ {filename}")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการบันทึกไฟล์ของ {ticker}: {e}")

# ฟังก์ชันดึงข้อมูลงบการเงินของหุ้นใน SET
def fetch_financials_for_set_yfinance():
    stocks = get_set_stocks()  # ฟังก์ชันที่ดึงรายชื่อหุ้นใน SET (คุณต้องกำหนดเอง)

    for ticker in stocks:
        print(f"Fetching financial data for: {ticker}")
        financials_df = get_financials_yfinance(ticker)
        if financials_df is not None:
            financials_df = financials_df.T  # ทรานสโพสข้อมูลเพื่อให้อ่านง่ายขึ้น
            save_financials_to_csv(financials_df, ticker)  # บันทึกเป็นไฟล์แยก

if __name__ == "__main__":
    fetch_financials_for_set_yfinance()

