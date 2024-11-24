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
    

# ฟังก์ชันดึงข้อมูลหุ้นตามรายชื่อ
def fetch_stock_data(stock_list):
    stock_data = []

    for symbol in stock_list:
        try:
            # สร้าง ticker สำหรับหุ้นในตลาด SET
            ticker = f"{symbol}.BK"  # ".BK" ใช้สำหรับหุ้นในตลาด SET
            print(f"Fetching data for {ticker} from the beginning...")

            # ดึงข้อมูลจาก yfinance
            stock = yf.Ticker(ticker)
            history = stock.history(period="max")

            if not history.empty:
                # เพิ่มข้อมูลการเปลี่ยนแปลงและการเปลี่ยนแปลงเป็น %
                history['Change'] = history['Close'] - history['Open']
                history['Change (%)'] = (history['Change'] / history['Open']) * 100

                # บันทึกข้อมูลสำหรับหุ้นแต่ละตัว
                for date, row in history.iterrows():
                    stock_data.append({
                        "Stock": symbol,
                        "Date": date,
                        "Open": row['Open'],
                        "Close": row['Close'],
                        "High": row['High'],
                        "Low": row['Low'],
                        "Volume": row['Volume'],
                        "Change": row['Change'],
                        "Change (%)": row['Change (%)']
                    })

                print(f"Data fetched successfully for {symbol}.")
            else:
                print(f"No data found for {symbol}.")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    return pd.DataFrame(stock_data)


if __name__ == "__main__":
    # ชื่อโฟลเดอร์ที่ต้องการสร้าง
    folder_name = "stock_thai"

    # ตรวจสอบและสร้างโฟลเดอร์หากยังไม่มี
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")

    # ดึงรายชื่อหุ้นจาก SET
    stock_list = get_set_stocks()

    # DataFrame ว่างเพื่อเก็บข้อมูลทั้งหมด
    all_data = pd.DataFrame()

    # ตรวจสอบว่ามีหุ้นในรายการหรือไม่
    if stock_list:
        # วนลูปเพื่อดึงข้อมูลหุ้นแต่ละตัว
        for symbol in stock_list:
            df = fetch_stock_data([symbol])  # ดึงข้อมูลสำหรับหุ้นแต่ละตัว

            if not df.empty:
                # เพิ่มคอลัมน์ชื่อหุ้นใน DataFrame
                df['Stock'] = symbol
                # รวมข้อมูลใน DataFrame หลัก
                all_data = pd.concat([all_data, df], ignore_index=True)
                print(f"Data for {symbol} fetched successfully.")
            else:
                print(f"No data fetched for {symbol}.")
        
        # บันทึกข้อมูลทั้งหมดลงไฟล์ CSV
        if not all_data.empty:
            output_file_csv = os.path.join(folder_name, "SET_Stock_History_All.csv")
            all_data.to_csv(output_file_csv, index=False)
            print(f"All stock data saved to {output_file_csv}.")
        else:
            print("No stock data to save. Exiting.")
    else:
        print("No stock symbols found. Exiting.")


