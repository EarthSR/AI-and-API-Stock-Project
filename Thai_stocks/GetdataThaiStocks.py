import requests
import pandas as pd
import os
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
import yfinance as yf
from datetime import datetime, timedelta



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
    

# ฟังก์ชันดึงข้อมูลหุ้นตามรายชื่อ พร้อมช่วงเวลา
def fetch_stock_data(stock_list, start_date=None, end_date=None):
    stock_data = []
    
    start_date = '2018-01-01'  # กำหนดวันที่เริ่มต้น
    end_date = '2025-02-03'    # กำหนดวันที่สิ้นสุด


    for symbol in stock_list:
        try:
            # สร้าง ticker สำหรับหุ้นในตลาด SET
            ticker = f"{symbol}.BK"
            print(f"Fetching data for {ticker}...")

            # ดึงข้อมูลจาก yfinance
            stock = yf.Ticker(ticker)

            # กำหนดช่วงเวลา
            if start_date:
                history = stock.history(start=start_date, end=end_date)
            else:
                history = stock.history(period="max")  # ดึงข้อมูลทั้งหมด

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

                print(f"Data for {symbol} fetched successfully.")
            else:
                print(f"No data found for {symbol}.")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    return pd.DataFrame(stock_data)


# ฟังก์ชันดึงข้อมูลหุ้นเพิ่มเฉพาะข้อมูลใหม่
def fetch_new_stock_data(stock_list, existing_data_file):
    # ตรวจสอบว่ามีไฟล์ข้อมูลเก่าหรือไม่
    if os.path.exists(existing_data_file):
        # อ่านข้อมูลที่มีอยู่แล้ว
        existing_data = pd.read_csv(existing_data_file)
        print(f"Existing data loaded from {existing_data_file}.")
    else:
        # ถ้าไม่มีไฟล์เก่า ให้สร้าง DataFrame เปล่า
        existing_data = pd.DataFrame()

    new_data = []

    for symbol in stock_list:
        try:
            # สร้าง ticker สำหรับหุ้นในตลาด SET
            ticker = f"{symbol}.BK"
            print(f"Fetching new data for {ticker}...")

            # ตรวจสอบวันที่ล่าสุดที่มีข้อมูล
            if not existing_data.empty and symbol in existing_data['Stock'].unique():
                last_date = pd.to_datetime(existing_data[existing_data['Stock'] == symbol]['Date']).max()
                print(f"Last date for {symbol} is {last_date}. Fetching new data...")
                stock = yf.Ticker(ticker)
                history = stock.history(start=str(last_date + pd.Timedelta(days=1)))
            else:
                print(f"No existing data for {symbol}. Fetching all data...")
                stock = yf.Ticker(ticker)
                history = stock.history(period="max")

            if not history.empty:
                # เพิ่มข้อมูลการเปลี่ยนแปลงและการเปลี่ยนแปลงเป็น %
                history['Change'] = history['Close'] - history['Open']
                history['Change (%)'] = (history['Change'] / history['Open']) * 100

                # บันทึกข้อมูลใหม่
                for date, row in history.iterrows():
                    new_data.append({
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
                print(f"New data fetched successfully for {symbol}.")
            else:
                print(f"No new data for {symbol}.")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    # แปลงข้อมูลใหม่เป็น DataFrame
    new_data_df = pd.DataFrame(new_data)
     # รวมข้อมูลเก่าและใหม่
    if not new_data_df.empty:
        combined_data = pd.concat([existing_data, new_data_df], ignore_index=True).drop_duplicates(subset=["Stock", "Date"])
        combined_data.to_csv(existing_data_file, index=False)
        print(f"Updated data saved to {existing_data_file}.")
    else:
        print("No new data to add.")


if __name__ == "__main__":
    # ชื่อโฟลเดอร์ที่ต้องการสร้าง
    folder_name = "stock_thai"

    # ตรวจสอบและสร้างโฟลเดอร์หากยังไม่มี
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")

    # ดึงรายชื่อหุ้นจาก SET
    # stock_list = ['ADVANC', 'INTUCH', 'TRUE', 'DITTO', 'DIF','INSET', 'JMART', 'INET', 'JAS', 'HUMAN']
    stock_list = ['^SET']
    # ไฟล์ CSV ที่เก็บข้อมูลหุ้นทั้งหมด
    output_file_csv = os.path.join(folder_name, "SET50_Stock_History.csv")

    # ตรวจสอบว่าไฟล์ข้อมูลเก่ามีอยู่หรือไม่
    if os.path.exists(output_file_csv):
        # โหลดข้อมูลเก่าจากไฟล์
        all_data = pd.read_csv(output_file_csv)
        print(f"Loaded existing data from {output_file_csv}.")
    else:
        # สร้าง DataFrame เปล่าสำหรับข้อมูลใหม่
        all_data = pd.DataFrame()

    # ตรวจสอบว่ามีหุ้นในรายการหรือไม่
    if stock_list:
        for symbol in stock_list:
            try:
                # ตรวจสอบวันที่ล่าสุดของหุ้นในไฟล์
                if not all_data.empty and symbol in all_data['Stock'].unique():
                    last_date = pd.to_datetime(all_data[all_data['Stock'] == symbol]['Date']).max()
                    print(f"Fetching new data for {symbol} since {last_date}...")
                    df = fetch_stock_data([symbol], start_date=str(last_date + pd.Timedelta(days=1)))
                else:
                    print(f"Fetching all data for {symbol}...")
                    df = fetch_stock_data([symbol])  # ดึงข้อมูลทั้งหมด

                if not df.empty:
                    # เพิ่มคอลัมน์ชื่อหุ้นใน DataFrame
                    df['Stock'] = symbol
                    # รวมข้อมูลใหม่กับข้อมูลเก่า
                    all_data = pd.concat([all_data, df], ignore_index=True)
                    print(f"Data for {symbol} fetched successfully.")
                else:
                    print(f"No new data fetched for {symbol}.")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        # บันทึกข้อมูลที่อัปเดตแล้วลงไฟล์ CSV
        all_data = all_data.drop_duplicates(subset=["Stock", "Date"])  # ลบข้อมูลซ้ำซ้อน
        all_data.to_csv(output_file_csv, index=False)
        print(f"All stock data saved to {output_file_csv}.")
    else:
        print("No stock symbols found. Exiting.")



