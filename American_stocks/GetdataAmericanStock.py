import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import yfinance as yf
import logging

# ตั้งค่าการบันทึก log
logging.basicConfig(
    filename='stock_data_fetch.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_all_us_tickers():
    """ดึงรายชื่อหุ้นจาก NASDAQ"""
    nasdaq_url = "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed-symbols.csv"
    try:
        logging.info("Fetching NASDAQ tickers...")
        nasdaq_data = pd.read_csv(nasdaq_url)
        tickers = nasdaq_data['Symbol'].tolist()
        logging.info(f"Total NASDAQ tickers fetched: {len(tickers)}")
        print(f"Total NASDAQ tickers fetched: {len(tickers)}")
        return tickers
    except Exception as e:
        logging.error(f"Error fetching NASDAQ tickers: {e}")
        return []

def fetch_stock_data(symbol, start_date=None, end_date=None):
    """ดึงข้อมูลหุ้นจาก Yahoo Finance"""
    try:
        ticker = f"{symbol}"
        logging.info(f"Fetching data for {ticker}...")

        # ตรวจสอบว่า start_date และ end_date ถูกกำหนดหรือไม่
        if start_date is None or end_date is None:
            start_date = pd.Timestamp("2014-01-01")
            end_date = pd.Timestamp("2024-11-01")
        # ดึงข้อมูลจาก yfinance
        stock = yf.Ticker(ticker)
        history = stock.history(start=start_date, end=end_date)

        if history.empty:
            logging.warning(f"No data found for {symbol}.")
            print(f"No data found for {symbol}.")
            return pd.DataFrame()


        # เพิ่มข้อมูลการเปลี่ยนแปลงและการเปลี่ยนแปลงเป็น %
        history['Change'] = history['Close'] - history['Open']
        history['Change (%)'] = (history['Change'] / history['Open']) * 100

        stock_data = []
        for date, row in history.iterrows():
            stock_data.append({
                "Stock": symbol,
                "Date": date,
                "Open": row['Open'],
                "Close": row['Close'],
                "High": row['High'],
                "Low": row['Low'],
                "Volume": row['Volume']
            })

        logging.info(f"Data for {symbol} fetched successfully.")
        print(f"Data for {symbol} fetched successfully.")
        return pd.DataFrame(stock_data)

    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def fetch_stock_data_with_retry(symbol, start_date=None, end_date=None, retries=3):
    """ดึงข้อมูลหุ้นพร้อมการ retry"""
    for attempt in range(retries):
        result = fetch_stock_data(symbol, start_date, end_date)
        if not result.empty:
            return result
        else:
            logging.warning(f"Retrying {symbol} (Attempt {attempt + 1})...")  # Log retry
            print(f"Retrying {symbol} (Attempt {attempt + 1})...")
            time.sleep(random.uniform(1, 3))  # หน่วงเวลาแบบสุ่มเพื่อหลีกเลี่ยงการถูกจำกัด

    logging.error(f"Failed to fetch data for {symbol} after {retries} attempts.")  # Log final failure
    return pd.DataFrame()

def fetch_new_stock_data(stock_list, existing_data_file):
    """ดึงข้อมูลใหม่สำหรับหุ้นที่ยังไม่มีในไฟล์"""
    # ตรวจสอบว่าไฟล์ข้อมูลเก่ามีอยู่หรือไม่
    if os.path.exists(existing_data_file):
        # อ่านข้อมูลที่มีอยู่แล้ว
        existing_data = pd.read_csv(existing_data_file)
        logging.info(f"Existing data loaded from {existing_data_file}.")
        print(f"Existing data loaded from {existing_data_file}.")
    else:
        existing_data = pd.DataFrame()
    
    new_data = []
    failed_stocks = []  # List to track failed stocks

    for symbol in stock_list:
        try:
            # สร้าง ticker สำหรับหุ้นในตลาด SET
            ticker = f"{symbol}.BK"
            logging.info(f"Fetching new data for {ticker}...")
            print(f"Fetching new data for {ticker}...")

            # ตรวจสอบวันที่ล่าสุดที่มีข้อมูล
            if not existing_data.empty and symbol in existing_data['Stock'].unique():
                last_date = pd.to_datetime(existing_data[existing_data['Stock'] == symbol]['Date']).max()
                logging.info(f"Last date for {symbol} is {last_date}. Fetching new data...")
                stock = yf.Ticker(ticker)
                history = stock.history(start=str(last_date + pd.Timedelta(days=1)))
            else:
                logging.info(f"No existing data for {symbol}. Fetching all data...")
                print(f"No existing data for {symbol}. Fetching all data...")
                stock = yf.Ticker(ticker)
                history = stock.history(period="max")

            if not history.empty:
                history['Change'] = history['Close'] - history['Open']
                history['Change (%)'] = (history['Change'] / history['Open']) * 100

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

                logging.info(f"New data fetched successfully for {symbol}.")
                print(f"New data fetched successfully for {symbol}.")
            else:
                logging.warning(f"No new data for {symbol}.")
                print(f"No new data for {symbol}.")
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            failed_stocks.append(symbol)  # Add the failed stock symbol to the list
            print(f"Error fetching data for {symbol}: {e}")

    new_data_df = pd.DataFrame(new_data)

    if not new_data_df.empty:
        combined_data = pd.concat([existing_data, new_data_df], ignore_index=True).drop_duplicates(subset=["Stock", "Date"])
        combined_data.to_csv(existing_data_file, index=False)
        logging.info(f"Updated data saved to {existing_data_file}.")
        print(f"Updated data saved to {existing_data_file}.")
    else:
        logging.info("No new data to add.")
        print("No new data to add.")
    
    # Create a report of failed stocks
    if failed_stocks:
        failed_stocks_df = pd.DataFrame(failed_stocks, columns=["Failed Stock"])
        failed_stocks_df.to_csv("failed_stocks_report.csv", index=False)
        logging.info("Failed stocks report saved to 'failed_stocks_report.csv'.")
        print("Failed stocks report saved to 'failed_stocks_report.csv'.")

if __name__ == "__main__":
    # ชื่อโฟลเดอร์ที่ต้องการสร้าง
    folder_name = "stock_America"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        logging.info(f"Folder '{folder_name}' created.")

    # ดึงรายชื่อหุ้น
    stock_list = fetch_all_us_tickers()

    # ไฟล์ CSV ที่เก็บข้อมูลหุ้นทั้งหมด
    output_file_csv = os.path.join(folder_name, "NDQ_Stock_History_10Y.csv")

    # ตรวจสอบว่าไฟล์ข้อมูลเก่ามีอยู่หรือไม่
    if os.path.exists(output_file_csv):
        # โหลดข้อมูลเก่าจากไฟล์
        all_data = pd.read_csv(output_file_csv)
        logging.info(f"Loaded existing data from {output_file_csv}.")
        print(f"Loaded existing data from {output_file_csv}.")
    else:
        all_data = pd.DataFrame()

    # ตรวจสอบว่ามีหุ้นในรายการหรือไม่
    if stock_list:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_stock_data_with_retry, symbol): symbol for symbol in stock_list}

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        df['Stock'] = symbol
                        all_data = pd.concat([all_data, df], ignore_index=True)
                        logging.info(f"Data for {symbol} fetched and added successfully.")
                        print(f"Data for {symbol} fetched and added successfully.")
                except Exception as e:
                    logging.error(f"Error fetching data for {symbol}: {e}")
                    print(f"Error fetching data for {symbol}: {e}")

        # Save the combined data
        all_data.to_csv(output_file_csv, index=False)
        logging.info(f"Combined data saved to {output_file_csv}.")
        print(f"Combined data saved to {output_file_csv}.")
    else:
        logging.error("No stock list fetched.")
        print("No stock list fetched.")
