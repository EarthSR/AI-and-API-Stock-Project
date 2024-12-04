import os
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
        return tickers
    except Exception as e:
        logging.error(f"Error fetching NASDAQ tickers: {e}")
        return []

def is_valid_ticker(ticker):
    """ตรวจสอบว่าหุ้นมีอยู่ใน Yahoo Finance หรือไม่"""
    try:
        yf.Ticker(ticker).info
        return True
    except Exception:
        return False

def fetch_stock_data(symbol, start_date=None, end_date=None):
    """ดึงข้อมูลหุ้นแต่ละตัว"""
    stock_data = []
    try:
        ticker = yf.Ticker(symbol)
        logging.info(f"Fetching data for {symbol}...")

        end_date = pd.to_datetime('today')
        start_date = start_date or (end_date - pd.DateOffset(years=10))

        history = ticker.history(period='1d', start=start_date, end=end_date)

        if history.empty:
            logging.warning(f"No data found for {symbol}. Possibly delisted.")
            return stock_data

        history['Change'] = history['Close'] - history['Open']
        history['Change (%)'] = (history['Change'] / history['Open']) * 100

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
        logging.info(f"Data for {symbol} fetched successfully.")
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")

    return stock_data

def fetch_with_retries(symbol, retries=3, delay=2, backoff_factor=2):
    """ดึงข้อมูลหุ้นพร้อม Retry และจัดการ 429"""
    for attempt in range(retries):
        try:
            return fetch_stock_data(symbol)
        except Exception as e:
            if "429" in str(e):  # ตรวจสอบข้อผิดพลาด 429
                wait_time = delay * (backoff_factor ** attempt)
                logging.warning(f"Rate limited for {symbol}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"Attempt {attempt + 1} failed for {symbol}: {e}")
    logging.error(f"All {retries} attempts failed for {symbol}.")
    return []


def fetch_all_stock_data(stock_list, start_date=None, end_date=None):
    """ดึงข้อมูลหุ้นหลายตัวพร้อมกัน"""
    stock_data = []
    max_threads = 5  # จำกัดจำนวนเธรด
    with ThreadPoolExecutor(max_threads) as executor:
        futures = {executor.submit(fetch_with_retries, symbol, start_date, end_date): symbol for symbol in stock_list}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                data = future.result()
                stock_data.extend(data)

                # บันทึกข้อมูลที่ดึงมาได้บางส่วน
                if len(data) > 0:
                    pd.DataFrame(data).to_csv('partial_results.csv', mode='a', index=False, header=False)
            except Exception as e:
                logging.error(f"Error fetching data for {symbol}: {e}")
    return stock_data


if __name__ == "__main__":
    folder_name = "stock_American"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        logging.info(f"Folder '{folder_name}' created.")

    stock_list = fetch_all_us_tickers()

    if stock_list:
        # กรองหุ้นที่ไม่ถูกต้อง
        logging.info("Validating tickers...")
        stock_list = [ticker for ticker in stock_list if is_valid_ticker(ticker)]
        logging.info(f"Valid tickers: {len(stock_list)}")

        output_file_csv = os.path.join(folder_name, "NASDAQ_Stock_History_10Y.csv")

        if os.path.exists(output_file_csv):
            all_data = pd.read_csv(output_file_csv)
            logging.info(f"Loaded existing data from {output_file_csv}.")
        else:
            all_data = pd.DataFrame()

        stock_data = fetch_all_stock_data(stock_list)

        new_data_df = pd.DataFrame(stock_data)

        if not new_data_df.empty:
            all_data = pd.concat([all_data, new_data_df], ignore_index=True).drop_duplicates(subset=["Stock", "Date"])
            all_data.to_csv(output_file_csv, index=False)
            logging.info(f"All stock data saved to {output_file_csv}.")
        else:
            logging.info("No new data to save.")
    else:
        logging.error("No stock symbols found. Exiting.")
