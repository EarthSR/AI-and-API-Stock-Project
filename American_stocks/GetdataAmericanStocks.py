import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import yfinance as yf

def fetch_single_stock_data(ticker):
    """
    ดึงข้อมูลประวัติย้อนหลังสำหรับหุ้นตัวเดียว
    """
    try:
        print(f"Fetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        hist = stock.history(period="max")
        if not hist.empty:
            hist['Ticker'] = ticker
            return {"data": hist, "status": "success"}
        else:
            print(f"No data for {ticker}, skipping...")
            return {"data": None, "status": "no_data"}  # ไม่มีข้อมูล ข้ามไป
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return {"data": None, "status": "failed"}  # เมินกรณีล้มเหลว

def fetch_historical_data_combined(tickers, max_workers=20):
    """
    ดึงข้อมูลประวัติย้อนหลังของหุ้นในตลาด NASDAQ และประมวลผลเฉพาะ Success
    """
    success_data = pd.DataFrame()  # DataFrame ว่างสำหรับหุ้นที่สำเร็จ

    start_time = time.perf_counter()  # เริ่มจับเวลา

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(fetch_single_stock_data, ticker): ticker for ticker in tickers}
        total_tickers = len(tickers)

        for i, future in enumerate(as_completed(future_to_ticker)):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                if result["status"] == "success" and result["data"] is not None:
                    success_data = pd.concat([success_data, result["data"]], ignore_index=True)
            except Exception as e:
                print(f"Unexpected error processing {ticker}: {e}")

            # คำนวณเวลาที่ใช้และเวลาที่เหลือ
            elapsed_time = time.perf_counter() - start_time
            average_time_per_ticker = elapsed_time / (i + 1)
            remaining_time = average_time_per_ticker * (total_tickers - (i + 1))

            print(f"[{i + 1}/{total_tickers}] Ticker {ticker} processed. "
                  f"Elapsed time: {elapsed_time:.2f}s, Estimated remaining time: {remaining_time:.2f}s")

    # บันทึกข้อมูลสำเร็จ
    if not success_data.empty:
        success_filename = "US_Historical.csv"
        success_data.to_csv(success_filename, index=False)
        print(f"Success data saved to {success_filename}")

    # สรุปผล
    total_success = len(success_data['Ticker'].unique()) if not success_data.empty else 0
    success_rate = (total_success / total_tickers) * 100

    print(f"Total tickers processed: {total_tickers}")
    print(f"Successful: {total_success} ({success_rate:.2f}%)")
    print("Failed tickers ignored.")

def fetch_all_us_tickers():
    """
    ดึงรายชื่อหุ้นทั้งหมดจาก NASDAQ
    """
    nasdaq_url = "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed-symbols.csv"

    try:
        print("Fetching NASDAQ tickers...")
        nasdaq_data = pd.read_csv(nasdaq_url)
        tickers = nasdaq_data['Symbol'].tolist()
        print(f"Total NASDAQ tickers fetched: {len(tickers)}")
        return tickers
    except Exception as e:
        print(f"Error fetching NASDAQ tickers: {e}")
        return []

if __name__ == "__main__":
    # ดึงรายชื่อหุ้นจาก NASDAQ
    tickers = fetch_all_us_tickers()
    if tickers:
        fetch_historical_data_combined(tickers, max_workers=20)  # ใช้ 20 threads
    else:
        print("No tickers found to process.")
