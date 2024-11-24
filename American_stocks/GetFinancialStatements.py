import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import yfinance as yf

def fetch_financial_data(ticker):
    """
    ดึงงบการเงินของบริษัทสำหรับ Ticker เดียว
    """
    try:
        print(f"Fetching financial data for {ticker}...")
        stock = yf.Ticker(ticker)

        # ดึงข้อมูลงบการเงิน
        balance_sheet = stock.balance_sheet
        income_statement = stock.financials
        cash_flow = stock.cashflow

        # เพิ่มคอลัมน์ Ticker ในแต่ละงบ
        for df in [balance_sheet, income_statement, cash_flow]:
            if not df.empty:
                df['Ticker'] = ticker

        # ตรวจสอบว่ามีข้อมูลในงบการเงินหรือไม่
        if not balance_sheet.empty or not income_statement.empty or not cash_flow.empty:
            return {
                "balance_sheet": balance_sheet,
                "income_statement": income_statement,
                "cash_flow": cash_flow,
                "status": "success"
            }
        else:
            print(f"No financial data for {ticker}, skipping...")
            return {"status": "no_data"}

    except Exception as e:
        print(f"Error fetching financial data for {ticker}: {e}")
        return {"status": f"failed: {e}"}

def fetch_all_financial_data(tickers, max_workers=20):
    """
    ดึงงบการเงินของหุ้นทุกตัวในตลาด NASDAQ และบันทึกเฉพาะ Success
    """
    success_data = {
        "balance_sheet": pd.DataFrame(),
        "income_statement": pd.DataFrame(),
        "cash_flow": pd.DataFrame()
    }

    start_time = time.perf_counter()  # เริ่มจับเวลา

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(fetch_financial_data, ticker): ticker for ticker in tickers}
        total_tickers = len(tickers)

        for i, future in enumerate(as_completed(future_to_ticker)):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                if data["status"] == "success":
                    for key in success_data.keys():
                        if key in data and data[key] is not None:
                            success_data[key] = pd.concat([success_data[key], data[key]], ignore_index=True)
                else:
                    # หุ้นที่ไม่มีข้อมูลหรือ Fail จะถูกเมิน
                    print(f"Skipping {ticker} due to {data['status']}.")
            except Exception as e:
                print(f"Unexpected error processing {ticker}: {e}")

            # คำนวณเวลาที่ใช้และเวลาที่เหลือ
            elapsed_time = time.perf_counter() - start_time
            average_time_per_ticker = elapsed_time / (i + 1)
            remaining_time = average_time_per_ticker * (total_tickers - (i + 1))

            print(f"[{i + 1}/{total_tickers}] Ticker {ticker} processed. "
                  f"Elapsed time: {elapsed_time:.2f}s, Estimated remaining time: {remaining_time:.2f}s")

    # บันทึกข้อมูลสำเร็จ
    for key, df in success_data.items():
        if not df.empty:
            filename = f"US_Financial_{key.capitalize().replace('_', '')}_Success.csv"
            df.to_csv(filename, index=False)
            print(f"{key.capitalize()} Success data saved to {filename}")

    # สรุปผล
    total_success = len(success_data["balance_sheet"]['Ticker'].unique()) if not success_data["balance_sheet"].empty else 0
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
        fetch_all_financial_data(tickers, max_workers=20)  # ใช้ 20 threads
    else:
        print("No tickers found to process.")
