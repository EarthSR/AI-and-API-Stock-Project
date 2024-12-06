import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import yfinance as yf

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
    
    # ฟังก์ชันดึงงบการเงินจาก Yahoo Finance
def get_financials_yfinance(ticker):
    try:
        stock = yf.Ticker(ticker)  # เพิ่ม .BK สำหรับหุ้นไทย
        financials = stock.financials  # ดึงงบการเงิน
        return financials
    except Exception as e:
        print(f"Error fetching financials for {ticker}: {e}")
        return None

# ฟังก์ชันรวมข้อมูลการเงินของหุ้นทั้งหมด
def fetch_and_save_all_financials():
    stocks = fetch_all_us_tickers()  # ดึงรายชื่อหุ้น
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