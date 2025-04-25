import os
import sys
import time
import datetime
import pandas as pd
import subprocess
import threading

# ✅ แพตช์ undetected_chromedriver ถ้ามีการใช้งาน
try:
    import undetected_chromedriver as uc

    def patched_del(self):
        pass

    uc.Chrome.__del__ = patched_del
except ImportError:
    pass  # ไม่เป็นไรถ้ายังไม่ได้ติดตั้ง uc

# ✅ รายชื่อสคริปต์ที่จะรันต่อเนื่อง
get_news_us = ["./usa/getnewsUSA.py"]
ticker_news_us = ["./usa/ticker_news_USA.py"]
finbert_news_us = ["./usa/Finbert_News_USA.py"]
news_to_database_us = ["./usa/news_to_database_US.py"]

get_news_th = ["./thai/getnewsThai.py"]
ticker_news_th = ["./thai/ticker_news_Thai.py"]
finbert_news_th = ["./thai/Finbert_News_Thai.py"]
news_to_database_th = ["./thai/news_to_database_TH.py"]


get_stock_us = ["./usa/GetdataAmericanStock.py"]
get_financial_us = ["./usa/GetFinancialUSA.py"]
daily_sentiment_us = ["./usa/dailysentiment_USA.py"]
combine_all_us = ["./usa/combineall_USA.py"]
stock_to_database_us = ["./usa/stock_to_database_USA.py"]
daily_sentiment_th = ["./thai/dailysentiment_Thai.py"]
get_financial_th = ["./thai/GetFinancialThai.py"]
get_stock_th = ["./thai/GetdataThaiStocks.py"]
combine_all_thai = ["./thai/combineall_Thai.py"]
stock_to_database_th = ["./thai/stock_to_database_Thai.py"]

def update_yfinance():
    print("🔄 กำลังอัปเดต `yfinance` เป็นเวอร์ชันล่าสุด...")
    subprocess.run(["pip", "install", "--upgrade", "yfinance"], check=True)
    print("✅ อัปเดต `yfinance` สำเร็จ")


def run_scripts(scripts, group_name):
    print(f"\n▶️ Running {group_name}...")
    # อัปเดต yfinance ก่อนรันสคริปต์
    for script in scripts:
        print(f"  → Running: {script}")
        subprocess.run([sys.executable, script], check=True)
    print(f"✅ Done: {group_name}")

def clear_stock_csv():
    folder_paths = ["./usa/News", "./usa", "./thai/News", "./thai"]
    for folder in folder_paths:
        if os.path.exists(folder) and os.path.isdir(folder):
            for file in os.listdir(folder):
                if file.endswith(".csv"):
                    file_path = os.path.join(folder, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print(f"🗑 ลบไฟล์: {file_path}")
                    except Exception as e:
                        print(f"⚠️ ไม่สามารถลบไฟล์ {file_path}: {e}")
    print("✅ ลบไฟล์ .csv เรียบร้อยแล้ว")

def up_to_db():
    if time.now().hour == 20 and time.now().minute == 0:
        print("update to database stock...")
        update_yfinance()
        run_scripts(get_stock_us, "Get Stock US")
        run_scripts(get_financial_us, "Get Financial US")
        run_scripts(daily_sentiment_us, "Daily Sentiment US")
        run_scripts(combine_all_us, "Combine All US")
        run_scripts(stock_to_database_us, "Stock to Database")
        print("✅ อัปเดตข้อมูลในฐานข้อมูลเรียบร้อยแล้ว US")
    if time.now().hour == 9 and time.now().minute == 30:
        print("update to database stock...")
        update_yfinance()
        run_scripts(get_stock_th, "Get Stock TH")
        run_scripts(get_financial_th, "Get Financial TH") 
        run_scripts(daily_sentiment_th, "Daily Sentiment TH")
        run_scripts(combine_all_thai, "Combine All TH")   
        run_scripts(stock_to_database_th, "Stock to Database")
        print("✅ อัปเดตข้อมูลในฐานข้อมูลเรียบร้อยแล้ว TH")

def run_every_2_hours():
    last_run_hour = None
    user_input = []
    def ask_input():
        user_input.append(input("Enter mode (1 or 2): "))  
    print("select mode ")
    print("1. auto run")
    print("2. manual run")
    # สร้าง thread สำหรับ input
    input_thread = threading.Thread(target=ask_input)
    input_thread.start()
    # รอ 10 วินาที
    input_thread.join(timeout=10)
    # ถ้าผู้ใช้พิมพ์ทันในเวลา 10 วิ
    mode = user_input[0] if user_input else "1"
    print("Selected mode:", mode)

    if mode == "1":
        while True:
            now = datetime.datetime.now()
            current_hour = now.hour

            if current_hour % 2 == 0 and current_hour != last_run_hour and now.minute == 0:
                print(f"🕒 Running script at {now.strftime('%H:%M:%S')}")
                last_run_hour = current_hour
                try:
                    run_scripts(get_news_us, "Get News")
                    run_scripts(ticker_news_us, "Match Tickers")
                    run_scripts(finbert_news_us, "FinBERT Sentiment")
                    run_scripts(news_to_database_us, "News to Database")
                    # ถ้าต้องการรันฝั่งไทยด้วย ให้ปลดคอมเมนต์ด้านล่าง
                    run_scripts(get_news_th, "Get News TH")
                    run_scripts(ticker_news_th, "Match Tickers TH")
                    run_scripts(finbert_news_th, "FinBERT Sentiment TH")
                    run_scripts(news_to_database_th, "News to Database TH")
                    print("🎉 All scripts completed successfully.")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Script failed: {e}")
                except Exception as e:
                    print(f"❌ Unexpected error: {e}")
                if time.now().hour == 20 and time.now().minute == 0:
                    up_to_db()
                if time.now().hour == 9 and time.now().minute == 30:
                    up_to_db()
                # ลบไฟล์ CSV เฉพาะตอนเที่ยงคืน
                if now.hour == 0:
                    print("🗑️ Clearing stock CSV files...")
                    clear_stock_csv()
                
            else:
                print(f"⏳ Waiting for the next run at {now.strftime('%H:%M:%S')}...")
                time.sleep(60)  # ตรวจสอบทุกนาทีเผื่อเวลาผ่านไปยังชั่วโมงใหม่
    else:
        try:
            run_scripts(get_news_us, "Get News")
            run_scripts(ticker_news_us, "Match Tickers")
            run_scripts(finbert_news_us, "FinBERT Sentiment")
            run_scripts(news_to_database_us, "News to Database")
            # ถ้าต้องการรันฝั่งไทยด้วย ให้ปลดคอมเมนต์ด้านล่าง
            run_scripts(get_news_th, "Get News TH")
            run_scripts(ticker_news_th, "Match Tickers TH")
            run_scripts(finbert_news_th, "FinBERT Sentiment TH")
            run_scripts(news_to_database_th, "News to Database TH")
            print("🎉 All scripts completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Script failed: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            
def main():
    run_every_2_hours()

if __name__ == "__main__":
    main()
