import os
import sys
import time
import datetime
import pandas as pd
import subprocess

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

def update_yfinance():
    print("🔄 กำลังอัปเดต `yfinance` เป็นเวอร์ชันล่าสุด...")
    subprocess.run(["pip", "install", "--upgrade", "yfinance"], check=True)
    print("✅ อัปเดต `yfinance` สำเร็จ")


def run_scripts(scripts, group_name):
    print(f"\n▶️ Running {group_name}...")
    # อัปเดต yfinance ก่อนรันสคริปต์
    update_yfinance()
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





def run_every_2_hours():
    last_run_hour = None  
    print("select mode ")
    print("1. auto run")
    print("2. manual run")
    mode = input("Enter mode (1 or 2): ")
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
