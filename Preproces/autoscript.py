import os
import sys
import time
import datetime
import pandas as pd
import subprocess
import threading
import json
import logging
import platform
import pandas_market_calendars as mcal
from plyer import notification  # เพิ่มการนำเข้า plyer สำหรับแจ้งเตือน

# ✅ ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('script_log.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
def safe_notify(title, message, app_name, timeout):
    max_length = 256
    if len(message) > max_length:
        message = message[:max_length-3] + "..."  # ตัดข้อความและเพิ่ม ...
        logger.warning(f"ข้อความแจ้งเตือนยาวเกิน {max_length} ตัวอักษร ตัดเหลือ: {message}")
    notification.notify(  # เปลี่ยนจาก safe_notify เป็น notification.notify
        title=title,
        message=message,
        app_name=app_name,
        timeout=timeout
    )
# ✅ แพตช์ undetected_chromedriver ถ้ามีการใช้งาน
try:
    import undetected_chromedriver as uc

    def patched_del(self):
        pass

    uc.Chrome.__del__ = patched_del
    logger.info("✅ แพตช์ undetected_chromedriver สำเร็จ")
    safe_notify(
        title="System Info",
        message="แพตช์ undetected_chromedriver สำเร็จ",
        app_name="Stock Data Updater",
        timeout=10
    )
except ImportError:
    logger.info("ℹ️ ไม่พบ undetected_chromedriver - ข้าม")
    safe_notify(
        title="System Info",
        message="ไม่พบ undetected_chromedriver - ข้าม",
        app_name="Stock Data Updater",
        timeout=10
    )

# ✅ รายชื่อสคริปต์
SCRIPTS = {
    "news_us": {
        "get_news": ["./usa/getnewsUSA.py"],
        "ticker_news": ["./usa/ticker_news_USA.py"],
        "finbert_news": ["./usa/Finbert_News_USA.py"],
        "news_to_database": ["./usa/news_to_database_US.py"]
    },
    "news_th": {
        "get_news": ["./thai/getnewsThai.py"],
        "ticker_news": ["./thai/ticker_news_Thai.py"],
        "finbert_news": ["./thai/Finbert_News_Thai.py"],
        "news_to_database": ["./thai/news_to_database_TH.py"]
    },
    "stock_us": {
        "get_stock": ["./usa/GetdataAmericanStock.py"],
        "get_financial": ["./usa/GetFinancialUSA.py"],
        "daily_sentiment": ["./usa/dailysentiment_USA.py"],
        "combine_all": ["./usa/combineall_USA.py"],
        "stock_to_database": ["./usa/stock_to_database_USA.py"]
    },
    "stock_th": {
        "get_stock": ["./thai/GetdataThaiStocks.py"],
        "get_financial": ["./thai/GetFinancialThai.py"],
        "daily_sentiment": ["./thai/dailysentiment_Thai.py"],
        "combine_all": ["./thai/combineall_Thai.py"],
        "stock_to_database": ["./thai/stock_to_database_Thai.py"]
    },
    "update stock data": {
        "update_stock_data": ["./Autotrainmodel.py"]
    }
}

def update_chrome():
    system = platform.system()

    if system == "Windows":
        logger.info("Updating Chrome on Windows...")
        subprocess.run(["powershell", "-Command", "Start-ScheduledTask -TaskName 'GoogleUpdateTaskMachineUA'"], check=True)
        safe_notify(
            title="Chrome Update",
            message="อัปเดต Chrome บน Windows สำเร็จ",
            app_name="Stock Data Updater",
            timeout=10
        )

    elif system == "Linux":
        logger.info("Updating Chrome on Linux...")
        subprocess.run(["sudo", "apt", "update"], check=True)
        subprocess.run(["sudo", "apt", "--only-upgrade", "install", "google-chrome-stable", "-y"], check=True)
        safe_notify(
            title="Chrome Update",
            message="อัปเดต Chrome บน Linux สำเร็จ",
            app_name="Stock Data Updater",
            timeout=10
        )

    elif system == "Darwin":
        logger.info("Updating Chrome on macOS...")
        subprocess.run(["brew", "update"], check=True)
        subprocess.run(["brew", "upgrade", "--cask", "google-chrome"], check=True)
        safe_notify(
            title="Chrome Update",
            message="อัปเดต Chrome บน macOS สำเร็จ",
            app_name="Stock Data Updater",
            timeout=10
        )

    else:
        logger.error("OS not supported.")
        safe_notify(
            title="Chrome Update Error",
            message="ระบบปฏิบัติการไม่รองรับ",
            app_name="Stock Data Updater",
            timeout=10
        )
        sys.exit(1)

def update_yfinance():
    """อัปเดต yfinance เป็นเวอร์ชันล่าสุด"""
    try:
        logger.info("🔄 กำลังอัปเดต yfinance...")
        result = subprocess.run(
            ["pip", "install", "--upgrade", "yfinance"], 
            capture_output=True, 
            text=True, 
            timeout=300
        )
        if result.returncode == 0:
            logger.info("✅ อัปเดต yfinance สำเร็จ")
            safe_notify(
                title="yfinance Update",
                message="อัปเดต yfinance สำเร็จ",
                app_name="Stock Data Updater",
                timeout=10
            )
            return True
        else:
            logger.error(f"❌ อัปเดต yfinance ล้มเหลว: {result.stderr}")
            safe_notify(
                title="yfinance Update Error",
                message=f"อัปเดต yfinance ล้มเหลว: {result.stderr}",
                app_name="Stock Data Updater",
                timeout=10
            )
            return False
    except subprocess.TimeoutExpired:
        logger.error("❌ อัปเดต yfinance หมดเวลา")
        safe_notify(
            title="yfinance Update Error",
            message="อัปเดต yfinance หมดเวลา",
            app_name="Stock Data Updater",
            timeout=10
        )
        return False
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการอัปเดต yfinance: {e}")
        safe_notify(
            title="yfinance Update Error",
            message=f"เกิดข้อผิดพลาดในการอัปเดต yfinance: {e}",
            app_name="Stock Data Updater",
            timeout=10
        )
        return False

def validate_script_exists(script_path):
    """ตรวจสอบว่าไฟล์สคริปต์มีอยู่จริง"""
    if not os.path.exists(script_path):
        logger.warning(f"⚠️ ไม่พบไฟล์: {script_path}")
        safe_notify(
            title="Script Error",
            message=f"ไม่พบไฟล์สคริปต์: {script_path}",
            app_name="Stock Data Updater",
            timeout=10
        )
        return False
    return True

def run_scripts(scripts, group_name, critical=False):
    print(f"\n▶️ Running {group_name}...")
    for script in scripts:
        print(f"  → Running: {script}")
        result = subprocess.run([sys.executable, script], check=False)
        if result.returncode != 0:
            print(f"❌ Script failed: {script}")
            if critical:
                return False  # Stop execution if critical
            # Optionally continue if not critical
    print(f"✅ Done: {group_name}")
    return True


def run_all_news_scripts():
    """รันสคริปต์ข่าวทั้งหมด"""
    logger.info("🗞️ เริ่มต้นการดึงข่าว...")
    safe_notify(
        title="News Update",
        message="เริ่มต้นการดึงข่าว...",
        app_name="Stock Data Updater",
        timeout=10
    )
    
    # ข่าวสหรัฐ - ถ้าล้มเหลวให้หยุด
    if not run_scripts(SCRIPTS["news_us"]["get_news"], "Get News US", critical=True):
        logger.error("❌ Failed to fetch US news - หยุดการทำงาน")
        safe_notify(
            title="News Update Error",
            message="ดึงข่าวสหรัฐล้มเหลว - หยุดการทำงาน",
            app_name="Stock Data Updater",
            timeout=10
        )
        return False
    
    # ประมวลผลข่าวสหรัฐ - ไม่ critical
    run_scripts(SCRIPTS["news_us"]["ticker_news"], "Match Tickers US", critical=False)
    run_scripts(SCRIPTS["news_us"]["finbert_news"], "FinBERT Sentiment US", critical=False)
    run_scripts(SCRIPTS["news_us"]["news_to_database"], "News to Database US", critical=False)

    # ข่าวไทย - ถ้าดึงได้ค่อยประมวลผล
    if run_scripts(SCRIPTS["news_th"]["get_news"], "Get News TH", critical=False):
        run_scripts(SCRIPTS["news_th"]["ticker_news"], "Match Tickers TH", critical=False)
        run_scripts(SCRIPTS["news_th"]["finbert_news"], "FinBERT Sentiment TH", critical=False)
        run_scripts(SCRIPTS["news_th"]["news_to_database"], "News to Database TH", critical=False)
    else:
        logger.warning("⚠️ ดึงข่าวไทยไม่สำเร็จ - ข้ามการประมวลผล")
        safe_notify(
            title="News Update Warning",
            message="ดึงข่าวไทยไม่สำเร็จ - ข้ามการประมวลผล",
            app_name="Stock Data Updater",
            timeout=10
        )
    
    return True

def clear_stock_csv():
    """ลบไฟล์ CSV ในโฟลเดอร์ที่กำหนด"""
    folder_paths = ["./usa/News"]
    deleted_count = 0
    
    for folder in folder_paths:
        if not os.path.exists(folder) or not os.path.isdir(folder):
            logger.warning(f"⚠️ ไม่พบโฟลเดอร์: {folder}")
            safe_notify(
                title="Folder Error",
                message=f"ไม่พบโฟลเดอร์: {folder}",
                app_name="Stock Data Updater",
                timeout=10
            )
            continue
            
        try:
            csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
            for file in csv_files:
                file_path = os.path.join(folder, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"🗑 ลบไฟล์: {file_path}")
                        deleted_count += 1
                except Exception as e:
                    logger.error(f"⚠️ ไม่สามารถลบไฟล์ {file_path}: {e}")
                    safe_notify(
                        title="File Deletion Error",
                        message=f"ไม่สามารถลบไฟล์ {file_path}: {e}",
                        app_name="Stock Data Updater",
                        timeout=10
                    )
        except Exception as e:
            logger.error(f"❌ ไม่สามารถเข้าถึงโฟลเดอร์ {folder}: {e}")
            safe_notify(
                title="Folder Access Error",
                message=f"ไม่สามารถเข้าถึงโฟลเดอร์ {folder}: {e}",
                app_name="Stock Data Updater",
                timeout=10
            )
    
    logger.info(f"✅ ลบไฟล์ .csv เรียบร้อยแล้ว ({deleted_count} ไฟล์)")
    safe_notify(
        title="File Deletion",
        message=f"ลบไฟล์ .csv เรียบร้อยแล้ว ({deleted_count} ไฟล์)",
        app_name="Stock Data Updater",
        timeout=10
    )

def load_market_holidays():
    """โหลดวันหยุดตลาดจาก pandas_market_calendars หรือไฟล์"""
    try:
        holidays_file = "market_holidays_th.json"
        
        # ลองใช้ pandas_market_calendars ก่อน
        try:
            # ใช้ตลาดไทย XBKK (Stock Exchange of Thailand)
            set_calendar = mcal.get_calendar('XBKK')
            start_date = f"{datetime.now().year}-01-01"
            end_date = f"{datetime.now().year}-12-31"
            
            # สร้าง date range สำหรับปีนี้
            all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            valid_days = set_calendar.valid_days(start_date=start_date, end_date=end_date)
            
            # หาวันหยุด (วันที่ไม่ใช่วันเสาร์-อาทิตย์ แต่ตลาดปิด)
            weekdays = all_dates[all_dates.weekday < 5]  # จันทร์-ศุกร์
            holidays = weekdays.difference(valid_days)
            
            logger.info(f"✅ โหลดวันหยุดตลาดไทยจาก pandas_market_calendars: {len(holidays)} วัน")
            return [date.strftime("%Y-%m-%d") for date in holidays]
            
        except ImportError:
            logger.warning("⚠️ ไม่พบ pandas_market_calendars")
            
        # Fallback: อ่านจากไฟล์ JSON
        if os.path.exists(holidays_file):
            with open(holidays_file, "r", encoding="utf-8") as f:
                holidays_data = json.load(f).get("TH", [])
                logger.info(f"✅ โหลดวันหยุดตลาดไทยจากไฟล์: {len(holidays_data)} วัน")
                return holidays_data
        else:
            logger.warning(f"⚠️ ไม่พบไฟล์ {holidays_file}")
            safe_notify(
                title="Holiday File Error",
                message=f"ไม่พบไฟล์ {holidays_file}",
                app_name="Stock Data Updater",
                timeout=10
            )
            return []
            
    except Exception as e:
        logger.error(f"❌ โหลดวันหยุด TH ล้มเหลว: {e}")
        safe_notify(
            title="Holiday File Error",
            message=f"โหลดวันหยุด TH ล้มเหลว: {e}",
            app_name="Stock Data Updater",
            timeout=10
        )
        return []

def is_market_open(now, market):
    """ตรวจสอบว่าตลาดเปิดหรือไม่ - ใช้ pandas_market_calendars"""
    today = now.date()
    weekday = today.weekday()

    # ตรวจสอบวันหยุดสุดสัปดาห์
    if weekday >= 5:
        logger.info(f"📅 วันนี้เป็นวันหยุดสุดสัปดาห์: {today}")
        safe_notify(
            title="Market Closed",
            message=f"วันนี้เป็นวันหยุดสุดสัปดาห์: {today}",
            app_name="Stock Data Updater",
            timeout=10
        )
        return False

    if market == "TH":
        try:
            # ใช้ตลาดไทย XBKK (Stock Exchange of Thailand)
            set_calendar = mcal.get_calendar('XBKK')
            is_open = set_calendar.valid_days(start_date=today, end_date=today)
            is_working = len(is_open) > 0
            
            if not is_working:
                logger.info(f"📅 วันนี้เป็นวันหยุดตลาดไทย (XBKK): {today}")
                safe_notify(
                    title="Market Closed",
                    message=f"วันนี้เป็นวันหยุดตลาดไทย: {today}",
                    app_name="Stock Data Updater",
                    timeout=10
                )
            else:
                logger.info(f"🟢 ตลาดไทย (XBKK) เปิดทำการ: {today}")
                
            return is_working
            
        except ImportError:
            logger.error("❌ ไม่พบ pandas_market_calendars - ติดตั้งด้วย: pip install pandas-market-calendars")
            safe_notify(
                title="Module Error",
                message="ไม่พบ pandas_market_calendars - ติดตั้งด้วย: pip install pandas-market-calendars",
                app_name="Stock Data Updater",
                timeout=10
            )
            
            # Fallback: ใช้ load_market_holidays()
            holidays = load_market_holidays()
            is_holiday = today.strftime("%Y-%m-%d") in holidays
            if is_holiday:
                logger.info(f"📅 วันนี้เป็นวันหยุดตลาดไทย (Fallback): {today}")
                safe_notify(
                    title="Market Closed",
                    message=f"วันนี้เป็นวันหยุดตลาดไทย: {today}",
                    app_name="Stock Data Updater",
                    timeout=10
                )
            return not is_holiday
            
        except Exception as e:
            logger.error(f"❌ ตรวจสอบวันตลาด TH ล้มเหลว: {e}")
            safe_notify(
                title="Market Check Error",
                message=f"ตรวจสอบวันตลาด TH ล้มเหลว: {e}",
                app_name="Stock Data Updater",
                timeout=10
            )
            return weekday < 5

    elif market == "US":
        try:
            # ใช้ NYSE สำหรับตลาดสหรัฐ
            nyse = mcal.get_calendar('NYSE')
            is_open = nyse.valid_days(start_date=today, end_date=today)
            is_working = len(is_open) > 0
            
            if not is_working:
                logger.info(f"📅 วันนี้เป็นวันหยุดตลาดสหรัฐ (NYSE): {today}")
                safe_notify(
                    title="Market Closed",
                    message=f"วันนี้เป็นวันหยุดตลาดสหรัฐ: {today}",
                    app_name="Stock Data Updater",
                    timeout=10
                )
            else:
                logger.info(f"🟢 ตลาดสหรัฐ (NYSE) เปิดทำการ: {today}")
                
            return is_working
            
        except ImportError:
            logger.error("❌ ไม่พบ pandas_market_calendars - ติดตั้งด้วย: pip install pandas-market-calendars")
            safe_notify(
                title="Module Error",
                message="ไม่พบ pandas_market_calendars - ติดตั้งด้วย: pip install pandas-market-calendars",
                app_name="Stock Data Updater",
                timeout=10
            )
            return weekday < 5
            
        except Exception as e:
            logger.error(f"❌ ตรวจสอบวันตลาด US ล้มเหลว: {e}")
            safe_notify(
                title="Market Check Error",
                message=f"ตรวจสอบวันตลาด US ล้มเหลว: {e}",
                app_name="Stock Data Updater",
                timeout=10
            )
            return weekday < 5

    else:
        logger.warning(f"⚠️ ไม่รองรับตลาด: {market}")
        return False

# Initialize running_scripts
running_scripts = set()  # Stores running scripts

# Path for JSON file
LAST_RUN_FILE = "last_run.json"

def initialize_last_run_file():
    """Initialize last_run.json if it doesn't exist or is invalid."""
    if not os.path.exists(LAST_RUN_FILE) or os.path.getsize(LAST_RUN_FILE) == 0:
        with open(LAST_RUN_FILE, 'w') as f:
            json.dump({}, f)
        logger.info(f"Initialized empty {LAST_RUN_FILE}")

def load_last_run():
    """Load last_run data from JSON file."""
    initialize_last_run_file()  # Ensure file exists
    try:
        with open(LAST_RUN_FILE, 'r') as f:
            data = json.load(f)
            # Convert string dates back to datetime, including microseconds
            return {k: datetime.datetime.strptime(v, '%Y-%m-%dT%H:%M:%S.%f') for k, v in data.items()}
    except json.JSONDecodeError as e:
        logger.error(f"Error reading {LAST_RUN_FILE}: {e}")
        # Return empty dict if JSON is invalid
        return {}
    except ValueError as e:
        logger.error(f"Invalid datetime format in {LAST_RUN_FILE}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error reading {LAST_RUN_FILE}: {e}")
        return {}

def update_stock_data(now, market):
    global running_scripts
    last_run = load_last_run()
    
    logger.info(f"⏰ Checking to update market {market} at {now.strftime('%H:%M:%S')}")
    if market in last_run and last_run[market].date() == now.date():
        logger.info(f"⏩ ข้ามการอัปเดต {market} เพราะรันไปแล้ววันนี้ (Last run: {last_run[market]})")
        return
    
    # ตลาดสหรัฐ: เวลา 20:00-21:00 น.
    if market == "US" and now.hour >= 20 and now.hour < 21:
        logger.info("🗂 อัปเดตฐานข้อมูลหุ้นสหรัฐ...")
        safe_notify(
            title="Stock Update",
            message="เริ่มอัปเดตฐานข้อมูลหุ้นสหรัฐ...",
            app_name="Stock Data Updater",
            timeout=10
        )
        
        if update_yfinance():
            # Track success of each step
            all_success = True
            
            for script_group, group_name in [
                (SCRIPTS["stock_us"]["get_stock"], "Get Stock US"),
                (SCRIPTS["stock_us"]["get_financial"], "Get Financial US"),
                (SCRIPTS["stock_us"]["daily_sentiment"], "Daily Sentiment US"),
                (SCRIPTS["stock_us"]["combine_all"], "Combine All US"),
                (SCRIPTS["stock_us"]["stock_to_database"], "Stock to Database US"),
                (SCRIPTS["update stock data"]["update_stock_data"], "Update Stock Data")
            ]:
                for script in script_group:
                    if script in running_scripts:
                        logger.info(f"⏩ ข้าม {script} เพราะกำลังรันอยู่")
                        continue
                    
                    running_scripts.add(script)
                    try:
                        logger.info(f"▶️ เริ่มรัน: {script}")
                        success = run_scripts(script_group, group_name, critical=False)
                        if success:
                            logger.info(f"✅ สำเร็จ: {script}")
                        else:
                            logger.error(f"❌ ล้มเหลว: {script}")
                            all_success = False
                    except Exception as e:
                        logger.error(f"❌ Exception ใน {script}: {e}")
                        all_success = False
                    finally:
                        running_scripts.remove(script)
            
            # Only update last_run if all critical steps succeeded
            if all_success:
                logger.info("✅ อัปเดตข้อมูลหุ้นสหรัฐเรียบร้อย - บันทึก last_run")
                last_run[market] = now
                save_result = save_last_run(last_run)
                if save_result:
                    logger.info(f"✅ บันทึก last_run สำเร็จ: {market} = {now}")
                else:
                    logger.error(f"❌ บันทึก last_run ล้มเหลว")
            else:
                logger.error("❌ บางสคริปต์ล้มเหลว - ไม่บันทึก last_run")
                
            safe_notify(
                title="Stock Update Success" if all_success else "Stock Update Partial",
                message="อัปเดตข้อมูลหุ้นสหรัฐเรียบร้อย" if all_success else "อัปเดตข้อมูลหุ้นสหรัฐบางส่วน",
                app_name="Stock Data Updater",
                timeout=10
            )
        else:
            logger.error("❌ ล้มเหลวในการอัปเดต yfinance")
            safe_notify(
                title="Update Failed",
                message="ล้มเหลวในการอัปเดต yfinance",
                app_name="Stock Data Updater",
                timeout=10
            )
    
    # ตลาดไทย: เวลา 8:00-9:00 น.
    elif market == "TH" and now.hour >= 8 and now.hour < 9:
        logger.info("🗂 อัปเดตฐานข้อมูลหุ้นไทย...")
        safe_notify(
            title="Stock Update",
            message="เริ่มอัปเดตฐานข้อมูลหุ้นไทย...",
            app_name="Stock Data Updater",
            timeout=10
        )
        
        # Track success of each step
        all_success = True
        
        for script_group, group_name in [
            (SCRIPTS["stock_th"]["get_stock"], "Get Stock TH"),
            (SCRIPTS["stock_th"]["get_financial"], "Get Financial TH"),
            (SCRIPTS["stock_th"]["daily_sentiment"], "Daily Sentiment TH"),
            (SCRIPTS["stock_th"]["combine_all"], "Combine All TH"),
            (SCRIPTS["stock_th"]["stock_to_database"], "Stock to Database TH"),
            (SCRIPTS["update stock data"]["update_stock_data"], "Update Stock Data")
        ]:
            for script in script_group:
                if script in running_scripts:
                    logger.info(f"⏩ ข้าม {script} เพราะกำลังรันอยู่")
                    continue
                
                running_scripts.add(script)
                try:
                    logger.info(f"▶️ เริ่มรัน: {script}")
                    success = run_scripts(script_group, group_name, critical=False)
                    if success:
                        logger.info(f"✅ สำเร็จ: {script}")
                    else:
                        logger.error(f"❌ ล้มเหลว: {script}")
                        all_success = False
                except Exception as e:
                    logger.error(f"❌ Exception ใน {script}: {e}")
                    all_success = False
                finally:
                    running_scripts.remove(script)
        
        # Only update last_run if all critical steps succeeded
        if all_success:
            logger.info("✅ อัปเดตข้อมูลหุ้นไทยเรียบร้อย - บันทึก last_run")
            last_run[market] = now
            save_result = save_last_run(last_run)
            if save_result:
                logger.info(f"✅ บันทึก last_run สำเร็จ: {market} = {now}")
            else:
                logger.error(f"❌ บันทึก last_run ล้มเหลว")
        else:
            logger.error("❌ บางสคริปต์ล้มเหลว - ไม่บันทึก last_run")
            
        safe_notify(
            title="Stock Update Success" if all_success else "Stock Update Partial",
            message="อัปเดตข้อมูลหุ้นไทยเรียบร้อย" if all_success else "อัปเดตข้อมูลหุ้นไทยบางส่วน",
            app_name="Stock Data Updater",
            timeout=10
        )
    
    else:
        # ไม่ใช่เวลาอัปเดตหรือตลาดไม่รองรับ
        if market == "US":
            logger.info(f"⏰ ยังไม่ถึงเวลาอัปเดตตลาดสหรัฐ (ปัจจุบัน: {now.hour:02d}:xx, ต้องการ: 20:xx)")
        elif market == "TH":
            logger.info(f"⏰ ยังไม่ถึงเวลาอัปเดตตลาดไทย (ปัจจุบัน: {now.hour:02d}:xx, ต้องการ: 17:xx)")
        else:
            logger.warning(f"⚠️ ไม่รองรับตลาด: {market}")

def save_last_run(last_run):
    """Save last_run data to JSON file with better error handling."""
    try:
        # Convert datetime to ISO format string for JSON serialization
        data = {k: v.isoformat() for k, v in last_run.items()}
        
        # Create backup first
        backup_file = f"{LAST_RUN_FILE}.backup"
        if os.path.exists(LAST_RUN_FILE):
            import shutil
            shutil.copy2(LAST_RUN_FILE, backup_file)
            logger.info(f"สร้าง backup: {backup_file}")
        
        # Write new data
        with open(LAST_RUN_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved last_run to {LAST_RUN_FILE}: {data}")
        
        # Verify the file was written correctly
        verify_data = load_last_run()
        if verify_data != last_run:
            logger.error("❌ Verification failed - data mismatch!")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving {LAST_RUN_FILE}: {e}")
        # Try to restore backup
        backup_file = f"{LAST_RUN_FILE}.backup"
        if os.path.exists(backup_file):
            try:
                import shutil
                shutil.copy2(backup_file, LAST_RUN_FILE)
                logger.info(f"Restored from backup: {backup_file}")
            except Exception as restore_error:
                logger.error(f"Failed to restore backup: {restore_error}")
        return False

def run_scripts(scripts, group_name, critical=False):
    """Enhanced run_scripts with better return value tracking"""
    print(f"\n▶️ Running {group_name}...")
    logger.info(f"🚀 เริ่มรัน: {group_name}")
    
    all_success = True
    
    for script in scripts:
        print(f"  → Running: {script}")
        logger.info(f"  📋 รันสคริปต์: {script}")
        
        try:
            result = subprocess.run([sys.executable, script], 
                                  check=False, 
                                  capture_output=True, 
                                  text=True,
                                  encoding='utf-8',  # Explicitly set encoding to UTF-8
                                  errors='ignore',   # Ignore any residual problematic characters
                                  timeout=1800)  # 30 minutes timeout
            
            if result.returncode == 0:
                logger.info(f"  ✅ สำเร็จ: {script}")
                if result.stdout and result.stdout.strip():
                    logger.debug(f"  📤 Output: {result.stdout.strip()}")
            else:
                logger.error(f"  ❌ ล้มเหลว: {script} (Exit code: {result.returncode})")
                if result.stderr and result.stderr.strip():
                    logger.error(f"  📤 Error: {result.stderr.strip()}")
                all_success = False
                
                if critical:
                    logger.error(f"🛑 Critical script failed: {script}")
                    return False
                    
        except subprocess.TimeoutExpired:
            logger.error(f"  ⏰ Timeout: {script}")
            all_success = False
            if critical:
                return False
                
        except Exception as e:
            logger.error(f"  ❌ Exception: {script} - {e}")
            all_success = False
            if critical:
                return False
    
    if all_success:
        print(f"✅ Done: {group_name}")
        logger.info(f"🎉 เสร็จสิ้น: {group_name}")
    else:
        print(f"⚠️ Completed with errors: {group_name}")
        logger.warning(f"⚠️ เสร็จสิ้นแต่มีข้อผิดพลาด: {group_name}")
    
    return all_success
        
def update_stock_data_ignore_time():
    logger.info("🗂 อัปเดตฐานข้อมูลหุ้นสหรัฐ...")
    safe_notify(
        title="Stock Update",
        message="เริ่มอัปเดตฐานข้อมูลหุ้นสหรัฐ...",
        app_name="Stock Data Updater",
        timeout=10
    )
    last_run = load_last_run()
    now = datetime.datetime.now()
    market = "US"
    if market in last_run and last_run[market].date() == now.date():
        logger.info(f"⏩ ข้ามการอัปเดต {market} เพราะรันไปแล้ววันนี้")
        return
    if update_yfinance():
        run_scripts(SCRIPTS["stock_us"]["get_stock"], "Get Stock US", critical=False)
        run_scripts(SCRIPTS["stock_us"]["get_financial"], "Get Financial US", critical=False)
        run_scripts(SCRIPTS["stock_us"]["daily_sentiment"], "Daily Sentiment US", critical=False)
        run_scripts(SCRIPTS["stock_us"]["combine_all"], "Combine All US", critical=False)
        run_scripts(SCRIPTS["stock_us"]["stock_to_database"], "Stock to Database US", critical=False)
        logger.info("✅ อัปเดตข้อมูลหุ้นสหรัฐเรียบร้อย")
        safe_notify(
            title="Stock Update Success",
            message="อัปเดตข้อมูลหุ้นสหรัฐเรียบร้อย",
            app_name="Stock Data Updater",
            timeout=10
        )
        last_run[market] = now
        save_last_run(last_run)
    
    logger.info("🗂 อัปเดตฐานข้อมูลหุ้นไทย...")
    safe_notify(
        title="Stock Update",
        message="เริ่มอัปเดตฐานข้อมูลหุ้นไทย...",
        app_name="Stock Data Updater",
        timeout=10
    )
    market = "TH"
    if market in last_run and last_run[market].date() == now.date():
        logger.info(f"⏩ ข้ามการอัปเดต {market} เพราะรันไปแล้ววันนี้")
        return
    if update_yfinance():
        run_scripts(SCRIPTS["stock_th"]["get_stock"], "Get Stock TH", critical=False)
        run_scripts(SCRIPTS["stock_th"]["get_financial"], "Get Financial TH", critical=False)
        run_scripts(SCRIPTS["stock_th"]["daily_sentiment"], "Daily Sentiment TH", critical=False)
        run_scripts(SCRIPTS["stock_th"]["combine_all"], "Combine All TH", critical=False)
        run_scripts(SCRIPTS["stock_th"]["stock_to_database"], "Stock to Database TH", critical=False)
        logger.info("✅ อัปเดตข้อมูลหุ้นไทยเรียบร้อย")
        safe_notify(
            title="Stock Update Success",
            message="อัปเดตข้อมูลหุ้นไทยเรียบร้อย",
            app_name="Stock Data Updater",
            timeout=10
        )
        last_run[market] = now
        save_last_run(last_run)

def get_user_input():
    """รับ input จากผู้ใช้พร้อม timeout"""
    user_input = []

    def ask_input():
        try:
            mode = input("Enter mode (1 or 2 or 3): ").strip()
            user_input.append(mode if mode in ["1", "2", "3"] else "1")
        except:
            user_input.append("1")

    print("Select mode:")
    print("1. Auto run (ทำงานอัตโนมัติทุก 2 ชั่วโมง)")
    print("2. Manual run (รันครั้งเดียวแล้วหยุด)")
    print("3. Manual run stock (รันครั้งเดียวสำหรับข้อมูลหุ้น)")

    input_thread = threading.Thread(target=ask_input)
    input_thread.daemon = True
    input_thread.start()
    input_thread.join(timeout=10)
    
    mode = user_input[0] if user_input else "1"
    logger.info(f"mode: {mode}")
    safe_notify(
        title="Mode Selected",
        message=f"เลือกโหมด: {mode}",
        app_name="Stock Data Updater",
        timeout=10
    )
    return mode

def run_auto_mode():
    """โหมดอัตโนมัติ"""
    last_run_hour = None
    logger.info("🤖 เริ่มโหมดอัตโนมัติ - กด Ctrl+C เพื่อหยุด")
    safe_notify(
        title="Auto Mode",
        message="เริ่มโหมดอัตโนมัติ",
        app_name="Stock Data Updater",
        timeout=10
    )
    
    try:
        while True:
            now = datetime.datetime.now()
            current_hour = now.hour
            
            update_stock_data(now, "US")
            update_stock_data(now, "TH")
            
            if (current_hour % 2 == 0 and 
                current_hour != last_run_hour and 
                now.minute == 0):
                logger.info(f"🕒 Running news scripts at {now.strftime('%H:%M:%S')}")
                last_run_hour = current_hour
                
                try:
                    run_all_news_scripts()
                    
                    if now.hour == 0 and now.minute == 0:
                        logger.info("🗑️ ลบไฟล์ CSV ทุก 1 วัน")
                        clear_stock_csv()
                    
                    logger.info("🎉 All scripts completed successfully.")
                    safe_notify(
                        title="All Scripts Completed",
                        message="ทุกสคริปต์ทำงานสำเร็จ",
                        app_name="Stock Data Updater",
                        timeout=10
                    )
                    
                except Exception as e:
                    logger.error(f"❌ Unexpected error: {e}")
                    safe_notify(
                        title="Unexpected Error",
                        message=f"เกิดข้อผิดพลาด: {e}",
                        app_name="Stock Data Updater",
                        timeout=10
                    )
            
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("⏹️ หยุดการทำงานโดยผู้ใช้")
        safe_notify(
            title="Program Stopped",
            message="หยุดการทำงานโดยผู้ใช้",
            app_name="Stock Data Updater",
            timeout=10
        )
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในโหมดอัตโนมัติ: {e}")
        safe_notify(
            title="Auto Mode Error",
            message=f"เกิดข้อผิดพลาดในโหมดอัตโนมัติ: {e}",
            app_name="Stock Data Updater",
            timeout=10
        )

def run_manual_mode():
    """โหมดรันครั้งเดียว"""
    logger.info("🔧 เริ่มโหมดรันครั้งเดียว")
    safe_notify(
        title="Manual Mode",
        message="เริ่มโหมดรันครั้งเดียว",
        app_name="Stock Data Updater",
        timeout=10
    )
    try:
        success = run_all_news_scripts()
        if success:
            logger.info("🎉 All scripts completed successfully.")
            safe_notify(
                title="Manual Mode Success",
                message="ทุกสคริปต์ทำงานสำเร็จ",
                app_name="Stock Data Updater",
                timeout=10
            )
        else:
            logger.error("❌ บางสคริปต์ล้มเหลว")
            safe_notify(
                title="Manual Mode Error",
                message="บางสคริปต์ล้มเหลว",
                app_name="Stock Data Updater",
                timeout=10
            )
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        safe_notify(
            title="Unexpected Error",
            message=f"เกิดข้อผิดพลาด: {e}",
            app_name="Stock Data Updater",
            timeout=10
        )

def main():
    """ฟังก์ชันหลัก"""
    logger.info("🚀 เริ่มต้นโปรแกรม")
    safe_notify(
        title="Program Started",
        message="เริ่มต้นโปรแกรม",
        app_name="Stock Data Updater",
        timeout=10
    )
    
    try:
        mode = get_user_input()
        
        if mode == "1":
            run_auto_mode()
        elif mode == "2":
            run_manual_mode()
        else:
            update_stock_data_ignore_time()
            
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในโปรแกรมหลัก: {e}")
        safe_notify(
            title="Program Error",
            message=f"เกิดข้อผิดพลาดในโปรแกรมหลัก: {e}",
            app_name="Stock Data Updater",
            timeout=10
        )
    finally:
        logger.info("🔚 จบการทำงาน")
        safe_notify(
            title="Program Ended",
            message="จบการทำงาน",
            app_name="Stock Data Updater",
            timeout=10
        )

if __name__ == "__main__":
    main()