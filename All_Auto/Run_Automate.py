import subprocess
import sys
import os
import time
from datetime import datetime, timedelta

# ✅ บังคับให้ stdout ใช้ UTF-8 และข้ามอีโมจิโดยอัตโนมัติ
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# ✅ กำหนดชื่อไฟล์ Log (บันทึกต่อเนื่อง)
log_file = "system_run.log"

# ✅ ฟังก์ชันสำหรับบันทึก Log (Append Mode)
def write_log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"[{timestamp}] {message}\n")
    print(message)  # แสดงผลใน Console ด้วย


# ✅ ฟังก์ชันอัปเดต `yfinance` เป็นเวอร์ชันล่าสุดทุกครั้งก่อนรัน
def update_yfinance():
    write_log("🔄 กำลังอัปเดต `yfinance` เป็นเวอร์ชันล่าสุด...")
    subprocess.run(["pip", "install", "--upgrade", "yfinance"], check=True)
    write_log("✅ อัปเดต `yfinance` สำเร็จ")

# ✅ ฟังก์ชันรันสคริปต์เฉพาะตลาดที่กำหนด
def run_scripts_for_market(market):
    write_log(f"\n🚀 เริ่มต้นรันสคริปต์สำหรับตลาด {market} ณ เวลา: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 🔹 แยกสคริปต์ที่ใช้เฉพาะตลาดไทยและตลาดอเมริกา
    thai_scripts = [
        ["./Thai_News_Database.py"],  # ข่าวไทย
        ["./Financial_Thai_Quarter.py"],  # ข้อมูลงบการเงินไทย
        ["./GetdataThaiStocks.py"],  # ดึงข้อมูลหุ้นไทย
    ]

    america_scripts = [
        ["./America_News_Database.py"],  # ข่าวอเมริกา
        ["./Financial_America_Quarter.py"],  # ข้อมูลงบการเงินอเมริกา
        ["./GetdataAmericanStock.py"],  # ดึงข้อมูลหุ้นอเมริกา
    ]

    common_scripts = [
        ["./Final_News_Merge.py"],  # รวมข่าว
        ["./Finbert.py"],  # วิเคราะห์ Sentiment
        ["./calculate_weighted_sentiment.py"],  # ทำ daily sentiment
        ["./News_Database.py"],  # บันทึกข่าวลง Database
        ["./combine.py"],  # รวมข้อมูลทำ AI
        ["./Stock_Database.py"],  # บันทึกข้อมูลหุ้นลง Database
        ["./GDP_Database.py"],  # บันทึกข้อมูล GDP ลง Database
        ["./Autotrainmodel.py"],  # Retrain Model
    ]

    # ✅ เลือกสคริปต์ที่ต้องรัน
    scripts_to_run = thai_scripts if market == "ไทย" else america_scripts
    scripts_to_run += common_scripts  # รวมสคริปต์ที่ใช้ร่วมกัน

    def is_script_exist(script_path):
        """✅ ตรวจสอบว่า script มีอยู่จริง"""
        return os.path.isfile(script_path)

    for step, script_set in enumerate(scripts_to_run, start=1):
        processes = []
        write_log(f"\n--- Step {step}: Running {script_set} ---\n")

        for script in script_set:
            if not is_script_exist(script):
                write_log(f"❌ Error: ไม่พบไฟล์ `{script}` ข้ามการรัน...")
                continue

            try:
                # ✅ Force Overwrite: บังคับให้รันสคริปต์ใหม่ทุกครั้ง
                write_log(f"🔄 กำลังล้าง Cache ของ `{script}`...")
                if os.path.exists(script):
                    os.utime(script, None)  # บังคับให้ Python รู้ว่าไฟล์เปลี่ยนแปลง
                
                # ✅ รันสคริปต์พร้อมรอให้เสร็จ
                process = subprocess.run(
                    [sys.executable, script],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore"
                )
                processes.append((script, process))
            except Exception as e:
                write_log(f"❌ Error: ไม่สามารถรัน `{script}` ได้: {e}")
                continue

        for script, process in processes:
            stdout, stderr = process.stdout.strip(), process.stderr.strip()
            write_log(f"\n📌 Output จาก `{script}`:\n{stdout}")
            if stderr:
                write_log(f"⚠️ Error จาก `{script}`:\n{stderr}")

        write_log(f"\n✅ --- Step {step} เสร็จสมบูรณ์ ---\n")

# ✅ ลบไฟล์ทั้งหมดในโฟลเดอร์ Stock News หลังจากรันเสร็จ
def clear_stock_folder():
    folder_path = "./Stock"  # เปลี่ยนเป็นพาธที่ถูกต้อง
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    write_log(f"🗑 ลบไฟล์: {file_path}")
            except Exception as e:
                write_log(f"⚠️ ไม่สามารถลบไฟล์ {file_path}: {e}")

    write_log("✅ ลบไฟล์ในโฟลเดอร์ Stock News สำเร็จ")

def clear_News_folder():
    folder_path = "./News"  # เปลี่ยนเป็นพาธที่ถูกต้อง
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    write_log(f"🗑 ลบไฟล์: {file_path}")
            except Exception as e:
                write_log(f"⚠️ ไม่สามารถลบไฟล์ {file_path}: {e}")

    write_log("✅ ลบไฟล์ในโฟลเดอร์ Stock News สำเร็จ")

def delete_specific_files(file_names):
    folder_path = "."  # โฟลเดอร์หลัก (โฟลเดอร์ปัจจุบัน)

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):  # ตรวจสอบว่าเป็นไฟล์จริง
                os.remove(file_path)
                write_log(f"🗑 ลบไฟล์: {file_path}")
            else:
                write_log(f"⚠️ ไฟล์ {file_name} ไม่พบหรือไม่ใช่ไฟล์ที่ถูกต้อง")
        except Exception as e:
            write_log(f"⚠️ ไม่สามารถลบไฟล์ {file_path}: {e}")

    write_log("✅ การลบไฟล์ตามชื่อที่ระบุเสร็จสิ้น")

# ตัวอย่างการเรียกใช้: ลบไฟล์ 'test1.txt', 'data.csv', 'report.pdf'
delete_specific_files(["merged_stock_sentiment_financial.csv", "GDP_US.csv", "GDP_TH.csv", "GDP_AllData.csv"])

# ✅ เรียกใช้ฟังก์ชันลบไฟล์หลังจากรันสคริปต์ทั้งหมด
clear_stock_folder()
clear_News_folder()

# ✅ ฟังก์ชันรอจนถึงรอบถัดไป (ตลาดไทยหรืออเมริกา)
def wait_until_next_run():
    now = datetime.now()

    # ตั้งค่าช่วงเวลารัน
    market_times = [
        (now.replace(hour=8, minute=30, second=0, microsecond=0), "ไทย"),   # ตลาดไทย
        (now.replace(hour=19, minute=30, second=0, microsecond=0), "อเมริกา")   # ตลาดอเมริกา
    ]

    # ค้นหาช่วงเวลาถัดไปที่ต้องรัน
    next_run_time, market = None, None
    for market_time, market_name in market_times:
        if now < market_time:
            next_run_time, market = market_time, market_name
            break

    # ถ้าขณะนี้เลยเวลารันไปแล้ว ให้ข้ามไปรอบถัดไปของวันพรุ่งนี้
    if next_run_time is None:
        next_run_time, market = market_times[0][0] + timedelta(days=1), market_times[0][1]

    wait_seconds = (next_run_time - now).total_seconds()

    # 🔹 เพิ่มบรรทัดว่างเพื่อให้ดูง่ายขึ้น
    write_log("\n\n\n" + "=" * 80 + "\n" * 5)  
    write_log(f"⏳ จะรันตลาด {market} อีก {int(wait_seconds)} วินาที ({next_run_time.strftime('%Y-%m-%d %H:%M:%S')})...\n")
    write_log("\n" * 5 + "=" * 80 + "\n\n\n")  

    time.sleep(wait_seconds)

# ✅ อัปเดต `yfinance` ก่อนเริ่ม
update_yfinance()


# ✅ ลูปรันสคริปต์ทุกวันสองรอบ (ตลาดไทย + อเมริกา)
while True:
    now = datetime.now()
    if now.hour < 12:  # ถ้ายังเป็นช่วงเช้า รันตลาดไทย
        run_scripts_for_market("ไทย")
    else:  # ถ้าเป็นช่วงบ่ายไปแล้ว รันตลาดอเมริกา
        run_scripts_for_market("อเมริกา")
    
    wait_until_next_run()
