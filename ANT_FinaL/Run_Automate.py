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

# ✅ ฟังก์ชันรันสคริปต์ทั้งหมด
def run_all_scripts():
    write_log(f"\n🚀 เริ่มต้นรันสคริปต์ ณ เวลา: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    scripts = [
        ["ANT_Final/Thai_News_Database.py"],  # Step 1 - ข่าวไทย
        ["ANT_Final/America_News_Database.py"],  # Step 2 - ข่าว Investing.com
        ["ANT_Final/Final_News_Merge.py"],  # Step 3 - รวมข่าว
        ["Finbert/Finbert.py"],  # Step 4 - วิเคราะห์ Sentiment
        ["Finbert/calculate_weighted_sentiment.py"],  # Step 5 - ทำ daily sentiment
        ["ANT_Final/News_Database.py"],  # Step 6 - บันทึกลง Database
        ["ANT_Final/Financial_America_Quarter.py"],  # Step 7 - บันทึก Financial USA
        ["ANT_Final/Financial_Thai_Quarter.py"],  # Step 8 - บันทึก Financial Thai
        ["American_stocks/GetdataAmericanStock.py"],  # Step 9 - บันทึก USA Stock
        ["Thai_stocks/GetdataThaiStocks.py"],  # Step 10 - บันทึก Thai Stock
        ["combine.py"],  # Step 11 - รวมข้อมูลทำ AI
        ["ANT_Final/Stock_Database.py"],  # Step 12 - บันทึกลง Stock ลง Database
        ["ANT_Final/GDP_Database.py"],  # Step 13 - บันทึกลง GDP ลง Database
    ]

    def is_script_exist(script_path):
        """✅ ตรวจสอบว่า script มีอยู่จริง"""
        return os.path.isfile(script_path)

    for step, script_set in enumerate(scripts, start=1):
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

# ✅ ฟังก์ชันรอจนถึงเที่ยงคืน 1 นาที
def wait_until_midnight():
    now = datetime.now()
    next_run_time = (now + timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)
    wait_seconds = (next_run_time - now).total_seconds()

    # 🔹 เพิ่มบรรทัดว่างเพื่อให้ดูง่ายขึ้น
    write_log("\n\n\n" + "=" * 80 + "\n" * 5)  # 🔹 แสดงเส้นขั้นและเว้นบรรทัดเยอะๆ
    write_log(f"⏳ จะรันรอบถัดไปในอีก {int(wait_seconds)} วินาที ({next_run_time.strftime('%Y-%m-%d %H:%M:%S')})...\n")
    write_log("\n" * 5 + "=" * 80 + "\n\n\n")  # 🔹 แสดงเส้นขั้นและเว้นบรรทัดเยอะๆ

    time.sleep(wait_seconds)

# ✅ ลูปรันสคริปต์ทุกวันตอน 00:01 น.
while True:
    run_all_scripts()
    wait_until_midnight()
