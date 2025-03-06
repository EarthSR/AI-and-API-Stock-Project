import subprocess
import sys
import os
import time
from datetime import datetime, timedelta

# ✅ บังคับให้ stdout ใช้ UTF-8 และข้ามอีโมจิโดยอัตโนมัติ
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# ✅ ฟังก์ชันรันสคริปต์ทั้งหมด
def run_all_scripts():
    print(f"\n🚀 เริ่มต้นรันสคริปต์ ณ เวลา: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    scripts = [
        ["ANT_Final/Thai_News_Database.py"],  # Step 1 - ข่าวไทย
        ["ANT_Final/America_News_Database.py"],  # Step 2 - ข่าว Investing.com
        ["ANT_Final/Final_News_Merge.py"],  # Step 3 - รวมข่าว
        ["Finbert/Finbert.py"],  # Step 4 - วิเคราะห์ Sentiment
        ["Finbert/calculate_weighted_sentiment.py"],  # Step 5 - ทำ daily sentiment
        ["ANT_FinaL/News_Database.py"],  # Step 6 - บันทึกลง Database
        ["American_stocks/GetdataAmericanStock.py"],  # Step 7 - บันทึก USA Stock
        ["Thai_stocks/GetdataThaiStocks.py"],  # Step 8 - บันทึก Thai Stock
        ["combine.py"],  # Step 9 - รวมข้อมูลทำ AI
        ["ANT_FinaL/Stock_Database.py"],  # Step 10 - บันทึกลง Stock ลง Database
    ]

    def is_script_exist(script_path):
        """✅ ตรวจสอบว่า script มีอยู่จริง"""
        return os.path.isfile(script_path)

    for step, script_set in enumerate(scripts, start=1):
        processes = []
        print(f"\n--- Step {step}: Running {script_set} ---\n")

        for script in script_set:
            if not is_script_exist(script):
                print(f"❌ Error: ไม่พบไฟล์ `{script}` ข้ามการรัน...")
                continue

            try:
                process = subprocess.Popen(
                    [sys.executable, script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="ignore"  # ✅ ข้าม Unicode Error เช่น อีโมจิ
                )
                processes.append((script, process))
            except Exception as e:
                print(f"❌ Error: ไม่สามารถรัน `{script}` ได้: {e}")
                continue

        for script, process in processes:
            stdout, stderr = process.communicate()
            print(f"\n📌 Output จาก `{script}`:\n{stdout.strip()}")
            if stderr.strip():
                print(f"⚠️ Error จาก `{script}`:\n{stderr.strip()}")

        print(f"\n✅ --- Step {step} เสร็จสมบูรณ์ ---\n")

# ✅ ฟังก์ชันรอจนถึงเที่ยงคืน 1 นาที
def wait_until_midnight():
    now = datetime.now()
    next_run_time = (now + timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)
    wait_seconds = (next_run_time - now).total_seconds()

    print(f"\n⏳ จะรันรอบถัดไปในอีก {int(wait_seconds)} วินาที ({next_run_time.strftime('%Y-%m-%d %H:%M:%S')})...\n")
    time.sleep(wait_seconds)

# ✅ ลูปรันสคริปต์ทุกวันตอน 00:01 น.
while True:
    run_all_scripts()
    wait_until_midnight()
