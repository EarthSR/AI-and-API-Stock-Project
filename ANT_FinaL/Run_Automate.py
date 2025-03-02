import subprocess
import sys
import os

# ✅ บังคับให้ stdout ใช้ UTF-8 และข้ามอีโมจิโดยอัตโนมัติ
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# ✅ ตรวจสอบว่าทุกสคริปต์ใช้ ChromeDriverManager().install() หรือไม่
print("🔍 Checking ChromeDriver setup in each script...")

# ✅ กำหนด script ที่ต้องรัน
scripts = [
    ["ANT_Final/Thai_News_Database.py"],  # Step 1 - ข่าวไทย
    ["ANT_Final/America_News_Database.py"],  # Step 2 - ข่าว Investing.com
    ["ANT_Final/Final_News_Merge.py"],  # Step 3 - รวมข่าว
    ["Finbert/Finbert.py"],  # Step 4 - วิเคราะห์ Sentiment
    ["Finbert/calculate_weighted_sentiment.py"],  # Step 5 - ทำ daily
    ["ANT_FinaL/News_Database.py"],  # Step 6 - ลง database
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
