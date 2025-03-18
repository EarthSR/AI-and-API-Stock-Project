import subprocess
import sys
import os
import time
import psutil  # ✅ ใช้ตรวจสอบ CPU/RAM
from datetime import datetime, timedelta

# ✅ บังคับให้ stdout ใช้ UTF-8 และข้ามอีโมจิโดยอัตโนมัติ
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# ✅ กำหนดชื่อไฟล์ Log
log_file = "system_run.log"

# ✅ ฟังก์ชันสำหรับบันทึก Log
def write_log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"[{timestamp}] {message}\n")
    print(message)  # แสดงผลใน Console ด้วย

# ✅ ฟังก์ชันตรวจสอบระบบ (CPU, RAM)
def check_system_resources():
    memory = psutil.virtual_memory()
    cpu_usage = psutil.cpu_percent(interval=1)

    write_log(f"🔍 Memory ใช้ไป: {memory.percent:.2f}% | CPU ใช้ไป: {cpu_usage:.2f}%")

    if memory.percent > 90:
        write_log("⚠️ RAM ใช้เกิน 90%! หยุดการทำงานเพื่อป้องกันค้าง...")
        exit(1)

    if cpu_usage > 95:
        write_log("⚠️ CPU ใช้เกิน 95%! หยุดการทำงานเพื่อป้องกัน Overload...")
        exit(1)

# ✅ ฟังก์ชันอัปเดต `yfinance`
def update_yfinance():
    write_log("🔄 กำลังอัปเดต `yfinance` เป็นเวอร์ชันล่าสุด...")
    subprocess.run(["pip", "install", "--upgrade", "yfinance"], check=True)
    write_log("✅ อัปเดต `yfinance` สำเร็จ")

# ✅ ฟังก์ชันรันสคริปต์เฉพาะตลาดที่กำหนด
def run_scripts_for_market(market):
    write_log(f"\n🚀 เริ่มต้นรันสคริปต์สำหรับตลาด {market} ณ เวลา: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 🔹 แยกสคริปต์ตามตลาด
    thai_scripts = [
        "./Thai_News_Database.py",
        "./Financial_Thai_Quarter.py",
        "./GetdataThaiStocks.py"
    ]

    america_scripts = [
        "./America_News_Database.py",
        "./Financial_America_Quarter.py",
        "./GetdataAmericanStock.py"
    ]

    common_scripts = [
        "./Final_News_Merge.py",
        "./Finbert.py",
        "./calculate_weighted_sentiment.py",
        "./News_Database.py",
        "./combine.py",
        "./Stock_Database.py",
        "./GDP_Database.py",
        "./Autotrainmodel.py"
    ]

    scripts_to_run = thai_scripts if market == "ไทย" else america_scripts
    scripts_to_run += common_scripts  # รวมสคริปต์ที่ใช้ร่วมกัน

    def is_script_exist(script_path):
        """✅ ตรวจสอบว่า script มีอยู่จริง"""
        return os.path.isfile(script_path)

    for step, script in enumerate(scripts_to_run, start=1):
        write_log(f"\n--- Step {step}: Running `{script}` ---\n")

        if not is_script_exist(script):
            write_log(f"❌ Error: ไม่พบไฟล์ `{script}` ข้ามการรัน...")
            continue

        try:
            # ✅ บังคับให้ Python รับรู้ว่าไฟล์มีการเปลี่ยนแปลง
            write_log(f"🔄 กำลังล้าง Cache ของ `{script}`...")
            os.utime(script, None)

            # ✅ รันสคริปต์พร้อมกำหนด Timeout 10 นาที
            process = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=600
            )

            stdout, stderr = process.stdout.strip(), process.stderr.strip()
            write_log(f"\n📌 Output จาก `{script}`:\n{stdout}")
            if stderr:
                write_log(f"⚠️ Error จาก `{script}`:\n{stderr}")

        except subprocess.TimeoutExpired:
            write_log(f"⚠️ Timeout: `{script}` ใช้เวลานานเกิน 10 นาที ข้ามไปยังขั้นตอนถัดไป")
        except Exception as e:
            write_log(f"❌ Error: `{script}` ล้มเหลว: {e}")

        write_log(f"\n✅ --- Step {step} เสร็จสมบูรณ์ ---\n")

# ✅ ฟังก์ชันรอจนถึงรอบถัดไป
def wait_until_next_run():
    now = datetime.now()

    # ตั้งค่าช่วงเวลารัน
    market_times = [
        (now.replace(hour=8, minute=30, second=0, microsecond=0), "ไทย"),
        (now.replace(hour=19, minute=30, second=0, microsecond=0), "อเมริกา")
    ]

    # ค้นหาช่วงเวลาถัดไปที่ต้องรัน
    next_run_time, market = None, None
    for market_time, market_name in market_times:
        if now < market_time:
            next_run_time, market = market_time, market_name
            break

    if next_run_time is None:
        next_run_time, market = market_times[0][0] + timedelta(days=1), market_times[0][1]

    wait_seconds = (next_run_time - now).total_seconds()

    write_log("\n\n\n" + "=" * 80 + "\n" * 5)  
    write_log(f"⏳ จะรันตลาด {market} อีก {int(wait_seconds)} วินาที ({next_run_time.strftime('%Y-%m-%d %H:%M:%S')})...\n")
    write_log("\n" * 5 + "=" * 80 + "\n\n\n")  

    time.sleep(wait_seconds)

# ✅ เช็คว่าไม่มี Process ค้าง
def kill_old_processes():
    write_log("🔍 กำลังตรวจสอบ Process ค้าง...")
    for proc in psutil.process_iter(attrs=['pid', 'name']):
        try:
            if "python" in proc.info['name'].lower():
                write_log(f"🛑 กำลังหยุด Process `{proc.info['name']}` (PID: {proc.info['pid']})")
                os.kill(proc.info['pid'], 9)
        except Exception as e:
            write_log(f"⚠️ ไม่สามารถหยุด Process {proc.info['pid']}: {e}")

# ✅ อัปเดต `yfinance` ก่อนเริ่ม
update_yfinance()

# ✅ ตรวจสอบทรัพยากรระบบ
check_system_resources()

# ✅ ล้าง Process ค้าง
kill_old_processes()

# ✅ ลูปรันสคริปต์ทุกวันสองรอบ (ตลาดไทย + อเมริกา)
while True:
    now = datetime.now()
    if now.hour < 12:  # ถ้ายังเป็นช่วงเช้า รันตลาดไทย
        run_scripts_for_market("ไทย")
    else:  # ถ้าเป็นช่วงบ่ายไปแล้ว รันตลาดอเมริกา
        run_scripts_for_market("อเมริกา")
    
    wait_until_next_run()
