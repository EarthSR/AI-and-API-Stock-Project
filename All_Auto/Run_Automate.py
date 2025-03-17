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

# ✅ อัปเดต `yfinance` ทุกครั้ง
def update_yfinance():
    try:
        write_log("🔄 กำลังอัปเดต `yfinance` เป็นเวอร์ชันล่าสุด...")
        subprocess.run(["pip", "install", "--upgrade", "yfinance"], check=True, timeout=300)
        write_log("✅ อัปเดต `yfinance` สำเร็จ")
    except subprocess.TimeoutExpired:
        write_log("⚠️ อัปเดต `yfinance` ไม่สำเร็จ (Timeout)")
    except subprocess.CalledProcessError:
        write_log("❌ Error: อัปเดต `yfinance` ล้มเหลว")

# ✅ ฟังก์ชันตรวจสอบไลบรารีก่อนรันทุกสคริปต์
def check_libraries():
    required_libraries = ["yfinance", "pandas", "tensorflow", "numpy"]  # ✅ ลดจำนวนให้เร็วขึ้น
    installed_libs = subprocess.run(["pip", "list"], capture_output=True, text=True).stdout.lower()

    for lib in required_libraries:
        if lib.lower() not in installed_libs:
            try:
                write_log(f"⚠️ ไม่พบ {lib} → กำลังติดตั้ง...")
                subprocess.run(["pip", "install", lib], check=True, timeout=300)
                write_log(f"✅ ติดตั้ง {lib} สำเร็จ")
            except subprocess.TimeoutExpired:
                write_log(f"⚠️ ติดตั้ง {lib} ไม่สำเร็จ (Timeout)")
            except subprocess.CalledProcessError:
                write_log(f"❌ Error: ติดตั้ง {lib} ล้มเหลว")

# ✅ ฟังก์ชันรันสคริปต์แบบป้องกันค้าง
def run_script(script_path):
    """ รันสคริปต์และป้องกันค้าง """
    if not os.path.isfile(script_path):
        write_log(f"❌ Error: ไม่พบไฟล์ `{script_path}` ข้ามการรัน...")
        return

    try:
        write_log(f"🔄 กำลังรัน `{script_path}`...")
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )

        try:
            stdout, stderr = process.communicate(timeout=600)  # ป้องกันค้าง
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()  # ดึง log หลังจาก kill
            write_log(f"⚠️ `{script_path}` ค้างเกิน 10 นาที → ข้ามไป")
        except Exception as e:
            write_log(f"❌ `{script_path}` ล้มเหลว: {e}")
            return

        # ✅ เขียน log
        write_log(f"📌 Output จาก `{script_path}`:\n{stdout}")
        if stderr:
            write_log(f"⚠️ Error จาก `{script_path}`:\n{stderr}")

        if process.returncode != 0:
            write_log(f"⚠️ `{script_path}` จบด้วยรหัสผิดพลาด ({process.returncode})")

        write_log(f"✅ `{script_path}` เสร็จสมบูรณ์")

    except Exception as e:
        write_log(f"❌ `{script_path}` ล้มเหลว: {e}")


# ✅ ฟังก์ชันรันสคริปต์เฉพาะตลาดที่กำหนด
def run_scripts_for_market(market):
    write_log(f"\n🚀 เริ่มต้นรันสคริปต์สำหรับตลาด {market} ณ เวลา: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    scripts = {
        "ไทย": ["./Thai_News_Database.py", "./Financial_Thai_Quarter.py", "./GetdataThaiStocks.py"],
        "อเมริกา": ["./America_News_Database.py", "./Financial_America_Quarter.py", "./GetdataAmericanStock.py"],
        "common": ["./Final_News_Merge.py", "./Finbert.py", "./calculate_weighted_sentiment.py", "./News_Database.py",
                   "./combine.py", "./Stock_Database.py", "./GDP_Database.py", "./Autotrainmodel.py"]
    }

    scripts_to_run = scripts[market] + scripts["common"]

    for script in scripts_to_run:
        run_script(script)

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

    write_log("\n\n\n" + "=" * 80 + "\n" * 5)
    write_log(f"⏳ จะรันตลาด {market} อีก {int(wait_seconds)} วินาที ({next_run_time.strftime('%Y-%m-%d %H:%M:%S')})...\n")
    write_log("\n" * 5 + "=" * 80 + "\n\n\n")

    time.sleep(wait_seconds)

# ✅ อัปเดต `yfinance` ก่อนเริ่ม
update_yfinance()

# ✅ เช็คไลบรารีก่อนรัน
check_libraries()

# ✅ ลูปรันสคริปต์ทุกวันสองรอบ (ตลาดไทย + อเมริกา)
try:
    while True:
        now = datetime.now()
        if now.hour < 12:  # ถ้ายังเป็นช่วงเช้า รันตลาดไทย
            run_scripts_for_market("ไทย")
        else:  # ถ้าเป็นช่วงบ่ายไปแล้ว รันตลาดอเมริกา
            run_scripts_for_market("อเมริกา")

        wait_until_next_run()
except KeyboardInterrupt:
    write_log("❌ ถูกตัดการทำงานโดย `KeyboardInterrupt` → หยุดลูปหลัก")
