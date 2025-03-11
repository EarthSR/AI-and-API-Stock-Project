import os
import mysql.connector
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import sys

# ✅ ป้องกัน UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# ✅ โหลดตัวแปรจาก .env
load_dotenv()

# ✅ ตรวจสอบระดับของโฟลเดอร์ (ปรับ `..` ตามตำแหน่งของไฟล์)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ✅ โหลดข้อมูลจากไฟล์ CSV
predictions_df = pd.read_csv("all_predictions_per_day_multi_task.csv")

# ✅ แปลงคอลัมน์ 'Date' ให้เป็น datetime
predictions_df["Date"] = pd.to_datetime(predictions_df["Date"], errors='coerce')

# ✅ เปลี่ยนชื่อคอลัมน์ให้เหมาะสม
predictions_df.rename(columns={'Predicted_Price': 'PredictionClose', 'Predicted_Dir': 'PredictionTrend', 'Ticker': 'StockSymbol'}, inplace=True)

# ✅ เลือกเฉพาะคอลัมน์ที่ต้องการ: 'StockSymbol', 'Date', 'PredictionClose', 'PredictionTrend'
predictions_df = predictions_df[['StockSymbol', 'Date', 'PredictionClose', 'PredictionTrend']]

# ✅ ตรวจสอบข้อมูลซ้ำ: ถ้ามี `StockSymbol` และ `Date` ซ้ำ ให้เก็บแค่แถวแรก
predictions_df.drop_duplicates(subset=['StockSymbol', 'Date'], keep='first', inplace=True)

# ✅ บันทึกข้อมูลลงไฟล์ CSV ก่อน
predictions_df.to_csv("cleaned_predictions.csv", index=False)
print("✅ บันทึกข้อมูลลงไฟล์ cleaned_predictions.csv")

# ✅ เชื่อมต่อฐานข้อมูล MySQL
try:
    print("🔗 กำลังเชื่อมต่อกับฐานข้อมูล ...")
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        autocommit=True
    )
    cursor = conn.cursor()
    print("✅ เชื่อมต่อฐานข้อมูลสำเร็จ!")

    # ✅ ฟังก์ชันแปลง NaN เป็น None ก่อน Insert
    def convert_nan_to_none(data_list):
        return [[None if (isinstance(x, float) and np.isnan(x)) else x for x in row] for row in data_list]

    # ✅ แปลง `NaN` → `None` สำหรับ MySQL
    predictions_values = convert_nan_to_none(predictions_df[['Date', 'StockSymbol', 'PredictionClose', 'PredictionTrend']].values.tolist())

    # ✅ ตัวอย่างคำสั่ง SQL Insert
    insert_query = """
    INSERT INTO StockDetail (Date, StockSymbol, PredictionClose, PredictionTrend)
    VALUES (%s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE 
        PredictionClose=COALESCE(VALUES(PredictionClose), PredictionClose),
        PredictionTrend=COALESCE(VALUES(PredictionTrend), PredictionTrend);
    """

    # ✅ บันทึกข้อมูลลงฐานข้อมูล
    cursor.executemany(insert_query, predictions_values)
    print(f"✅ บันทึกข้อมูลลง StockDetail: {len(predictions_values)} รายการ")

except mysql.connector.Error as err:
    print(f"❌ เกิดข้อผิดพลาด: {err}")

finally:
    cursor.close()
    conn.close()
    print("🔹 ปิดการเชื่อมต่อฐานข้อมูลแล้ว")
