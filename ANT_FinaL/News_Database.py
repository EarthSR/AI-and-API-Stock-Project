import pandas as pd
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv
import sys
import os

# ✅ ป้องกัน UnicodeEncodeError (ข้ามอีโมจิที่ไม่รองรับ)
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")


# ✅ โหลดตัวแปรจาก .env
load_dotenv()

# 🔹 ตั้งค่าพาธไฟล์ CSV
CSV_FILE_PATH = "D:\\Stock_Project\\AI-and-API-Stock-Project\\Finbert\\news_with_sentiment_gpu.csv"

# 🔹 โหลดข้อมูลจาก news_with_sentiment_gpu.csv
print("📥 กำลังโหลดไฟล์ news_with_sentiment_gpu.csv ...")
df_new = pd.read_csv(CSV_FILE_PATH)
print(f"✅ โหลดสำเร็จ! จำนวนข่าวทั้งหมด: {len(df_new)}")

# 🔹 เปลี่ยนชื่อคอลัมน์ให้ตรงกับ Database
column_mapping = {
    "title": "Title",
    "description": "Content",
    "date": "PublishedDate",
    "Sentiment": "Sentiment",
    "Confidence": "ConfidenceScore",
    "link": "URL",
    "Source": "Source"
}
df_new.rename(columns=column_mapping, inplace=True)

# 🔹 ลบข่าวที่มีค่า NaN ออกก่อน
df_new.dropna(inplace=True)

# 🔹 แปลง PublishedDate เป็น datetime และกรองข่าวเก่าออก
df_new["PublishedDate"] = pd.to_datetime(df_new["PublishedDate"], errors="coerce")
df_new = df_new[df_new["PublishedDate"] >= "2018-01-01"]
print(f"✅ คงเหลือข่าวทั้งหมดหลังลบ NaN และข่าวเก่า: {len(df_new)} รายการ")

# 🔹 จำกัดความยาว `URL` ไม่ให้เกิน `VARCHAR(255)`
df_new["URL"] = df_new["URL"].astype(str).str[:255]

# 🔹 เชื่อมต่อกับฐานข้อมูล MySQL
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

    # ✅ ใช้ `INSERT IGNORE` เพื่อลดปัญหาข่าวซ้ำ
    insert_query = """
    INSERT IGNORE INTO News (Title, Content, PublishedDate, Sentiment, ConfidenceScore, URL, Source)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    # ✅ เช็คข่าวที่ยังไม่ได้บันทึกลงฐานข้อมูล
    cursor.execute("SELECT URL FROM News")
    existing_urls = set(url[0] for url in cursor.fetchall())

    # ✅ คัดกรองข่าวที่ยังไม่ได้บันทึก (ข้ามข่าวที่ซ้ำ)
    df_new = df_new[~df_new["URL"].isin(existing_urls)]
    print(f"🚀 ข่าวใหม่ที่ยังไม่เคยบันทึก: {len(df_new)} รายการ")

    # ✅ เตรียมข้อมูลสำหรับเพิ่มลง Database
    data_to_insert = df_new[["Title", "Content", "PublishedDate", "Sentiment", "ConfidenceScore", "URL", "Source"]].values.tolist()

    # 🔹 บันทึกทีละ 500 แถว ป้องกัน Timeout
    batch_size = 500
    for i in range(0, len(data_to_insert), batch_size):
        batch = data_to_insert[i:i + batch_size]
        cursor.executemany(insert_query, batch)
        print(f"✅ บันทึกข้อมูล {i + len(batch)} / {len(data_to_insert)} ข่าว")

    print(f"🎯 บันทึกข้อมูลลงฐานข้อมูลสำเร็จ! ({len(data_to_insert)} รายการ)")

except mysql.connector.Error as err:
    print(f"❌ เกิดข้อผิดพลาด: {err}")

finally:
    if "conn" in locals() and conn.is_connected():
        cursor.close()
        conn.close()
        print("🔹 ปิดการเชื่อมต่อฐานข้อมูลแล้ว")
