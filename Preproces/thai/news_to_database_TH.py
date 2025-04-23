import pandas as pd
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv
import sys
import os
from tqdm import tqdm

# ✅ ป้องกัน UnicodeEncodeError (รองรับภาษาไทย/ข้ามอีโมจิ)
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# ✅ โหลด .env สำหรับการเชื่อมต่อ MySQL
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.env')
load_dotenv(path)

# ✅ กำหนดข้อมูลเชื่อมต่อฐานข้อมูล
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

if not all(db_config.values()):
    print("❌ กรุณาตรวจสอบตัวแปรสำหรับการเชื่อมต่อกับฐานข้อมูล MySQL")
    sys.exit(1)

# ✅ ฟังก์ชันสำหรับจัดการ NaN
def safe_value(val, default=None):
    return val if pd.notna(val) else default

try:
    # ✅ เชื่อมต่อ MySQL
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # ✅ โหลดไฟล์ข่าว
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'thai', 'News', 'Thai_News_Hybrid.csv')
    df_new = pd.read_csv(file_path)
    df_new = df_new[df_new['title'].notna()]

    # ✅ เปลี่ยนชื่อคอลัมน์ให้ตรงกับฐานข้อมูล
    column_mapping = {
        "title": "Title",
        "description": "Content",
        "date": "PublishedDate",
        "Sentiment": "Sentiment",
        "Confidence": "ConfidenceScore",
        "link": "URL",
        "Source": "Source",
        "image": "Img"
    }
    df_new.rename(columns=column_mapping, inplace=True)

    # ✅ วนลูปบันทึกข่าว + หุ้น พร้อมแถบสถานะ
    for _, row in tqdm(df_new.iterrows(), total=len(df_new), desc="📰 Bangkokpost news"):
        url = safe_value(row.get('URL'), '')
        cursor.execute("SELECT NewsID FROM News WHERE URL = %s", (url,))
        existing = cursor.fetchone()
        if existing:
            print(f"⚠️ ข่าวซ้ำ (ไม่บันทึกซ้ำ): {url}")
            continue

        # ✅ Insert ข่าวหลัก
        sql_news = """
            INSERT INTO News (Title, Source, Sentiment, ConfidenceScore, PublishedDate, Content, URL, Img)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        values_news = (
            safe_value(row.get('Title')),
            safe_value(row.get('Source')),
            safe_value(row.get('Sentiment')),
            round(float(safe_value(row.get('ConfidenceScore'), 0.0)), 2),
            safe_value(row.get('PublishedDate')),
            safe_value(row.get('Content')),
            url,
            safe_value(row.get('Img'))
        )

        cursor.execute(sql_news, values_news)
        news_id = cursor.lastrowid

        # ✅ Insert หุ้นที่เกี่ยวข้อง
        matched_stock = safe_value(row.get('MatchedStock'), '')
        if isinstance(matched_stock, str) and matched_stock.strip():
            for stock in matched_stock.split(", "):
                sql_stock = "INSERT INTO NewsStock (NewsID, StockSymbol) VALUES (%s, %s)"
                cursor.execute(sql_stock, (news_id, stock))

    # ✅ Commit เมื่อเสร็จทั้งหมด
    conn.commit()
    print("✅ บันทึกข่าวและหุ้นที่เกี่ยวข้องเรียบร้อยแล้ว")

finally:
    if 'cursor' in locals(): cursor.close()
    if 'conn' in locals(): conn.close()

