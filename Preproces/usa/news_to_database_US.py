# -*- coding: utf-8 -*-
import os, sys, io
import pandas as pd
from tqdm import tqdm
import mysql.connector
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ====== ENV / DB CONFIG ======
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.env')
load_dotenv(path)
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "charset": "utf8mb4"
}
if not all(db_config.values()):
    print("❌ กรุณาตรวจสอบตัวแปรสำหรับการเชื่อมต่อกับฐานข้อมูล MySQL")
    sys.exit(1)

# ====== TUNING ======
CONNECT_TIMEOUT = 120
SESSION_WAIT_TIMEOUT = 28800
SESSION_INTERACTIVE_TIMEOUT = 28800
SESSION_NET_READ_TIMEOUT = 120
SESSION_NET_WRITE_TIMEOUT = 120

BATCH_NEWS = 2000        # ใส่ข่าวครั้งละกี่เรคคอร์ด
BATCH_STOCK = 5000       # ใส่สัมพันธ์ NewsStock ครั้งละกี่แถว
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'usa', 'News', 'USA_News_Hybrid.csv')

# ====== HELPERS ======
def safe(v, default=None):
    return v if pd.notna(v) else default

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# ====== LOAD CSV (อ่านเฉพาะคอลัมน์ที่ใช้) ======
usecols = ["title","description","date","Sentiment","Confidence","link","Source","image","MatchedStock"]
df = pd.read_csv(CSV_PATH, usecols=[c for c in usecols if c in pd.read_csv(CSV_PATH, nrows=0).columns])
df = df[df["title"].notna()]
df = df[df["link"].notna()]  # ต้องมี URL
# map ชื่อคอลัมน์ -> ตาราง
df = df.rename(columns={
    "title": "Title",
    "description": "Content",
    "date": "PublishedDate",
    "Sentiment": "Sentiment",
    "Confidence": "ConfidenceScore",
    "link": "URL",
    "Source": "Source",
    "image": "Img"
})
# เดดูกระดับไฟล์ (URL ซ้ำในไฟล์เดียวกันเอาแถวแรก)
df = df.sort_index().drop_duplicates(subset=["URL"], keep="first")

# เตรียม list of dict เร็วกว่า iterrows
rows = df.to_dict(orient="records")

# ====== CONNECT DB ======
conn = mysql.connector.connect(**db_config, connection_timeout=CONNECT_TIMEOUT)
cur = conn.cursor()

# ขยาย timeout เฉพาะ session นี้
cur.execute(f"SET SESSION wait_timeout = {SESSION_WAIT_TIMEOUT}")
cur.execute(f"SET SESSION interactive_timeout = {SESSION_INTERACTIVE_TIMEOUT}")
cur.execute(f"SET SESSION net_read_timeout = {SESSION_NET_READ_TIMEOUT}")
cur.execute(f"SET SESSION net_write_timeout = {SESSION_NET_WRITE_TIMEOUT}")

# (ทางเลือก) สร้างดัชนีธรรมดาเพื่อเร่ง SELECT IN (...) ถ้ายังไม่มี (ignore error ถ้ามีอยู่แล้ว)
for ddl in [
    "CREATE INDEX idx_news_url ON News (URL(255))",
    "CREATE INDEX idx_newsstk ON NewsStock (NewsID, StockSymbol)"
]:
    try:
        cur.execute(ddl)
        conn.commit()
    except Exception:
        conn.rollback()

# ====== SQL ======
SQL_INSERT_NEWS = """
INSERT INTO News (Title, Source, Sentiment, ConfidenceScore, PublishedDate, Content, URL, Img)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
"""
# ใช้ SELECT mapping แบบเป็นก้อน เพื่อตัด per-row query
def select_news_ids_by_urls(urls):
    if not urls:
        return {}
    placeholders = ",".join(["%s"] * len(urls))
    cur.execute(f"SELECT URL, NewsID FROM News WHERE URL IN ({placeholders})", urls)
    return dict(cur.fetchall())  # {URL: NewsID}

SQL_INSERT_NEWS_STOCK = "INSERT INTO NewsStock (NewsID, StockSymbol) VALUES (%s, %s)"

inserted_news = 0
inserted_links = 0

# เพื่อกันแทรก NewsStock ซ้ำใน "รันนี้" (แม้ DB จะยังไม่มี UNIQUE)
seen_pairs = set()

# ====== MAIN PIPELINE ======
for batch in tqdm(list(chunked(rows, BATCH_NEWS)), desc="🚀 Upserting News (batch)"):
    urls = [safe(r.get("URL"), "") for r in batch if safe(r.get("URL"), "")]
    # หา URL ที่มีอยู่แล้วในฐาน
    existing_map = select_news_ids_by_urls(urls)

    # เตรียม new_rows เฉพาะ URL ที่ยังไม่มี
    new_rows = []
    for r in batch:
        url = safe(r.get("URL"), "")
        if not url or url in existing_map:
            continue
        conf = safe(r.get("ConfidenceScore"), 0.0)
        try:
            conf = round(float(conf), 2)
        except Exception:
            conf = 0.0
        new_rows.append((
            safe(r.get("Title")),
            safe(r.get("Source")),
            safe(r.get("Sentiment")),
            conf,
            safe(r.get("PublishedDate")),
            safe(r.get("Content")),
            url,
            safe(r.get("Img"))
        ))

    # ใส่ข่าวใหม่แบบ batch
    if new_rows:
        for sub in chunked(new_rows, 1000):
            cur.executemany(SQL_INSERT_NEWS, sub)
            conn.commit()
        inserted_news += len(new_rows)

    # ดึง NewsID map อีกรอบ (ครอบคลุมทั้งที่มีอยู่แล้ว + ที่เพิ่งแทรก)
    news_map = select_news_ids_by_urls(urls)

    # รวบรวม NewsStock เป็น batch เดียว
    stock_pairs = []
    for r in batch:
        url = safe(r.get("URL"), "")
        nid = news_map.get(url)
        if not nid:
            continue
        ms = safe(r.get("MatchedStock"), "")
        if isinstance(ms, str) and ms.strip():
            for s in [x.strip() for x in ms.split(",") if x.strip()]:
                key = (nid, s)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                stock_pairs.append(key)

    # ใส่ NewsStock แบบ batch
    if stock_pairs:
        for sub in chunked(stock_pairs, BATCH_STOCK):
            cur.executemany(SQL_INSERT_NEWS_STOCK, sub)
            conn.commit()
        inserted_links += len(stock_pairs)

cur.close()
conn.close()

print(f"✅ เสร็จแล้ว: เพิ่มข่าวใหม่ {inserted_news:,} เรคคอร์ด | เพิ่มลิงก์หุ้น {inserted_links:,} แถว (เดดูกในรันนี้)")
