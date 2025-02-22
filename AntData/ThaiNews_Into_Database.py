import os
import time
import urllib.parse
import pandas as pd
import requests
import pymysql
import torch
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from dotenv import load_dotenv

# ✅ โหลดค่าจากไฟล์ .env
load_dotenv()
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT"))
}

NEWS_CATEGORIES = {
    "Investment": "https://search.bangkokpost.com/search/result?publishedDate=&q=&category=news&sort=newest&rows=10&refinementFilter=AQppbnZlc3RtZW50DGNoYW5uZWxhbGlhcwEBXgEk",
    "Motoring": "https://search.bangkokpost.com/search/result?publishedDate=&q=&category=news&sort=newest&rows=10&refinementFilter=AQhtb3RvcmluZwxjaGFubmVsYWxpYXMBAV4BJA%3D%3D",
    "General": "https://search.bangkokpost.com/search/result?publishedDate=&q=&category=news&sort=newest&rows=10&refinementFilter=AQdnZW5lcmFsDGNoYW5uZWxhbGlhcwEBXgEk"
}

def is_news_exists(title, url):
    """ตรวจสอบว่าข่าวนี้มีอยู่ในฐานข้อมูลแล้วหรือไม่"""
    connection = pymysql.connect(**DB_CONFIG)
    cursor = connection.cursor()
    
    query = "SELECT COUNT(*) FROM News WHERE Title = %s OR URL = %s"
    cursor.execute(query, (title, url))
    result = cursor.fetchone()[0]

    cursor.close()
    connection.close()
    return result > 0  # คืนค่า True ถ้าข่าวมีอยู่แล้ว

def insert_news_to_db(df):
    """บันทึกข้อมูลข่าวลงใน Database table: News โดยไม่บันทึกข่าวซ้ำ"""
    connection = pymysql.connect(**DB_CONFIG)
    cursor = connection.cursor()

    insert_query = """
        INSERT INTO News (Title, URL, PublishedDate, Content, Source, Sentiment, ConfidenceScore)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    new_entries = 0
    skipped_entries = 0

    try:
        for _, row in df.iterrows():
            if is_news_exists(row["Title"], row["URL"]):
                print(f"⚠️ ข่าว '{row['Title']}' มีอยู่แล้วในฐานข้อมูล! ข้ามการบันทึก...")
                skipped_entries += 1
                continue
            
            cursor.execute(insert_query, (
                row["Title"], row["URL"], row["PublishedDate"], row["Content"],
                row["Source"], row["Sentiment"], row["ConfidenceScore"]
            ))
            new_entries += 1

        connection.commit()
        print(f"✅ บันทึกข่าวใหม่ {new_entries} รายการลงใน Database เรียบร้อยแล้ว!")
        print(f"⚠️ ข่าวที่ถูกข้ามเพราะซ้ำ: {skipped_entries} รายการ")

    except pymysql.Error as e:
        print(f"❌ เกิดข้อผิดพลาดในการบันทึกข้อมูล: {e}")
        connection.rollback()

    finally:
        cursor.close()
        connection.close()

chrome_options = Options()
chrome_options.add_argument('--headless')  
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--ignore-certificate-errors')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--blink-settings=imagesEnabled=false')  

def scrape_news(category, url, driver):
    """ ดึงข่าวจากหมวดที่กำหนด แล้วเก็บไว้ใน DataFrame """
    print(f"🔍 กำลังดึงข่าวจาก {category}...")
    driver.get(url)

    news_data = []
    while True:
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'mk-listnew--title'))
            )
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            articles = soup.find_all('div', class_='mk-listnew--title')

            if not articles:
                print(f"❌ ไม่พบข่าวในหมวด {category}")
                break

            stop_scraping = False  

            for article in articles:
                try:
                    title_tag = article.find('h3').find('a')
                    title = title_tag.get_text(strip=True)
                    link = title_tag['href']

                    response = requests.get(link, timeout=5)
                    news_soup = BeautifulSoup(response.content, 'html.parser')

                    date_tag = news_soup.find('div', class_='article-info--col')
                    date_raw = date_tag.find('p', string=lambda x: x and 'PUBLISHED :' in x).get_text(strip=True).replace('PUBLISHED :', '').strip() if date_tag else 'No Date'

                    try:
                        published_date = datetime.strptime(date_raw, "%d %b %Y at %H:%M").strftime("%Y-%m-%d")
                    except Exception:
                        continue

                    if published_date < datetime.now().strftime("%Y-%m-%d"):
                        print(f"✅ หยุดดึงข่าว {category} เพราะเป็นข่าวเก่า")
                        stop_scraping = True
                        break  

                    news_data.append({
                        "Title": title,
                        "URL": link,          
                        "PublishedDate": published_date,    
                        "Source": "Bangkok Post"
                    })

                except Exception:
                    continue

            if stop_scraping:
                break

        except Exception:
            print(f"❌ ดึงข่าว {category} ล้มเหลว")
            break

    return pd.DataFrame(news_data)

if __name__ == '__main__':
    # ✅ โหลดโมเดล FinBERT **แค่ครั้งเดียว**
    print("🔍 กำลังโหลดโมเดล FinBERT...")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").to("cpu")
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    finbert_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1, truncation=True, max_length=512)
    print("✅ โหลดโมเดลเสร็จแล้ว!")

    # ✅ เก็บวันที่ล่าสุดที่รัน
    last_run_date = None  

    # ✅ เริ่ม loop รันตลอดเวลา
    while True:
        now = datetime.now()
        if now.hour == 0 and (last_run_date is None or last_run_date != now.date()):
            print("✅ เริ่มดึงข่าวใหม่ประจำวัน...")
            last_run_date = now.date()
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            news_dataframes = [scrape_news(category, url, driver) for category, url in NEWS_CATEGORIES.items()]
            driver.quit()
            df_merged = pd.concat(news_dataframes, ignore_index=True)

            if not df_merged.empty:
                print("✅ ข่าวรวมทั้งหมดพร้อมทำ Sentiment Analysis")
                sentiment_results = finbert_pipeline(df_merged["Title"].tolist())
                df_merged["Sentiment"] = [r["label"] for r in sentiment_results]
                df_merged["ConfidenceScore"] = [r["score"] for r in sentiment_results]
                insert_news_to_db(df_merged)

        time.sleep(60)  # ✅ เช็คทุก 60 วินาที
