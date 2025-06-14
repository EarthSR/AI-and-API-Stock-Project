import os
import urllib.parse
import pandas as pd
import requests
from datetime import datetime, timedelta
import mysql.connector
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import sys
from dotenv import load_dotenv, find_dotenv
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv(find_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.env')))
# 🔹 ตั้งค่าการเชื่อมต่อฐานข้อมูล
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME]):
    raise ValueError("❌ ขาดค่าการตั้งค่าฐานข้อมูลในไฟล์ .env")

conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME,
    autocommit=True
)
cursor = conn.cursor()
print("✅ เชื่อมต่อฐานข้อมูลสำเร็จ!")

def get_latest_news_date_from_db():
    query = "SELECT MAX(PublishedDate) FROM News WHERE Source = 'BangkokPost'"
    cursor.execute(query)
    latest_date = cursor.fetchone()[0]
    if latest_date:
        latest_date = latest_date.date()
        print(f"🗓️ ข่าวล่าสุดในฐานข้อมูลคือวันที่: {latest_date}")
        return latest_date
    else:
        print("⚠️ ไม่มีข่าวในฐานข้อมูล เริ่มดึงข่าวย้อนหลัง 7 วัน")
        return datetime.now().date() - timedelta(days=7)

latest_date = get_latest_news_date_from_db()

NEWS_CATEGORIES = {
    "Business": "https://search.bangkokpost.com/search/result?category=news&sort=newest&rows=10&refinementFilter=AQhidXNpbmVzcwxjaGFubmVsYWxpYXMBAV4BJA%3D%3D",
    "Investment": "https://search.bangkokpost.com/search/result?category=news&sort=newest&rows=10&refinementFilter=AQppbnZlc3RtZW50DGNoYW5uZWxhbGlhcwEBXgEk",
}

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'thai')

News_FOLDER = os.path.join(path, "News")
os.makedirs(News_FOLDER, exist_ok=True)
RAW_CSV_FILE = os.path.join(News_FOLDER, "Thai_News.csv")

def setup_driver():
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--headless=new')
    service = Service(os.path.join(os.path.dirname(os.path.abspath(__file__)),'chromedriver.exe'))  # เปลี่ยน path ตรงนี้ถ้าจำเป็น
    return webdriver.Chrome(service=service, options=options)

def parse_and_format_datetime(date_str):
    date_formats = ["%d %b %Y at %H:%M", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"]
    for date_format in date_formats:
        try:
            parsed_date = datetime.strptime(date_str.strip(), date_format)
            return parsed_date.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return None

def fetch_news_content(real_link):
    try:
        response = requests.get(real_link, timeout=5)
        news_soup = BeautifulSoup(response.content, 'html.parser')

        date_tag = news_soup.find('div', class_='article-info--col')
        date = date_tag.find('p', string=lambda x: x and 'PUBLISHED :' in x).get_text(strip=True).replace('PUBLISHED :', '').strip() if date_tag else 'No Date'

        # ดึงรูปภาพ - วิธีที่ 1
        img_url = "No Image"
        box_img = news_soup.find('div', class_='box-img')
        if box_img and box_img.find('figure') and box_img.find('figure').find('img'):
            img_url = box_img.find('figure').find('img').get('src')
        
        # ถ้าวิธีที่ 1 ไม่พบ ลองวิธีที่ 2
        if img_url == "No Image":
            article_content = news_soup.find('div', class_='article-content')
            if article_content:
                box_img = article_content.find('div', class_='box-img')
                if box_img and box_img.find('figure') and box_img.find('figure').find('img'):
                    img_url = box_img.find('figure').find('img').get('src')

        # ถ้าวิธีที่ 2 ยังไม่พบ ลองวิธีที่ 3
        if img_url == "No Image":
            img_tags = news_soup.find_all('img', class_='img-fluid')
            if img_tags and len(img_tags) > 0:
                for img in img_tags:
                    src = img.get('src', '')
                    if 'content' in src and not 'icon' in src.lower():
                        img_url = src
                        break

        content_div = news_soup.find('div', class_='article-content')
        paragraphs = content_div.find_all('p') if content_div else []
        full_content = '\n'.join([p.get_text(strip=True) for p in paragraphs])

        return date, full_content.replace(',', '').replace('""', ''), img_url
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content: {e}")
        return 'No Date', 'Content not found', 'No Image'

def scrape_news_from_category(category_name, url):
    print(f" [START] ดึงข่าวจาก {category_name}")
    driver = setup_driver()
    news_data = []

    try:
        driver.get(url)

        while True:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'mk-listnew--title')))
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            articles = soup.find_all('div', class_='mk-listnew--title')

            if not articles:
                break

            for article in articles:
                try:
                    title_tag = article.find('h3').find('a')
                    title = title_tag.get_text(strip=True)
                    link = title_tag['href']

                    real_link = urllib.parse.parse_qs(urllib.parse.urlparse(link).query).get('href', [link])[0] if 'track/visitAndRedirect' in link else link
                    date, full_content, img_url = fetch_news_content(real_link)
                    formatted_datetime = parse_and_format_datetime(date)

                    if formatted_datetime and datetime.strptime(formatted_datetime, "%Y-%m-%d %H:%M:%S").date() <= latest_date:
                        print(f"[STOP] พบข่าวที่มีอยู่แล้ว ({latest_date}), หยุดดึง {category_name}")
                        return news_data

                    news_data.append({
                        "title": title,
                        "date": formatted_datetime,
                        "link": real_link,
                        "description": full_content,
                        "image": img_url
                    })
                except Exception as e:
                    print(f"Error processing article: {e}")
                    continue
            break  # ถ้าต้องการดึงแค่หน้าเดียว ลบ break นี้ออกถ้าอยากไล่หลายหน้า

    finally:
        driver.quit()

    return news_data

def ensure_csv_file_exists():
    """สร้างไฟล์ CSV ว่างพร้อม header ถ้ายังไม่มี"""
    if not os.path.exists(RAW_CSV_FILE):
        # สร้างไฟล์ CSV ว่างพร้อม header
        empty_df = pd.DataFrame(columns=["title", "date", "link", "description", "image"])
        empty_df.to_csv(RAW_CSV_FILE, index=False, encoding='utf-8')
        print(f"[CREATED] สร้างไฟล์ CSV ใหม่: {RAW_CSV_FILE}")

def scrape_all_news():
    print(" [START] เริ่มต้นดึงข่าว...")
    all_news_data = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(scrape_news_from_category, name, url): name for name, url in NEWS_CATEGORIES.items()}
        for future in futures:
            result = future.result()
            all_news_data.extend(result)

    # ตรวจสอบว่ามีไฟล์ CSV อยู่หรือไม่ ถ้าไม่มีให้สร้าง
    ensure_csv_file_exists()

    if len(all_news_data) > 0:
        df = pd.DataFrame(all_news_data)
        df.to_csv(RAW_CSV_FILE, mode='a', header=False, index=False, encoding='utf-8')
        print(f"[SAVED] ข่าวทั้งหมด {len(all_news_data)} ข่าวถูกบันทึกเรียบร้อย!")
        return {
            "status": "success",
            "count": len(all_news_data),
            "file_path": RAW_CSV_FILE,
            "message": f"ดึงข่าวใหม่ {len(all_news_data)} ข่าว"
        }
    else:
        print("⚠️ ไม่มีข่าวใหม่ให้บันทึก!")
        return {
            "status": "no_new_data",
            "count": 0,
            "file_path": RAW_CSV_FILE,
            "message": "ไม่มีข่าวใหม่ แต่ไฟล์ CSV พร้อมใช้งาน"
        }

def get_scraping_result():
    """ฟังก์ชันหลักที่จะเรียกใช้ - รับประกันว่าจะมี output เสมอ"""
    try:
        result = scrape_all_news()
        
        # ตรวจสอบว่าไฟล์ CSV มีอยู่และพร้อมใช้งาน
        if os.path.exists(RAW_CSV_FILE):
            file_size = os.path.getsize(RAW_CSV_FILE)
            result["file_exists"] = True
            result["file_size"] = file_size
        else:
            result["file_exists"] = False
            result["file_size"] = 0
            
        return result
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        # แม้เกิดข้อผิดพลาดก็ยังคืนค่า output
        ensure_csv_file_exists()
        return {
            "status": "error",
            "count": 0,
            "file_path": RAW_CSV_FILE,
            "message": f"เกิดข้อผิดพลาด: {str(e)}",
            "file_exists": os.path.exists(RAW_CSV_FILE),
            "file_size": os.path.getsize(RAW_CSV_FILE) if os.path.exists(RAW_CSV_FILE) else 0
        }

if __name__ == "__main__":
    result = get_scraping_result()
    print(f"\n📊 ผลลัพธ์การดึงข่าว:")
    print(f"   สถานะ: {result['status']}")
    print(f"   จำนวนข่าว: {result['count']}")
    print(f"   ไฟล์: {result['file_path']}")
    print(f"   ข้อความ: {result['message']}")
    print(f"   ไฟล์มีอยู่: {result['file_exists']}")
    print(f"   ขนาดไฟล์: {result['file_size']} bytes")