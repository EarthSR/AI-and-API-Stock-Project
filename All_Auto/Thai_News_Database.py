import os
import urllib.parse
import pandas as pd
import requests
from datetime import datetime, timedelta
import mysql.connector  # ✅ เพิ่มการเชื่อมต่อฐานข้อมูล
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
from concurrent.futures import ThreadPoolExecutor
import sys
from dotenv import load_dotenv

# ✅ ป้องกัน UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# ✅ โหลดค่าจาก .env และตรวจสอบว่าถูกต้องหรือไม่
load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME]):
    raise ValueError("❌ ขาดค่าการตั้งค่าฐานข้อมูลในไฟล์ .env")

# ✅ เชื่อมต่อฐานข้อมูล
conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME,
    autocommit=True
)
cursor = conn.cursor()
print("✅ เชื่อมต่อฐานข้อมูลสำเร็จ!")

# ✅ ดึงวันที่ล่าสุดจากฐานข้อมูล
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

# ✅ ตั้งค่าหมวดหมู่ข่าวที่ต้องการดึง
NEWS_CATEGORIES = {
    "Business": "https://search.bangkokpost.com/search/result?category=news&sort=newest&rows=10&refinementFilter=AQhidXNpbmVzcwxjaGFubmVsYWxpYXMBAV4BJA%3D%3D",
    "Investment": "https://search.bangkokpost.com/search/result?category=news&sort=newest&rows=10&refinementFilter=AQppbnZlc3RtZW50DGNoYW5uZWxhbGlhcwEBXgEk",
}

# ✅ ใช้โฟลเดอร์ปัจจุบัน
CURRENT_DIR = os.getcwd()
NEWS_FOLDER = os.path.join(CURRENT_DIR, "News")
os.makedirs(NEWS_FOLDER, exist_ok=True)

# ✅ ไฟล์ CSV
RAW_CSV_FILE = os.path.join(NEWS_FOLDER, "Thai_News.csv")

# ✅ ตั้งค่า Selenium Driver
def setup_driver():
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--headless=new')

    chromedriver_path = ChromeDriverManager().install()
    service = Service(chromedriver_path)

    return webdriver.Chrome(service=service, options=options)

# ✅ แปลงรูปแบบวันที่
def parse_and_format_datetime(date_str):
    date_formats = ["%d %b %Y at %H:%M", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"]
    for date_format in date_formats:
        try:
            parsed_date = datetime.strptime(date_str.strip(), date_format)
            return parsed_date.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return None

# ✅ ดึงเนื้อหาข่าว
def fetch_news_content(real_link):
    try:
        response = requests.get(real_link, timeout=5)
        news_soup = BeautifulSoup(response.content, 'html.parser')

        date_tag = news_soup.find('div', class_='article-info--col')
        date = date_tag.find('p', string=lambda x: x and 'PUBLISHED :' in x).get_text(strip=True).replace('PUBLISHED :', '').strip() if date_tag else 'No Date'

        content_div = news_soup.find('div', class_='article-content')
        paragraphs = content_div.find_all('p') if content_div else []
        full_content = '\n'.join([p.get_text(strip=True) for p in paragraphs])

        return date, full_content.replace(',', '').replace('""', '')
    except requests.exceptions.RequestException:
        return 'No Date', 'Content not found'

# 🔹 ✅ เพิ่ม Debug เช็คจำนวนข่าวที่ดึงมา
def scrape_all_news():
    print(" [START] เริ่มต้นดึงข่าว...")

    all_news_data = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(scrape_news_from_category, name, url): name for name, url in NEWS_CATEGORIES.items()}
        for future in futures:
            result = future.result()
            all_news_data.extend(result)

    # ✅ เช็คว่ามีข่าวจริงไหม
    print(f"📊 ดึงข่าวทั้งหมดได้ {len(all_news_data)} ข่าว")

    # 🔹 บันทึกทับไฟล์เก่า
    if len(all_news_data) > 0:
        df = pd.DataFrame(all_news_data)
        df.to_csv(RAW_CSV_FILE, index=False, encoding='utf-8')
        print(f"[SAVED] ข่าวทั้งหมด {len(all_news_data)} ข่าวถูกบันทึกเรียบร้อย!")
    else:
        print("⚠️ ไม่มีข่าวให้บันทึก! ตรวจสอบว่าเว็บโหลดถูกต้องหรือไม่")


# ✅ ดึงข่าวจากแต่ละหมวด
def scrape_news_from_category(category_name, url):
    print(f" [START] ดึงข่าวจาก {category_name}")

    driver = setup_driver()
    driver.get(url)
    news_data = []

    while True:
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'mk-listnew--title')))
        except Exception:
            break

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
                date, full_content = fetch_news_content(real_link)
                formatted_datetime = parse_and_format_datetime(date)

                if formatted_datetime and datetime.strptime(formatted_datetime, "%Y-%m-%d %H:%M:%S").date() <= latest_date:
                    print(f"[STOP] พบข่าวที่มีอยู่แล้ว ({latest_date}), หยุดดึง {category_name}")
                    driver.quit()
                    return news_data

                news_data.append({"title": title, "date": formatted_datetime, "link": real_link, "description": full_content})

            except Exception:
                continue

        driver.quit()
        return news_data

# ✅ ฟังก์ชันหลัก
def scrape_all_news():
    print(" [START] เริ่มต้นดึงข่าว...")
    all_news_data = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(scrape_news_from_category, name, url): name for name, url in NEWS_CATEGORIES.items()}
        for future in futures:
            result = future.result()
            all_news_data.extend(result)

    df = pd.DataFrame(all_news_data)
    df.to_csv(RAW_CSV_FILE, index=False, encoding='utf-8')
    print(f"[SAVED] ข่าวทั้งหมด {len(all_news_data)} ข่าวถูกบันทึกเรียบร้อย!")

if __name__ == "__main__":
    scrape_all_news()
