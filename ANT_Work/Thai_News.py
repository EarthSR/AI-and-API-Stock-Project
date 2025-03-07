import os
import urllib.parse
import pandas as pd
import requests
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
from concurrent.futures import ThreadPoolExecutor

# ✅ ตรวจสอบระดับของโฟลเดอร์ (ปรับ `..` ตามตำแหน่งของไฟล์)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 

# 🔹 ตั้งค่าหมวดหมู่ข่าวที่ต้องการดึง
NEWS_CATEGORIES = {
    "Business": "https://search.bangkokpost.com/search/result?publishedDate=&q=&category=news&sort=newest&rows=10&refinementFilter=AQhidXNpbmVzcwxjaGFubmVsYWxpYXMBAV4BJA%3D%3D",
    "Investment": "https://search.bangkokpost.com/search/result?publishedDate=&q=&category=news&sort=newest&rows=10&refinementFilter=AQppbnZlc3RtZW50DGNoYW5uZWxhbGlhcwEBXgEk",
    "Motoring": "https://search.bangkokpost.com/search/result?publishedDate=&q=&category=news&sort=newest&rows=10&refinementFilter=AQhtb3RvcmluZwxjaGFubmVsYWxpYXMBAV4BJA%3D%3D",
    "General": "https://search.bangkokpost.com/search/result?publishedDate=&q=&category=news&sort=newest&rows=10&refinementFilter=AQdnZW5lcmFsDGNoYW5uZWxhbGlhcwEBXgEk"
}

# 🔹 ไฟล์ CSV ที่ใช้เก็บข่าวดิบ และไฟล์ที่ใช้เก็บข่าวที่สะอาด
RAW_CSV_FILE = os.path.join(BASE_DIR, "BangkokPost_Folder", "Thai_News.csv")
CLEAN_CSV_FILE = os.path.join(BASE_DIR, "BangkokPost_Folder", "Thai_News.csv")

# 🔹 โหลดข่าวที่เคยบันทึกไว้
def load_existing_titles():
    if os.path.exists(RAW_CSV_FILE):
        try:
            df_existing = pd.read_csv(RAW_CSV_FILE, usecols=['title'])
            return set(df_existing['title'].astype(str))
        except Exception:
            return set()
    return set()

# 🔹 ตั้งค่า Selenium Driver
def setup_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--blink-settings=imagesEnabled=false')

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

# 🔹 ดึงเนื้อหาข่าว
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

# 🔹 แปลงรูปแบบวันที่ให้ตรงกัน
def parse_and_format_date(date_str):
    date_formats = [
        "%d %b %Y at %H:%M",
        "%a, %b %d, %Y, %I:%M %p",
        "%Y-%m-%d %H:%M:%S",
        "%b %d, %Y at %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%B %d, %Y"
    ]

    for date_format in date_formats:
        try:
            parsed_date = datetime.strptime(date_str.strip(), date_format)
            return parsed_date.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

    print(f"⚠️ Error parsing date: {date_str}")
    return None

# 🔹 ดึงข่าวทั้งหมด
def scrape_all_news():
    existing_titles = load_existing_titles()
    global_news_count = [0]

    print("🚀 [START] เริ่มต้นดึงข่าว...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(scrape_news_from_category, name, url, existing_titles, global_news_count): name for name, url in NEWS_CATEGORIES.items()}
        for future in futures:
            future.result()

    print("\n🎯 [DONE] ดึงข่าวเสร็จสมบูรณ์")

    clean_and_process_data()

# 🔹 ดึงข่าวจากหมวดหมู่ที่กำหนด
def scrape_news_from_category(category_name, url, existing_titles, global_news_count):
    print(f"🔍 [START] ดึงข่าวจาก {category_name}")

    driver = setup_driver()
    driver.get(url)
    news_data = []
    total_saved = 0

    while True:
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'mk-listnew--title')))
        except Exception:
            print(f"❌ [STOP] {category_name} หมวดนี้ไม่โหลดข่าว (Timeout)")
            break

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        articles = soup.find_all('div', class_='mk-listnew--title')

        if not articles:
            print(f"❌ [STOP] ไม่พบข่าวใน {category_name}")
            break

        for article in articles:
            try:
                title_tag = article.find('h3').find('a')
                title = title_tag.get_text(strip=True)
                link = title_tag['href']

                if title in existing_titles:
                    print(f"⏩ [SKIP] {category_name} พบข่าวซ้ำ -> หยุดดึงข่าว")
                    driver.quit()
                    return total_saved

                real_link = urllib.parse.parse_qs(urllib.parse.urlparse(link).query).get('href', [link])[0] if 'track/visitAndRedirect' in link else link
                date, full_content = fetch_news_content(real_link)

                news_data.append({"title": title.replace(',', ''), "date": date, "link": real_link, "description": full_content})
                existing_titles.add(title)
                global_news_count[0] += 1
                total_saved += 1

                if len(news_data) >= 10:
                    save_news_to_csv(news_data)
                    news_data = []

            except Exception:
                continue

        next_page = soup.find('a', string='Next')
        if next_page and 'href' in next_page.attrs:
            driver.get(next_page['href'])
        else:
            break

    driver.quit()

    if news_data:
        save_news_to_csv(news_data)

    return total_saved

# 🔹 บันทึกข่าวลง CSV
def save_news_to_csv(news_data):
    df = pd.DataFrame(news_data)
    df.to_csv(RAW_CSV_FILE, mode='a', header=not os.path.exists(RAW_CSV_FILE), index=False)
    print(f"💾 [SAVE] บันทึกข่าว {len(news_data)} ข่าว")

# 🔹 ทำความสะอาดข้อมูล
CUTOFF_DATE = datetime(2018, 1, 1)

def clean_and_process_data():
    df = pd.read_csv(RAW_CSV_FILE, encoding='utf-8')

    df['date'] = df['date'].apply(parse_and_format_date)
    df = df.dropna(subset=['date'])
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= CUTOFF_DATE]
    df = df.sort_values(by='date', ascending=False)

    df.to_csv(CLEAN_CSV_FILE, index=False, encoding='utf-8')
    print(f"✅ [CLEANED] ไฟล์ข่าวที่ทำความสะอาดแล้วถูกบันทึกที่: {CLEAN_CSV_FILE}")

# 🔹 เรียกใช้งานฟังก์ชันหลัก
if __name__ == "__main__":
    scrape_all_news()
