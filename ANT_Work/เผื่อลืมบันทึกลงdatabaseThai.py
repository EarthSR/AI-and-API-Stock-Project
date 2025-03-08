import os
import urllib.parse
import pandas as pd
import requests
from datetime import datetime, timedelta
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

# ✅ ป้องกัน UnicodeEncodeError (ข้ามอีโมจิที่ไม่รองรับ)
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")


# 🔹 ตั้งค่าหมวดหมู่ข่าวที่ต้องการดึง
NEWS_CATEGORIES = {
    "Business": "https://search.bangkokpost.com/search/result?publishedDate=&q=&category=news&sort=newest&rows=10&refinementFilter=AQhidXNpbmVzcwxjaGFubmVsYWxpYXMBAV4BJA%3D%3D",
    "Investment": "https://search.bangkokpost.com/search/result?publishedDate=&q=&category=news&sort=newest&rows=10&refinementFilter=AQppbnZlc3RtZW50DGNoYW5uZWxhbGlhcwEBXgEk",
    "Motoring": "https://search.bangkokpost.com/search/result?publishedDate=&q=&category=news&sort=newest&rows=10&refinementFilter=AQhtb3RvcmluZwxjaGFubmVsYWxpYXMBAV4BJA%3D%3D",
    "General": "https://search.bangkokpost.com/search/result?publishedDate=&q=&category=news&sort=newest&rows=10&refinementFilter=AQdnZW5lcmFsDGNoYW5uZWxhbGlhcwEBXgEk"
}

# 🔹 ไฟล์ CSV
RAW_CSV_FILE = "D:/Stock_Project/AI-and-API-Stock-Project/BangkokPost_Folder/Thai_News.csv"
CLEAN_CSV_FILE = "D:/Stock_Project/AI-and-API-Stock-Project/BangkokPost_Folder/Thai_News.csv"

# 🔹 ตั้งค่าวันที่เมื่อวาน (เริ่มที่ 00:00:00)
yesterday_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)

# 🔹 ตั้งค่า Selenium Driver ให้ใช้ ChromeDriver ล่าสุด
def setup_driver():
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--headless=new')  # ✅ ใช้ headless mode (new API)

    # 🔹 ระบุพาธของ ChromeDriver ที่ติดตั้งใหม่
    chromedriver_path = ChromeDriverManager().install()
    service = Service(chromedriver_path)

    return webdriver.Chrome(service=service, options=options)


# 🔹 แปลงรูปแบบวันที่และเวลา
def parse_and_format_datetime(date_str):
    date_formats = [
        "%d %b %Y at %H:%M",
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

    return None  # ถ้าแปลงไม่ได้

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

# 🔹 ✅ เพิ่ม Debug เช็คจำนวนข่าวที่ดึงมา
def scrape_all_news():
    global yesterday_start
    current_fake_today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    stop_date = datetime(2025, 3, 1, 0, 0, 0)  # ✅ ให้หยุดเมื่อถึง "วันนี้" = 2 มีนาคม 2025

    while current_fake_today >= stop_date:  # ✅ เปลี่ยนจาก ">" เป็น ">=" เพื่อให้ดึงข่าวของวันที่ 1 มีนาคม 2025 ด้วย
        print(f" [START] หลอกระบบว่า 'วันนี้' คือ {current_fake_today.strftime('%Y-%m-%d')}")

        yesterday_start = current_fake_today - timedelta(days=1)

        all_news_data = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(scrape_news_from_category, name, url): name for name, url in NEWS_CATEGORIES.items()}
            for future in futures:
                result = future.result()
                all_news_data.extend(result)

        print(f"📊 ดึงข่าวทั้งหมดได้ {len(all_news_data)} ข่าว")

        if len(all_news_data) > 0:
            df = pd.DataFrame(all_news_data)
            df.to_csv(RAW_CSV_FILE, mode='a', index=False, encoding='utf-8', header=not os.path.exists(RAW_CSV_FILE))
            print(f"[SAVED] ข่าวทั้งหมด {len(all_news_data)} ข่าวถูกบันทึกเรียบร้อย!")
        else:
            print("⚠️ ไม่มีข่าวให้บันทึก! ตรวจสอบว่าเว็บโหลดถูกต้องหรือไม่")

        clean_and_process_data()

        current_fake_today -= timedelta(days=1)  # ✅ ลดวันที่ลง 1 วันเพื่อทำซ้ำ

# 🔹 ดึงข่าวจากแต่ละหมวด
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

                if not formatted_datetime:
                    continue

                news_datetime = datetime.strptime(formatted_datetime, "%Y-%m-%d %H:%M:%S")
                if news_datetime < yesterday_start:
                    print(f"[STOP] พบข่าวเก่ากว่า {yesterday_start} → หยุดดึง {category_name}")
                    driver.quit()
                    return news_data

                news_data.append({"title": title, "date": formatted_datetime, "link": real_link, "description": full_content})

            except Exception:
                continue

        next_page = soup.find('a', string='Next')
        if next_page and 'href' in next_page.attrs:
            driver.get(next_page['href'])
        else:
            break

    driver.quit()
    print(f"📌 ดึงข่าวจาก {category_name} ได้ {len(news_data)} ข่าว")  # ✅ Debug จำนวนข่าวที่ดึงมา
    return news_data

def clean_and_process_data():
    if not os.path.exists(RAW_CSV_FILE):
        print("⚠️ ไม่มีไฟล์ CSV ให้ clean")
        return

    df = pd.read_csv(RAW_CSV_FILE, encoding='utf-8')
    print(f"📊 ตรวจสอบข่าวที่โหลดมา: {len(df)} ข่าว")  # ✅ Debug ก่อน clean

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 🔹 ลบเฉพาะข่าวที่เก่ากว่า 01/03/2025
    cutoff_date = datetime(2025, 3, 1, 0, 0, 0)
    df = df[df['date'] >= cutoff_date]

    # 🔹 ลบข่าวซ้ำ โดยพิจารณาจาก 'title' และ 'date'
    df = df.drop_duplicates(subset=['title', 'date'], keep='first')

    # 🔹 เรียงลำดับข่าวจากใหม่ไปเก่า
    df = df.sort_values(by='date', ascending=False)

    if len(df) > 0:
        df.to_csv(CLEAN_CSV_FILE, index=False, encoding='utf-8')
        print(f"✅ [CLEANED] ลบข่าวซ้ำและเรียงลำดับข่าวจากใหม่ไปเก่า!")
    else:
        print("⚠️ ไม่มีข่าวที่อยู่ในช่วงเวลาที่ต้องการให้บันทึก!")

# 🔹 เรียกใช้งานฟังก์ชันหลัก
if __name__ == "__main__":
    scrape_all_news()
