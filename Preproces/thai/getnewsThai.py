# -*- coding: utf-8 -*-
import os
import sys
import io
import time
import random
import threading
import urllib.parse
import platform
import shutil
import tarfile
import zipfile
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import requests
import mysql.connector
from dotenv import load_dotenv, find_dotenv

from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# -------------------- STDOUT UTF-8 --------------------
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# -------------------- ENV / DB ------------------------
load_dotenv(find_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.env')))

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

# -------------------- CONFIG -------------------------
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

driver_lock = threading.Lock()
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0"
]

# =====================================================
#      GECKODRIVER RESOLUTION (ไม่เรียก GitHub API)
# =====================================================
def _gecko_direct_url(version="v0.34.0"):
    sysname = platform.system().lower()
    arch = platform.machine().lower()
    if "linux" in sysname:
        fname = f"geckodriver-{version}-linux64.tar.gz"
    elif "windows" in sysname:
        fname = f"geckodriver-{version}-win64.zip"
    elif "darwin" in sysname and arch in ("arm64", "aarch64"):
        fname = f"geckodriver-{version}-macos-aarch64.tar.gz"
    else:
        fname = f"geckodriver-{version}-macos.tar.gz"
    return f"https://github.com/mozilla/geckodriver/releases/download/{version}/{fname}"

def _extract_driver(archive_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    else:
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(dest_dir)
    exe = "geckodriver.exe" if platform.system().lower().startswith("win") else "geckodriver"
    driver_path = os.path.join(dest_dir, exe)
    if not os.path.isfile(driver_path):
        for root, _, files in os.walk(dest_dir):
            if exe in files:
                driver_path = os.path.join(root, exe)
                break
    try:
        if not platform.system().lower().startswith("win"):
            os.chmod(driver_path, 0o755)
    except Exception:
        pass
    return driver_path

def _download_direct(version, target_dir):
    url = _gecko_direct_url(version)
    os.makedirs(target_dir, exist_ok=True)
    tmp_path = os.path.join(target_dir, f"geckodriver-{version}.tmp")
    print(f"⬇️ ดาวน์โหลด Geckodriver (direct): {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    driver_path = _extract_driver(tmp_path, target_dir)
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return driver_path

def resolve_geckodriver_path():
    """
    ลำดับความพยายาม:
      1) GECKODRIVER_PATH (env)
      2) geckodriver ใน PATH
      3) แคชส่วนตัว ~/.cache/geckodriver
      4) ดาวน์โหลดแบบ direct (ปักเวอร์ชัน) — ไม่เรียก API
    """
    # 1) Env
    env_path = os.getenv("GECKODRIVER_PATH")
    if env_path and os.path.isfile(env_path):
        print(f"🔧 ใช้ GECKODRIVER_PATH: {env_path}")
        return env_path

    # 2) PATH?
    found = shutil.which("geckodriver")
    if found:
        print(f"🔎 พบ geckodriver ใน PATH: {found}")
        return found

    # 3) Cache
    cache_dir = os.path.join(Path.home(), ".cache", "geckodriver")
    os.makedirs(cache_dir, exist_ok=True)
    exe = "geckodriver.exe" if platform.system().lower().startswith("win") else "geckodriver"
    cached = os.path.join(cache_dir, exe)
    if os.path.isfile(cached):
        print(f"💾 ใช้ geckodriver จาก cache: {cached}")
        return cached

    # 4) Direct download fallback
    version = os.getenv("GECKODRIVER_VERSION", "v0.34.0")
    try:
        driver_path = _download_direct(version, cache_dir)
        print(f"✅ ติดตั้ง (direct) สำเร็จ: {driver_path}")
        return driver_path
    except Exception as e:
        print(f"❌ ดาวน์โหลดแบบ direct ล้มเหลว: {e}")
        raise

# ทำครั้งเดียว
try:
    GECKO_PATH = resolve_geckodriver_path()
except Exception as e:
    print("❌ ตั้งค่า geckodriver ไม่สำเร็จ หยุดโปรแกรม:", e)
    sys.exit(1)

# =====================================================
#                 DRIVER & NAV HELPERS
# =====================================================
def init_driver():
    """Headless Firefox แบบไม่รอโหลดทั้งหน้า (กัน timeout)"""
    with driver_lock:
        options = FirefoxOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--headless")
        options.add_argument(f"--user-agent={random.choice(USER_AGENTS)}")
        # ไม่รอให้โหลดครบทั้งหน้า
        options.page_load_strategy = "none"

        try:
            service = Service(executable_path=GECKO_PATH)
            driver = webdriver.Firefox(service=service, options=options)
            driver.set_page_load_timeout(12)   # เราจะ stop เอง
            driver.set_script_timeout(20)
            time.sleep(0.8)
            try:
                print(f"Firefox version: {driver.capabilities.get('browserVersion')}")
            except Exception:
                pass
            return driver
        except Exception as e:
            print(f"❌ ไม่สามารถเริ่ม GeckoDriver: {e}")
            return None

def smart_get(driver, url, hard_timeout=12, wait_selector=(By.CLASS_NAME, "mk-listnew--title"), wait_timeout=20):
    """
    เข้า URL แบบไม่รอโหลดทั้งหน้า:
    - ให้เวลาเบื้องต้นกับ driver.get()
    - ครบเวลาแล้วสั่ง window.stop() เพื่อตัดสคริปต์/แอด
    - จากนั้นรอ element เป้าหมายให้โผล่
    """
    try:
        driver.set_page_load_timeout(hard_timeout)
        try:
            driver.get(url)
        except TimeoutException:
            try:
                driver.execute_script("window.stop();")
            except Exception:
                pass

        WebDriverWait(driver, wait_timeout).until(EC.presence_of_element_located(wait_selector))
        return True
    except Exception as e:
        print(f"⚠️ smart_get fail: {e}")
        return False

def safe_quit(driver):
    if driver:
        try:
            driver.quit()
            if getattr(driver, "service", None):
                driver.service.stop()
        except Exception:
            pass
        print("✅ ปิด WebDriver")

# =====================================================
#                    SCRAPER LOGIC
# =====================================================
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
        response = requests.get(real_link, timeout=8)
        response.raise_for_status()
        news_soup = BeautifulSoup(response.content, 'html.parser')

        date_tag = news_soup.find('div', class_='article-info--col')
        date = date_tag.find('p', string=lambda x: x and 'PUBLISHED :' in x).get_text(strip=True).replace('PUBLISHED :', '').strip() if date_tag else 'No Date'

        img_url = "No Image"
        box_img = news_soup.find('div', class_='box-img')
        if box_img and box_img.find('figure') and box_img.find('figure').find('img'):
            img_url = box_img.find('figure').find('img').get('src')

        if img_url == "No Image":
            article_content = news_soup.find('div', class_='article-content')
            if article_content:
                box_img = article_content.find('div', class_='box-img')
                if box_img and box_img.find('figure') and box_img.find('figure').find('img'):
                    img_url = box_img.find('figure').find('img').get('src')

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
    driver = init_driver()
    news_data = []
    if not driver:
        print(f"❌ เปิดเบราว์เซอร์ไม่ได้: {category_name}")
        return news_data
    try:
        # ใช้ smart_get กันหน้าโหลดช้า
        ok = smart_get(driver, url, hard_timeout=12, wait_selector=(By.CLASS_NAME, 'mk-listnew--title'), wait_timeout=20)
        if not ok:
            raise TimeoutException("smart_get timed out")

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        articles = soup.find_all('div', class_='mk-listnew--title')

        if not articles:
            print(f"⚠️ ไม่พบข่าวในหมวด {category_name}")
            return news_data

        for article in articles:
            try:
                title_tag = article.find('h3').find('a')
                title = title_tag.get_text(strip=True)
                link = title_tag['href']

                real_link = urllib.parse.parse_qs(urllib.parse.urlparse(link).query).get('href', [link])[0] if 'track/visitAndRedirect' in link else link
                date, full_content, img_url = fetch_news_content(real_link)
                formatted_datetime = parse_and_format_datetime(date)

                if formatted_datetime:
                    try:
                        if datetime.strptime(formatted_datetime, "%Y-%m-%d %H:%M:%S").date() <= latest_date:
                            print(f"[STOP] พบข่าวที่มีอยู่แล้ว ({latest_date}), หยุดดึง {category_name}")
                            return news_data
                    except Exception:
                        pass

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

    except (TimeoutException, WebDriverException) as e:
        # fallback: หยุดโหลดแล้วเก็บที่มี
        try:
            driver.execute_script("window.stop();")
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            articles = soup.find_all('div', class_='mk-listnew--title')
            for article in articles:
                try:
                    title_tag = article.find('h3').find('a')
                    title = title_tag.get_text(strip=True)
                    link = title_tag['href']
                    real_link = urllib.parse.parse_qs(urllib.parse.urlparse(link).query).get('href', [link])[0] if 'track/visitAndRedirect' in link else link
                    date, full_content, img_url = fetch_news_content(real_link)
                    formatted_datetime = parse_and_format_datetime(date)
                    news_data.append({
                        "title": title,
                        "date": formatted_datetime,
                        "link": real_link,
                        "description": full_content,
                        "image": img_url
                    })
                except Exception:
                    continue
            if news_data:
                print(f"✅ เก็บได้ {len(news_data)} ข่าวแม้จะ timeout")
        except Exception:
            print(f"❌ Error: {e}")

    finally:
        safe_quit(driver)

    return news_data

def ensure_csv_file_exists():
    """สร้างไฟล์ CSV ว่างพร้อม header ถ้ายังไม่มี"""
    if not os.path.exists(RAW_CSV_FILE):
        empty_df = pd.DataFrame(columns=["title", "date", "link", "description", "image"])
        empty_df.to_csv(RAW_CSV_FILE, index=False, encoding='utf-8')
        print(f"[CREATED] สร้างไฟล์ CSV ใหม่: {RAW_CSV_FILE}")

def scrape_all_news():
    print(" [START] เริ่มต้นดึงข่าว...")
    all_news_data = []

    # ลด parallel เพื่อกันล้ม/กันโดน block
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(scrape_news_from_category, name, url): name for name, url in NEWS_CATEGORIES.items()}
        for future in as_completed(futures):
            result = future.result()
            all_news_data.extend(result)
            time.sleep(0.5)

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
