# -*- coding: utf-8 -*-
import sys, os, time, random, threading, platform, shutil, tarfile, zipfile
from datetime import datetime, timedelta
from pathlib import Path

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')

import pandas as pd
import requests
import mysql.connector
from dotenv import load_dotenv

from bs4 import BeautifulSoup
import concurrent.futures

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# ===================== CONFIG PATHS =====================
ROOT = Path(__file__).resolve().parent
USA_DIR = (ROOT / '..' / 'usa').resolve()
NEWS_DIR = USA_DIR / 'News'
NEWS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = NEWS_DIR / 'USA_News.csv'

# ===================== ENV / DB =========================
dotenv_path = (ROOT / '..' / 'config.env').resolve()
load_dotenv(dotenv_path)

db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": os.getenv("DB_PORT"),
    "autocommit": True
}
try:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    print("✅ DB connected")
except mysql.connector.Error as e:
    print(f"❌ DB connect failed: {e}")
    sys.exit(1)

# ===================== SCRAPER TARGET ===================
base_url = "https://www.investing.com/news/stock-market-news"

# ===================== UA & LOCK ========================
driver_lock = threading.Lock()
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0"
]

# =======================================================
#           GECKODRIVER RESOLUTION (No API limit)
# =======================================================
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
    print(f"⬇️ Downloading geckodriver direct: {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
    driver_path = _extract_driver(tmp_path, target_dir)
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return driver_path

def resolve_geckodriver_path():
    # 1) Env path
    env_path = os.getenv("GECKODRIVER_PATH")
    if env_path and os.path.isfile(env_path):
        print(f"🔧 Using GECKODRIVER_PATH: {env_path}")
        return env_path
    # 2) PATH
    found = shutil.which("geckodriver")
    if found:
        print(f"🔎 Found geckodriver in PATH: {found}")
        return found
    # 3) Cache
    cache_dir = os.path.join(Path.home(), ".cache", "geckodriver")
    os.makedirs(cache_dir, exist_ok=True)
    exe = "geckodriver.exe" if platform.system().lower().startswith("win") else "geckodriver"
    cached = os.path.join(cache_dir, exe)
    if os.path.isfile(cached):
        print(f"💾 Using cached geckodriver: {cached}")
        return cached
    # 4) Direct download fallback
    version = os.getenv("GECKODRIVER_VERSION", "v0.34.0")
    path = _download_direct(version, cache_dir)
    print(f"✅ Installed geckodriver direct: {path}")
    return path

try:
    GECKO_PATH = resolve_geckodriver_path()
except Exception as e:
    print("❌ geckodriver setup failed:", e)
    sys.exit(1)

# =======================================================
#                HELPERS: DRIVER & NAVIGATION
# =======================================================
def init_driver():
    """Headless Firefox, ไม่ปิดรูป (เพื่อเอารูปข่าวได้) + ไม่รอโหลดทั้งหน้า"""
    with driver_lock:
        options = FirefoxOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--headless")
        options.add_argument(f"--user-agent={random.choice(USER_AGENTS)}")

        # ไม่รอ page load จนจบ เพื่อกัน timeout
        options.page_load_strategy = "none"

        service = Service(executable_path=GECKO_PATH)
        try:
            driver = webdriver.Firefox(service=service, options=options)
            driver.set_page_load_timeout(12)   # เราจะ stop เอง
            driver.set_script_timeout(20)
            time.sleep(0.8)
            print(f"Firefox version: {driver.capabilities.get('browserVersion')}")
            return driver
        except Exception as e:
            print(f"❌ ไม่สามารถเริ่ม GeckoDriver: {e}")
            return None

def smart_get(driver, url, hard_timeout=12, wait_selector=(By.TAG_NAME, "article"), wait_timeout=20):
    """
    เข้า URL แบบไม่รอโหลดทั้งหน้า:
    1) ให้เวลาสั้น ๆ กับ driver.get()
    2) ครบแล้วสั่ง window.stop() เพื่อตัดสคริปต์/แอด
    3) รอ element เป้าหมายโผล่
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

def smooth_scroll_to_bottom(driver, step=200, delay=0.03, max_time=10):
    try:
        last_height = driver.execute_script("return document.body.scrollHeight")
        pos, start = 0, time.time()
        while pos < last_height and (time.time() - start) < max_time:
            driver.execute_script(f"window.scrollTo(0, {pos});")
            time.sleep(delay)
            pos += step
            new_h = driver.execute_script("return document.body.scrollHeight")
            if new_h > last_height:
                last_height = new_h
        time.sleep(0.2)
    except Exception as e:
        print(f"⚠️ scroll issue: {e}")

def close_popup(driver):
    try:
        btn = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.XPATH, "//svg[@data-test='sign-up-dialog-close-button']"))
        )
        btn.click()
        print("✅ Popup closed")
    except Exception:
        pass

# =======================================================
#                   SCRAPERS (USA)
# =======================================================
def get_latest_news_date_from_database():
    try:
        cursor.execute("SELECT MAX(PublishedDate) FROM News WHERE Source = 'investing'")
        latest = cursor.fetchone()[0]
        if latest:
            if isinstance(latest, datetime):
                return latest.date()
            return datetime.strptime(latest, "%Y-%m-%d %H:%M:%S").date()
        else:
            print("⚠️ ไม่มีข่าวในฐานข้อมูล เริ่ม 7 วันก่อน")
            return (datetime.now() - timedelta(days=7)).date()
    except mysql.connector.Error as e:
        print(f"❌ DB query error: {e}")
        return (datetime.now() - timedelta(days=7)).date()

def scrape_news(driver):
    """ดึงข้อมูลข่าวจากหน้า list (มีรูป/คำอธิบาย/เวลา)"""
    try:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        articles = soup.find_all('article', {'data-test': 'article-item'})
        out = []
        for a in articles:
            title_tag = a.find('a', {'data-test': 'article-title-link'})
            title = title_tag.get_text(strip=True) if title_tag else 'No Title'
            link = title_tag.get("href", "No Link") if title_tag else "No Link"

            img_tag = a.find('img', {'data-test': 'item-image'})
            img = img_tag.get('src') if img_tag and img_tag.get('src') else 'No Image'

            desc_tag = a.find('p', {'data-test': 'article-description'})
            desc = desc_tag.get_text(strip=True) if desc_tag else 'No Description'

            date_tag = a.find('time', {'data-test': 'article-publish-date'})
            date_str = date_tag.get("datetime", None) if date_tag else None
            if date_str:
                try:
                    datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    date_str = None

            out.append({
                "title": title, "link": link, "description": desc, "date": date_str, "image": img
            })
        return out
    except Exception as e:
        print(f"❌ parse error: {e}")
        return []

def safe_quit(driver):
    if driver:
        try:
            driver.quit()
            if getattr(driver, "service", None):
                driver.service.stop()
            del driver
        except Exception:
            pass
        print("✅ WebDriver ปิดเรียบร้อย")

scraped_pages = set()

def scrape_page(page, max_retries=3, latest_date=None):
    if page in scraped_pages:
        print(f"⚠️ หน้าที่ {page} ถูกดึงไปแล้ว ข้าม...")
        return []

    url = f"{base_url}/{page}" if page > 1 else base_url
    driver = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"เริ่มดึงหน้า {page} (ครั้งที่ {attempt})")
            driver = init_driver()
            if not driver:
                raise WebDriverException("start WebDriver failed")

            ok = smart_get(driver, url, hard_timeout=12, wait_selector=(By.TAG_NAME, "article"), wait_timeout=20)
            if not ok:
                raise TimeoutException("smart_get timed out")

            smooth_scroll_to_bottom(driver)
            close_popup(driver)

            news = scrape_news(driver)
            if news:
                print(f"✅ ดึงข่าวจากหน้า {page} ได้ {len(news)} ข่าว")
                scraped_pages.add(page)
                time.sleep(2.0)
                return news
            else:
                raise WebDriverException("no articles found")

        except (TimeoutException, WebDriverException) as e:
            # Fallback: หยุดโหลดแล้วเก็บที่มี
            try:
                if driver:
                    driver.execute_script("window.stop();")
                    news = scrape_news(driver)
                    if news:
                        print(f"✅ เก็บได้ {len(news)} ข่าวแม้จะ timeout")
                        scraped_pages.add(page)
                        time.sleep(1.5)
                        return news
            except Exception:
                pass

            print(f"❌ Error หน้า {page} (ครั้งที่ {attempt}): {e}")
            if attempt == max_retries:
                print(f"⏳ ล้มเหลวหลังจากลอง {max_retries} ครั้งในหน้า {page}")
                return []
            time.sleep(10 * attempt)

        except Exception as e:
            print(f"❌ Unexpected error หน้า {page} (ครั้งที่ {attempt}): {e}")
            return []

        finally:
            if driver:
                safe_quit(driver)

    return []

def save_to_csv(data, filename, write_header=False):
    if not data:
        print("⚠️ ไม่มีข่าวใหม่ ไม่บันทึกไฟล์ CSV")
        return
    df_new = pd.DataFrame(data)
    existing_links = set()
    os.makedirs(os.path.dirname(str(filename)), exist_ok=True)

    if os.path.exists(filename):
        try:
            df_existing = pd.read_csv(filename, encoding='utf-8')
            if 'link' in df_existing.columns:
                existing_links = set(df_existing['link'].astype(str))
                before = len(df_new)
                df_new = df_new[~df_new['link'].astype(str).isin(existing_links)]
                print(f"🛑 พบข่าวซ้ำ {len(existing_links)} ลิงก์ | คัดออก {before - len(df_new)} แถว")
        except Exception as e:
            print(f"❌ อ่าน CSV เดิมไม่ได้: {e}")
            return
    else:
        print(f"📄 ไฟล์ใหม่: {filename}")

    if not df_new.empty:
        mode = 'w' if write_header else 'a'
        header = True if write_header else False
        try:
            df_new.to_csv(filename, index=False, encoding='utf-8', mode=mode, header=header)
            print(f"💾 บันทึกข่าว {len(df_new)} ข่าวลง CSV '{filename}' (mode={mode})")
        except Exception as e:
            print(f"❌ บันทึก CSV ล้มเหลว: {e}")
    else:
        print("✅ ไม่มีข่าวใหม่ให้บันทึกหลังคัดซ้ำ")

def main():
    latest_date = get_latest_news_date_from_database()
    batch_size = 2          # ลด parallel เพื่อกันล้ม / กันโดน block
    max_pages = 200
    all_news = []
    is_first_save = True
    stop_scraping = False
    total_articles = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for page in range(1, max_pages + 1):
            if stop_scraping:
                break
            futures.append(executor.submit(scrape_page, page, 3, latest_date))

            if len(futures) == batch_size or page == max_pages:
                time.sleep(1.0)  # เว้นจังหวะลดโหลด
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    all_news.extend(result)

                    # ตรวจวันที่เจอข่าวเก่า → หยุด
                    for item in result:
                        try:
                            if item['date']:
                                nd = datetime.strptime(item['date'], "%Y-%m-%d %H:%M:%S").date()
                            else:
                                nd = None
                        except (ValueError, TypeError):
                            nd = None
                        if nd and nd <= latest_date:
                            print(f"⏹️ พบข่าวที่มีอยู่แล้ว ({latest_date}), หยุดดึงข่าวทันที")
                            save_to_csv(all_news, str(OUTPUT_CSV), write_header=is_first_save)
                            stop_scraping = True
                            break

                save_to_csv(all_news, str(OUTPUT_CSV), write_header=is_first_save)
                total_articles += len(all_news)
                print(f"📊 สรุปรอบนี้: {len(all_news)} ข่าว | สะสม: {total_articles}")
                is_first_save = False
                all_news = []
                futures = []

    try:
        cursor.close()
        conn.close()
        print("✅ ปิดการเชื่อมต่อฐานข้อมูลเรียบร้อย")
    except mysql.connector.Error as e:
        print(f"❌ ปิดฐานข้อมูลล้มเหลว: {e}")

if __name__ == "__main__":
    print("Selenium version:", webdriver.__version__ if hasattr(webdriver, "__version__") else "N/A")
    print(f"บันทึกไฟล์ CSV ที่: {OUTPUT_CSV}")
    main()
