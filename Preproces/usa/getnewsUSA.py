import sys
import selenium
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options as FirefoxOptions  # ใช้ FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup
import pandas as pd
import concurrent.futures
import threading
import os
import gc
from datetime import datetime, timedelta
import mysql.connector
from dotenv import load_dotenv
from pathlib import Path
import time
import random
from webdriver_manager.firefox import GeckoDriverManager

# ✅ ป้องกัน UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# ✅ ตรวจสอบเวอร์ชัน Selenium
print(f"Selenium version: {selenium.__version__}")

# ✅ ใช้โฟลเดอร์ปัจจุบัน
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'usa')
News_FOLDER = os.path.join(path, "News")
os.makedirs(News_FOLDER, exist_ok=True)

# ✅ กำหนด path ของไฟล์ CSV
output_filename = os.path.join(News_FOLDER, "USA_News.csv")
print(f"บันทึกไฟล์ CSV ที่: {output_filename}")

# ✅ URL ของข่าว
base_url = "https://www.investing.com/news/stock-market-news"

# ✅ Lock สำหรับการใช้ Firefox instance
driver_lock = threading.Lock()

# รายการ User-Agent สำหรับสุ่ม
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:127.0) Gecko/20100101 Firefox/127.0"
]

# โหลด environment variables
dotenv_path = Path(__file__).resolve().parents[1] / "config.env"
load_dotenv(dotenv_path)

db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": os.getenv("DB_PORT")
}

try:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
except mysql.connector.Error as e:
    print(f"❌ ไม่สามารถเชื่อมต่อฐานข้อมูล: {e}")
    sys.exit(1)

def get_latest_news_date_from_database():
    """ดึงวันที่ล่าสุดของข่าวจากฐานข้อมูล"""
    try:
        cursor.execute("SELECT MAX(PublishedDate) FROM News WHERE Source = 'investing'")
        latest_date = cursor.fetchone()[0]
        if latest_date:
            if isinstance(latest_date, datetime):
                latest_date = latest_date.date()
            else:
                latest_date = datetime.strptime(latest_date, "%Y-%m-%d %H:%M:%S").date()
            print(f"🗓️ ข่าวล่าสุดคือวันที่: {latest_date}")
            return latest_date
        else:
            print("⚠️ ไม่มีข่าวในฐานข้อมูล เริ่มดึงข่าวตั้งแต่ 7 วันก่อน")
            return (datetime.now() - timedelta(days=7)).date()
    except mysql.connector.Error as e:
        print(f"❌ ข้อผิดพลาดในการดึงวันที่จากฐานข้อมูล: {e}")
        return (datetime.now() - timedelta(days=7)).date()

def init_driver():
    """สร้าง Firefox driver instance แบบปลอดภัย"""
    with driver_lock:
        options = FirefoxOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--headless")  # ใช้ headless mode ถ้าต้องการ
        options.add_argument(f"--user-agent={random.choice(USER_AGENTS)}")
        # ถ้ามี proxy ใช้งาน ให้ uncomment
        # options.add_argument("--proxy-server=http://your_proxy:port")
        try:
            service = Service(GeckoDriverManager().install())
            driver = webdriver.Firefox(service=service, options=options)
            driver.set_page_load_timeout(30)
            time.sleep(2)
            print(f"Firefox version: {driver.capabilities['browserVersion']}")
            return driver
        except Exception as e:
            print(f"❌ ไม่สามารถเริ่ม GeckoDriver: {e}")
            return None

def close_popup(driver):
    """ปิด popup ถ้ามี"""
    try:
        close_button = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.XPATH, "//svg[@data-test='sign-up-dialog-close-button']"))
        )
        close_button.click()
        print("✅ Popup ปิดเรียบร้อย")
    except Exception:
        pass

def smooth_scroll_to_bottom(driver, step=200, delay=0.03):
    """เลื่อนหน้าจอลงอย่างช้าๆ จนสุดหน้า"""
    try:
        last_height = driver.execute_script("return document.body.scrollHeight")
        current_position = 0
        max_scroll_time = 10
        start_time = time.time()
        while current_position < last_height and (time.time() - start_time) < max_scroll_time:
            driver.execute_script(f"window.scrollTo(0, {current_position});")
            time.sleep(delay)
            current_position += step
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height > last_height:
                last_height = new_height
        time.sleep(0.2)
    except Exception as e:
        print(f"⚠️ ปัญหาการเลื่อนหน้า: {e}")

def scrape_news(driver):
    """ดึงข่าวจากหน้าเว็บ"""
    try:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        articles = soup.find_all('article', {'data-test': 'article-item'})
        news_list = []
        for article in articles:
            title_tag = article.find('a', {'data-test': 'article-title-link'})
            title = title_tag.get_text(strip=True) if title_tag else 'No Title'
            link = title_tag.get("href", "No Link")
            img_tag = article.find('img', {'data-test': 'item-image'})
            img = img_tag.get('src', 'No Image') if img_tag else 'No Image'
            description_tag = article.find('p', {'data-test': 'article-description'})
            description = description_tag.get_text(strip=True) if description_tag else 'No Description'
            date_tag = article.find('time', {'data-test': 'article-publish-date'})
            date_str = date_tag.get("datetime", None) if date_tag else None
            if date_str:
                try:
                    datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    date_str = None
            news_list.append({'title': title, 'link': link, 'description': description, 'date': date_str, 'image': img})
        return news_list
    except Exception as e:
        print(f"❌ ปัญหาการสแครปข่าว: {e}")
        return []

def safe_quit(driver):
    """ปิด WebDriver อย่างปลอดภัย"""
    if driver:
        try:
            driver.quit()
            if driver.service:
                driver.service.stop()
            del driver
            gc.collect()
            print("✅ WebDriver ปิดเรียบร้อย")
        except Exception as e:
            print(f"⚠️ Warning: WebDriver ปิดไม่สมบูรณ์: {e}")

scraped_pages = set()

def scrape_page(page, max_retries=3):
    """Scrape ข่าวจากหน้าเว็บด้วยการ retry"""
    global scraped_pages
    if page in scraped_pages:
        print(f"⚠️ หน้าที่ {page} ถูกดึงไปแล้ว ข้าม...")
        return []
    driver = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"เริ่มดึงหน้า {page} (ครั้งที่ {attempt})")
            driver = init_driver()
            if not driver:
                raise WebDriverException("ไม่สามารถเริ่ม WebDriver ได้")
            driver.get(f"{base_url}/{page}" if page > 1 else base_url)
            try:
                status_code = driver.execute_script("return window.performance.getEntriesByType('navigation')[0].responseStatus")
                if status_code != 200:
                    print(f"❌ หน้า {page} คืนสถานะ HTTP {status_code}")
                    return []
            except Exception:
                print(f"⚠️ ไม่สามารถตรวจสอบ HTTP status สำหรับหน้า {page}")
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "article")))
            smooth_scroll_to_bottom(driver)
            close_popup(driver)
            news = scrape_news(driver)
            print(f"✅ ดึงข่าวจากหน้า {page} ได้ {len(news)} ข่าว")
            scraped_pages.add(page)
            time.sleep(5)
            return news
        except (TimeoutException, WebDriverException) as e:
            print(f"❌ Error ในหน้า {page} (ครั้งที่ {attempt}): {e}")
            if attempt == max_retries:
                print(f"⏳ ล้มเหลวหลังจากลอง {max_retries} ครั้งในหน้า {page}")
                return []
            time.sleep(15 * attempt)
        except Exception as e:
            print(f"❌ Error ไม่คาดคิดในหน้า {page} (ครั้งที่ {attempt}): {e}")
            return []
        finally:
            if driver:
                safe_quit(driver)
    return []

def save_to_csv(data, filename, write_header=False):
    """บันทึกข้อมูลลง CSV โดยป้องกันข่าวซ้ำ"""
    if not data:
        print("⚠️ ไม่มีข่าวใหม่ ไม่บันทึกไฟล์ CSV")
        return
    try:
        df_new = pd.DataFrame(data)
        existing_links = set()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if os.path.exists(filename):
            try:
                df_existing = pd.read_csv(filename, encoding='utf-8')
                if 'link' in df_existing.columns:
                    existing_links = set(df_existing['link'].astype(str))
                    df_new = df_new[~df_new['link'].astype(str).isin(existing_links)]
                print(f"🛑 พบข่าวซ้ำ {len(existing_links)} ข่าว")
            except Exception as e:
                print(f"❌ ไม่สามารถอ่านไฟล์ CSV เดิมได้: {e}")
                return
        else:
            print(f"📄 ไฟล์ CSV '{filename}' ยังไม่มี จะสร้างใหม่")
        if not df_new.empty:
            mode = 'w' if write_header else 'a'
            header = True if write_header else False
            try:
                df_new.to_csv(filename, index=False, encoding='utf-8', mode=mode, header=header)
                print(f"💾 บันทึกข่าว {len(df_new)} ข่าวลง CSV '{filename}' (mode={mode})")
            except Exception as e:
                print(f"❌ ไม่สามารถบันทึกไฟล์ CSV ได้: {e}")
        else:
            print("✅ ไม่มีข่าวใหม่ให้บันทึก")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดใน save_to_csv: {e}")

def main():
    """ฟังก์ชันหลักที่ดึงข่าวตั้งแต่วันที่ล่าสุด"""
    latest_date = get_latest_news_date_from_database()
    batch_size = 4
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
            futures.append(executor.submit(scrape_page, page))
            if len(futures) == batch_size or page == max_pages:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    all_news.extend(result)
                    for item in result:
                        try:
                            if item['date']:
                                news_date = datetime.strptime(item['date'], "%Y-%m-%d %H:%M:%S").date()
                            else:
                                news_date = None
                        except (ValueError, TypeError):
                            news_date = None
                        if news_date and news_date <= latest_date:
                            print(f"⏹️ พบข่าวที่มีอยู่แล้ว ({latest_date}), หยุดดึงข่าวทันที")
                            save_to_csv(all_news, output_filename, write_header=is_first_save)              
                            stop_scraping = True
                            break
                save_to_csv(all_news, output_filename, write_header=is_first_save)
                total_articles += len(all_news)
                print(f"📊 ดึงข่าวทั้งหมด {total_articles} ข่าว")
                is_first_save = False
                all_news = []
                futures = []
    try:
        cursor.close()
        conn.close()
        print("✅ ปิดการเชื่อมต่อฐานข้อมูลเรียบร้อย")
    except mysql.connector.Error as e:
        print(f"❌ ข้อผิดพลาดในการปิดฐานข้อมูล: {e}")

if __name__ == "__main__":
    main()