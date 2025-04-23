import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import concurrent.futures
import threading
import os
import gc
from datetime import datetime, timedelta
import sys
import mysql.connector
from dotenv import load_dotenv
from pathlib import Path
import time
# ✅ ป้องกัน UnicodeEncodeError (ข้ามอีโมจิที่ไม่รองรับ)
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")


# ✅ ใช้โฟลเดอร์ปัจจุบันแทน BASE_DIR
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'usa')

# ✅ ตรวจสอบและสร้างโฟลเดอร์ "Investing_Folder" ถ้ายังไม่มี
News_FOLDER = os.path.join(path, "News")
os.makedirs(News_FOLDER, exist_ok=True)

# ✅ กำหนด path ของไฟล์ CSV
output_filename = os.path.join(News_FOLDER, "USA_News.csv")

print(f"✅ บันทึกไฟล์ CSV ที่: {output_filename}")

# ✅ URL ของข่าว
base_url = "https://www.investing.com/news/stock-market-news"

# ✅ Lock สำหรับการใช้ Chrome instance
driver_lock = threading.Lock()


# ปิดการทำงานของ __del__ ที่ทำให้เกิด WinError 6
def patched_del(self):
    pass

uc.Chrome.__del__ = patched_del

uc.Chrome.__del__ = patched_del
dotenv_path = Path(__file__).resolve().parents[1] / "config.env"
load_dotenv(dotenv_path)

db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": os.getenv("DB_PORT")
}


conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()


from datetime import datetime

def get_latest_news_date_from_database():
    cursor.execute("SELECT MAX(PublishedDate) FROM News WHERE Source = 'investing'")
    latest_date = cursor.fetchone()[0]
    if latest_date:
        # ถ้า latest_date เป็น datetime แล้ว ให้ใช้ .date() เพื่อแปลงเป็น datetime.date
        if isinstance(latest_date, datetime):
            latest_date = latest_date.date()
        else:
            # ถ้าไม่ใช่ datetime (เช่น เป็น string) ค่อยแปลงด้วย strptime
            latest_date = datetime.strptime(latest_date, "%Y-%m-%d %H:%M:%S").date()

        print(f"🗓️ ข่าวล่าสุดคือวันที่: {latest_date}")
        return latest_date
    else:
        print("⚠️ ไม่มีข่าวในฐานข้อมูล เริ่มดึงข่าวตั้งแต่ 7 วันก่อน")
        return (datetime.now() - timedelta(days=7)).date()  # ดึงย้อนหลัง 7 วัน

# ในส่วนที่ต้องการเปรียบเทียบ news_date กับ latest_date
news_date = datetime.now().date()  # ตัวอย่าง
latest_date = get_latest_news_date_from_database()  # รับค่า latest_date จากฐานข้อมูล

if news_date and news_date <= latest_date:
    print("ข่าวนี้ไม่ใหม่เกินไป")
else:
    print("ข่าวนี้ใหม่มากพอที่จะดึงข้อมูล")




def init_driver():
    """สร้าง Chrome driver instance แบบปลอดภัย"""
    with driver_lock:
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        driver = uc.Chrome(options=options)
        driver._keep_alive = False
    return driver

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

def smooth_scroll_to_bottom(driver, step=300, delay=0.3):
    """เลื่อนหน้าจอลงอย่างช้าๆ จนสุดหน้า"""
    last_height = driver.execute_script("return document.body.scrollHeight")
    current_position = 0

    while current_position < last_height:
        driver.execute_script(f"window.scrollTo(0, {current_position});")
        time.sleep(delay)
        current_position += step
        last_height = driver.execute_script("return document.body.scrollHeight")
    
    # เลื่อนถึงสุดท้ายอีกครั้ง เผื่อโหลดช้า
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1.5)


def scrape_news(driver):
    """ดึงข่าวจากหน้าเว็บ"""
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
        date_str = date_tag.get("datetime", "No Date")

        news_list.append({'title': title, 'link': link, 'description': description, 'date': date_str, 'image': img})

    return news_list

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

def scrape_page(page):
    """Scrape ข่าวจากหน้าเว็บ"""
    global scraped_pages
    
    if page in scraped_pages:
        print(f"⚠️ หน้าที่ {page} ถูกดึงไปแล้ว ข้าม...")
        return []
    driver = None
    try:
        driver = init_driver()
        driver.get(f"{base_url}/{page}" if page > 1 else base_url)
        smooth_scroll_to_bottom(driver)
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, "article")))
        close_popup(driver)
        
        
        news = scrape_news(driver)

        print(f"✅ ดึงข่าวจากหน้า {page} ได้ {len(news)} ข่าว")
        scraped_pages.add(page)
        return news

    except Exception as e:
        print(f"❌ Error ในหน้า {page}: {e}")
        return []

    finally:
        if driver:
            safe_quit(driver)
            driver = None  # 🔒 ป้องกัน error จาก __del__ ของ undetected_chromedriver


def save_to_csv(data, filename, write_header=False):
    """บันทึกข้อมูลลง CSV โดยป้องกันข่าวซ้ำ"""
    if not data:
        print("⚠️ ไม่มีข่าวใหม่ ไม่บันทึกไฟล์ CSV")
        return
    
    df_new = pd.DataFrame(data)

    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)
        if 'link' in df_existing.columns:
            existing_links = set(df_existing['link'].astype(str))
            df_new = df_new[~df_new['link'].astype(str).isin(existing_links)]
        else:
            existing_links = set()

        print(f"🛑 พบข่าวซ้ำ {len(existing_links)} ข่าว, ข้ามการบันทึก")

    if not df_new.empty:
        mode = 'w' if write_header else 'a'
        header = True if write_header else False
        df_new.to_csv(filename, index=False, encoding='utf-8', mode=mode, header=header)
        print(f"💾 บันทึกข่าว {len(df_new)} ข่าวลง CSV (mode={mode})")
    else:
        print("✅ ไม่มีข่าวใหม่ให้บันทึก")


def main():
    """ฟังก์ชันหลักที่ดึงข่าวตั้งแต่วันที่ล่าสุด"""
    latest_date = get_latest_news_date_from_database()
    batch_size = 1
    max_pages = 7499
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
                            news_date = datetime.strptime(item['date'], "%Y-%m-%d %H:%M:%S").date()
                        except (ValueError, TypeError):
                            news_date = None

                        if news_date and news_date <= latest_date:
                            print(f"⏹️ พบข่าวที่มีอยู่แล้ว ({latest_date}), หยุดดึงข่าวทันที")
                            save_to_csv(all_news, output_filename, write_header=is_first_save)
                            save_to_csv(all_news, os.path.join("Combined_News.csv"), write_header=False)
                            stop_scraping = True
                            driver = init_driver()
                            safe_quit(driver)
                            driver = None 
                            break

                save_to_csv(all_news, output_filename, write_header=is_first_save)
                save_to_csv(all_news, os.path.join("Combined_News.csv"), write_header=False)
                total_articles += len(all_news)
                is_first_save = False
                all_news = []
                futures = []

if __name__ == "__main__":
    main()
