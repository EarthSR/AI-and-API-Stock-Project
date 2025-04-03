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

# ✅ ป้องกัน UnicodeEncodeError (ข้ามอีโมจิที่ไม่รองรับ)
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")


# ✅ ใช้โฟลเดอร์ปัจจุบันแทน BASE_DIR
CURRENT_DIR = os.getcwd()

# ✅ ตรวจสอบและสร้างโฟลเดอร์ "Investing_Folder" ถ้ายังไม่มี
News_FOLDER = os.path.join(CURRENT_DIR, "News")
os.makedirs(News_FOLDER, exist_ok=True)

# ✅ กำหนด path ของไฟล์ CSV
output_filename = os.path.join(News_FOLDER, "USA_News.csv")

print(f"✅ บันทึกไฟล์ CSV ที่: {output_filename}")

# ✅ URL ของข่าว
base_url = "https://www.investing.com/news/stock-market-news"

# ✅ Lock สำหรับการใช้ Chrome instance
driver_lock = threading.Lock()

def get_latest_news_date_from_csv():

    last_news_file = os.path.join("Combined_News.csv")
    df = pd.read_csv(last_news_file)
    latest_date = df['date'].max()
    if latest_date:
        latest_date = datetime.strptime(latest_date, "%Y-%m-%d %H:%M:%S").date()
        print(f"🗓️ ข่าวล่าสุดคือวันที่: {latest_date}")
        return latest_date
    else:
        print("⚠️ ไม่มีข่าวในฐานข้อมูล เริ่มดึงข่าวตั้งแต่ 7 วันก่อน")
        return datetime.now().date() - timedelta(days=7)  # ดึงย้อนหลัง 7 วัน

def init_driver():
    """สร้าง Chrome driver instance แบบปลอดภัย"""
    with driver_lock:
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        driver = uc.Chrome(options=options)
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

def scrape_news(driver):
    """ดึงข่าวจากหน้าเว็บ"""
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    articles = soup.find_all('article', {'data-test': 'article-item'})
    news_list = []
    
    for article in articles:
        title_tag = article.find('a', {'data-test': 'article-title-link'})
        title = title_tag.get_text(strip=True) if title_tag else 'No Title'
        link = title_tag.get("href", "No Link")

        description_tag = article.find('p', {'data-test': 'article-description'})
        description = description_tag.get_text(strip=True) if description_tag else 'No Description'

        date_tag = article.find('time', {'data-test': 'article-publish-date'})
        date_str = date_tag.get("datetime", "No Date")

        news_list.append({'title': title, 'link': link, 'description': description, 'date': date_str})

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
    """ฟังก์ชันหลักที่ดึงข่าวตั้งแต่วันที่ล่าสุดใน Database"""
    latest_date = get_latest_news_date_from_csv()
    batch_size = 5
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
                            print(f"⏹️ พบข่าวที่มีอยู่แล้วในฐานข้อมูล ({latest_date}), หยุดดึงข่าวทันที")
                            save_to_csv(all_news, output_filename, write_header=is_first_save)
                            save_to_csv(all_news, os.path.join("Combined_News.csv"), write_header=False)
                            stop_scraping = True
                            break

                save_to_csv(all_news, output_filename, write_header=is_first_save)
                save_to_csv(all_news, os.path.join("Combined_News.csv"), write_header=False)
                total_articles += len(all_news)
                is_first_save = False
                all_news = []
                futures = []

if __name__ == "__main__":
    main()
