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

# ✅ ตรวจสอบระดับของโฟลเดอร์ (ปรับ `..` ตามตำแหน่งของไฟล์)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 

# ✅ URL ของข่าว
base_url = "https://www.investing.com/news/stock-market-news"
output_filename = os.path.join(BASE_DIR, "Investing_Folder", "USA_News.csv")

# ✅ Lock สำหรับการใช้ Chrome instance
driver_lock = threading.Lock()

# ✅ กำหนด "เมื่อวาน"
yesterday = (datetime.now() - timedelta(days=1)).date()

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
        link = title_tag.get("href", "No Link")  # ✅ ใช้ .get() ป้องกัน KeyError

        description_tag = article.find('p', {'data-test': 'article-description'})
        description = description_tag.get_text(strip=True) if description_tag else 'No Description'

        date_tag = article.find('time', {'data-test': 'article-publish-date'})
        date_str = date_tag.get("datetime", "No Date")  # ✅ ใช้ .get() ป้องกัน KeyError

        news_list.append({'title': title, 'link': link, 'description': description, 'date': date_str})

    return news_list

def safe_quit(driver):
    """ ปิด WebDriver อย่างปลอดภัย และแก้ `WinError 6` """
    if driver:
        try:
            driver.quit()
            if driver.service:
                driver.service.stop()  # ✅ ปิด WebDriver service
            del driver
            gc.collect()  # ✅ เคลียร์หน่วยความจำ
            print("✅ WebDriver ปิดเรียบร้อย")
        except Exception as e:
            print(f"⚠️ Warning: WebDriver ปิดไม่สมบูรณ์: {e}")

scraped_pages = set()  # ✅ เพิ่มตัวแปร global เพื่อกันดึงหน้าซ้ำ

def scrape_page(page):
    """Scrape ข่าวจากหน้าเว็บ"""
    global scraped_pages

    # ✅ ตรวจสอบก่อนว่าหน้านี้ถูกดึงมาแล้วหรือยัง
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
        scraped_pages.add(page)  # ✅ บันทึกหน้าที่ถูกดึงแล้ว
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

    # ✅ โหลดไฟล์ CSV เดิมเพื่อเช็คข่าวซ้ำ
    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)
        
        # ✅ ป้องกัน KeyError
        if 'link' in df_existing.columns:
            existing_links = set(df_existing['link'].astype(str))
            df_new = df_new[~df_new['link'].astype(str).isin(existing_links)]
        else:
            existing_links = set()

        print(f"🛑 พบข่าวซ้ำ {len(existing_links)} ข่าว, ข้ามการบันทึก")

    # ✅ ถ้ามีข่าวใหม่ให้บันทึก
    if not df_new.empty:
        mode = 'w' if write_header else 'a'
        header = True if write_header else False
        df_new.to_csv(filename, index=False, encoding='utf-8', mode=mode, header=header)
        print(f"💾 บันทึกข่าว {len(df_new)} ข่าวลง CSV (mode={mode})")
    else:
        print("✅ ไม่มีข่าวใหม่ให้บันทึก")


def clean_csv(filename):
    """ลบข่าวที่ไม่ใช่ของเมื่อวานออกจาก CSV"""
    if not os.path.exists(filename):
        print("⚠️ ไม่มีไฟล์ CSV ให้ clean")
        return

    if not os.path.exists(filename):
        print(f"⚠️ ไม่มีไฟล์ {filename} ให้ clean")
        return

    df = pd.read_csv(filename)

    if df.empty:
        print("⚠️ ไฟล์ CSV ว่างเปล่า, ไม่มีข้อมูลให้ clean")
        return

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # ✅ คำนวณจำนวนข่าวทั้งหมดก่อน clean
    total_before_clean = len(df)

    # ✅ แบ่งข่าวเป็น 3 กลุ่ม
    df_yesterday = df[df['date'].dt.date == yesterday]  # ข่าวของเมื่อวาน
    df_old_news = df[df['date'].dt.date < yesterday]    # ข่าวที่เก่ากว่าเมื่อวาน
    df_today_news = df[df['date'].dt.date > yesterday]  # ข่าวของวันนี้

    # ✅ จำนวนข่าวที่ถูกลบ
    deleted_news = len(df_old_news) + len(df_today_news)

    # ✅ จำนวนข่าวที่เหลือ
    total_after_clean = len(df_yesterday)

    df_yesterday.to_csv(filename, index=False)

    print(f"\n🔍 **Clean CSV Summary** 🔍")
    print(f"📊 ข่าวทั้งหมดก่อน Clean: {total_before_clean} ข่าว")
    print(f"🗑️ ข่าวที่ถูกลบ (เก่า + วันนี้): {deleted_news} ข่าว")
    print(f"✅ ข่าวที่เหลือ (ของ {yesterday}): {total_after_clean} ข่าว")

def main():
    if os.path.exists(output_filename):
        os.remove(output_filename)  # ✅ ลบไฟล์เก่าก่อน

    batch_size = 5  # ✅ ลดจำนวน thread ลงเพื่อป้องกัน Chrome crash
    max_pages = 7499
    all_news = []
    is_first_save = True
    stop_scraping = False  # ✅ เพิ่ม flag ควบคุมการหยุดดึงข่าว
    total_articles = 0  # ✅ ตัวแปรเก็บจำนวนข่าวที่ดึงมาได้ทั้งหมด

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for page in range(1, max_pages + 1):
            if stop_scraping:
                break  # ✅ หยุดดึงข่าวเมื่อพบข่าวเก่ากว่าเมื่อวาน

            futures.append(executor.submit(scrape_page, page))
            
            if len(futures) == batch_size or page == max_pages:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    all_news.extend(result)

                    for item in result:
                        try:
                            news_date = datetime.strptime(item['date'], "%Y-%m-%d %H:%M:%S").date()
                        except (ValueError, TypeError):
                            news_date = None  # ✅ ป้องกัน error กรณีไม่มีวันที่

                        if news_date and news_date < yesterday:
                            print(f"⏹️ พบข่าวเก่ากว่า {yesterday}, หยุดดึงข่าวทันที")
                            save_to_csv(all_news, output_filename, write_header=is_first_save)
                            stop_scraping = True
                            break

                save_to_csv(all_news, output_filename, write_header=is_first_save)
                total_articles += len(all_news)
                is_first_save = False
                all_news = []
                futures = []

    clean_csv(output_filename)

if __name__ == "__main__":
    main()
