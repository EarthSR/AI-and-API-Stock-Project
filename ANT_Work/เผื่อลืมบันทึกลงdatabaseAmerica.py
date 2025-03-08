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


# ✅ URL ของข่าว
base_url = 'https://www.investing.com/news/stock-market-news'
output_filename = "D:/Stock_Project/AI-and-API-Stock-Project/Investing_Folder/investing_news.csv"

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
        link = title_tag['href'] if title_tag and 'href' in title_tag.attrs else 'No Link'
        description_tag = article.find('p', {'data-test': 'article-description'})
        description = description_tag.get_text(strip=True) if description_tag else 'No Description'
        date_tag = article.find('time', {'data-test': 'article-publish-date'})
        date_str = date_tag['datetime'] if date_tag and 'datetime' in date_tag.attrs else 'No Date'

        news_list.append({'title': title, 'link': link, 'description': description, 'date': date_str})
    
    return news_list

def safe_quit(driver):
    """ ปิด driver อย่างปลอดภัย และแก้ `WinError 6`"""
    if driver:
        try:
            if hasattr(driver, "service") and driver.service.process:
                driver.quit()
                del driver  # ✅ ลบ object ของ WebDriver
                gc.collect()  # ✅ เคลียร์หน่วยความจำ
                print("✅ WebDriver ปิดเรียบร้อย")
            else:
                print("⚠️ WebDriver ปิดไปแล้ว หรือไม่สามารถปิดได้")
        except Exception as e:
            print(f"⚠️ Warning: WebDriver ปิดไม่สมบูรณ์: {e}")

def scrape_page(page):
    """Scrape ข่าวจากหน้าเว็บ"""
    driver = None
    try:
        driver = init_driver()
        driver.get(f"{base_url}/{page}" if page > 1 else base_url)

        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, "article")))
        close_popup(driver)
        news = scrape_news(driver)

        print(f"✅ ดึงข่าวจากหน้า {page} ได้ {len(news)} ข่าว")
        return news

    except Exception as e:
        print(f"❌ Error ในหน้า {page}: {e}")
        return []

    finally:
        safe_quit(driver)

def save_to_csv(data, filename, write_header=False):
    """บันทึกข้อมูลลง CSV"""
    if not data:
        print("⚠️ ไม่มีข่าวใหม่ ไม่บันทึกไฟล์ CSV")
        return
    
    df = pd.DataFrame(data)
    mode = 'w' if write_header else 'a'
    header = True if write_header else False
    df.to_csv(filename, index=False, encoding='utf-8', mode=mode, header=header)
    print(f"💾 บันทึกข่าว {len(data)} ข่าวลง CSV (mode={mode})")

def clean_csv(filename):
    """ลบข่าวเก่ากว่า 1 มีนาคม 2025 ออกจาก CSV และ Clean ข่าวซ้ำ พร้อมเรียงลำดับ"""
    if not os.path.exists(filename):
        print("⚠️ ไม่มีไฟล์ CSV ให้ clean")
        return

    df = pd.read_csv(filename)
    if df.empty:
        print("⚠️ ไฟล์ CSV ว่างเปล่า, ไม่มีข้อมูลให้ clean")
        return

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # ✅ ตั้งค่าให้ลบเฉพาะข่าวที่เก่ากว่า 1 มีนาคม 2025
    cutoff_date = datetime(2025, 3, 1).date()
    df_valid_news = df[df['date'].dt.date >= cutoff_date]  # ✅ ข่าวตั้งแต่ 1 มีนาคมขึ้นไป
    df_old_news = df[df['date'].dt.date < cutoff_date]  # ✅ ข่าวที่เก่ากว่า 1 มีนาคม

    # ✅ ลบข่าวซ้ำ โดยพิจารณาจาก 'title' และ 'date'
    df_valid_news = df_valid_news.drop_duplicates(subset=['title', 'date'], keep='first')

    # ✅ เรียงลำดับข่าวจากใหม่ไปเก่า
    df_valid_news = df_valid_news.sort_values(by='date', ascending=False)

    # ✅ จำนวนข่าวที่เหลือ
    total_after_clean = len(df_valid_news)

    # ✅ บันทึกไฟล์ CSV ใหม่
    df_valid_news.to_csv(filename, index=False)

    print(f"\n🔍 **Clean CSV Summary** 🔍")
    print(f"📊 ข่าวทั้งหมดก่อน Clean: {len(df)} ข่าว")
    print(f"🗑️ ข่าวที่ถูกลบ (เก่ากว่า {cutoff_date}): {len(df_old_news)} ข่าว")
    print(f"🧹 ข่าวซ้ำที่ถูกลบ: {len(df) - len(df_valid_news) - len(df_old_news)} ข่าว")
    print(f"✅ ข่าวที่เหลือ (ของ {cutoff_date} ขึ้นไป): {total_after_clean} ข่าว")
    print(f"✅ ข่าวถูกจัดเรียงลำดับจากใหม่ไปเก่า!")

def main():
    if os.path.exists(output_filename):
        os.remove(output_filename)  # ✅ ลบไฟล์เก่าก่อน

    batch_size = 5  # ✅ ลดจำนวน thread ลงเพื่อป้องกัน Chrome crash
    max_pages = 7499
    all_news = []
    is_first_save = True
    stop_scraping = False  # ✅ เพิ่ม flag ควบคุมการหยุดดึงข่าว
    total_articles = 0  # ✅ ตัวแปรเก็บจำนวนข่าวที่ดึงมาได้ทั้งหมด
    cutoff_date = datetime(2025, 3, 1).date()  # ✅ ตั้งค่าขั้นต่ำเป็น 1 มีนาคม 2025

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for page in range(1, max_pages + 1):
            if stop_scraping:
                break  # ✅ หยุดดึงข่าวเมื่อพบข่าวเก่ากว่า 1 มีนาคม 2025

            futures.append(executor.submit(scrape_page, page))
            
            if len(futures) == batch_size or page == max_pages:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    all_news.extend(result)

                    # ✅ เช็คว่าใน batch มีข่าวเก่ากว่า 1 มีนาคม 2025 หรือไม่
                    for item in result:
                        try:
                            news_date = datetime.strptime(item['date'], "%Y-%m-%d %H:%M:%S").date()
                            if news_date < cutoff_date:
                                print(f"⏹️ พบข่าวเก่ากว่า {cutoff_date}, หยุดดึงข่าวทันที")
                                
                                # ✅ ลบข่าวซ้ำก่อนบันทึกลง CSV
                                df_all_news = pd.DataFrame(all_news)
                                df_all_news = df_all_news.drop_duplicates(subset=['title', 'date'], keep='first')

                                save_to_csv(df_all_news.to_dict(orient='records'), output_filename, write_header=is_first_save)
                                
                                stop_scraping = True
                                break

                        except ValueError:
                            pass

                # ✅ ลบข่าวซ้ำก่อนบันทึก CSV
                df_all_news = pd.DataFrame(all_news)
                df_all_news = df_all_news.drop_duplicates(subset=['title', 'date'], keep='first')

                save_to_csv(df_all_news.to_dict(orient='records'), output_filename, write_header=is_first_save)
                
                total_articles += len(df_all_news)
                is_first_save = False
                all_news = []
                futures = []

    clean_csv(output_filename)

if __name__ == "__main__":
    main()
