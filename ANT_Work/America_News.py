import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import pandas as pd
import concurrent.futures
import threading
import os

base_url = "https://www.investing.com/news/stock-market-news"
output_filename = os.path.join(BASE_DIR, "Investing_Folder", "investing_news.csv")

# ✅ ตรวจสอบระดับของโฟลเดอร์ (ปรับ `..` ตามตำแหน่งของไฟล์)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 

# Lock สำหรับการใช้ Chrome instance
driver_lock = threading.Lock()

def init_driver():
    """สร้าง Chrome driver instance แบบปลอดภัย"""
    with driver_lock:
        options = uc.ChromeOptions()
        options.add_argument('--headless')  # รันแบบไม่แสดงหน้าต่างเบราว์เซอร์
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
        print("✅ Popup closed.")
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
        date = date_tag['datetime'] if date_tag and 'datetime' in date_tag.attrs else 'No Date'
        news_list.append({'title': title, 'link': link, 'description': description, 'date': date})
    return news_list

def safe_quit(driver):
    """ ปิด driver อย่างปลอดภัย """
    try:
        if driver:
            driver.quit()
    except Exception as e:
        print(f"⚠️ Warning: Driver quit failed: {e}")

def scrape_page(page):
    """Scrape ข่าวจากหน้าเว็บ"""
    driver = None
    try:
        driver = init_driver()
        if page == 1:
            driver.get(base_url)
        else:
            page_url = f"{base_url}/{page}"
            driver.get(page_url)

        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, "article")))
        close_popup(driver)
        news = scrape_news(driver)
        print(f"✅ Scraped page {page} ({len(news)} articles)")
    except Exception as e:
        print(f"❌ Error on page {page}: {e}")
        news = []
    finally:
        safe_quit(driver)
    return news

def save_to_csv(data, filename, write_header=False):
    """บันทึกข้อมูลลง CSV"""
    df = pd.DataFrame(data)
    mode = 'w' if write_header else 'a'
    header = True if write_header else False
    df.to_csv(filename, index=False, encoding='utf-8', mode=mode, header=header)
    print(f"✅ Data saved to {filename} (mode={mode}, header={header})")

def main():
    # เตรียม driver
    temp_driver = init_driver()
    temp_driver.quit()

    if os.path.exists(output_filename):
        os.remove(output_filename)

    batch_size = 5  # ลดจำนวน thread ลงเพื่อป้องกัน Chrome crash
    max_pages = 7499
    all_news = []
    is_first_save = True

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for page in range(1, max_pages + 1):
            futures.append(executor.submit(scrape_page, page))
            
            if len(futures) == batch_size or page == max_pages:
                for future in concurrent.futures.as_completed(futures):
                    all_news.extend(future.result())
                save_to_csv(all_news, output_filename, write_header=is_first_save)
                is_first_save = False
                all_news = []
                futures = []
    print("✅ Scraping complete.")

if __name__ == "__main__":
    main()
