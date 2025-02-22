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

base_url = 'https://www.investing.com/news/stock-market-news'
output_filename = "investing_news.csv"

# สร้าง lock สำหรับการ initialize driver ให้ทำงานแบบ serial
driver_lock = threading.Lock()

def init_driver():
    with driver_lock:
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--blink-settings=imagesEnabled=false")  # ปิดการโหลดรูป
        options.add_argument("--disable-gpu")  # ปิด GPU acceleration
        options.add_argument("--disable-extensions")
        driver = uc.Chrome(options=options)
    return driver

def close_popup(driver):
    """ฟังก์ชันปิด popup ถ้ามี"""
    try:
        close_button = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.XPATH, "//svg[@data-test='sign-up-dialog-close-button']"))
        )
        close_button.click()
        print("Popup closed.")
    except Exception:
        pass

def scrape_news(driver):
    """ดึงข่าวจากหน้าเว็บโดยใช้ BeautifulSoup"""
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

def scrape_page(page):
    """
    เปิดหน้าเว็บที่กำหนด, ปิด popup, ดึงข้อมูลข่าว แล้วปิด driver
    โดยแต่ละ thread จะสร้าง instance ของ Chrome เอง
    """
    driver = init_driver()
    try:
        if page == 1:
            driver.get(base_url)
        else:
            page_url = f"{base_url}/{page}"
            driver.get(page_url)
        time.sleep(5)  # รอให้หน้าเว็บโหลด
        close_popup(driver)
        news = scrape_news(driver)
        print(f"Scraped page {page} with {len(news)} articles")
    except Exception as e:
        print(f"Error on page {page}: {e}")
        news = []
    finally:
        driver.quit()
    return news

def save_to_csv(data, filename, write_header=False):
    """บันทึกข้อมูลลง CSV ในโหมด append"""
    df = pd.DataFrame(data)
    mode = 'w' if write_header else 'a'
    header = True if write_header else False
    df.to_csv(filename, index=False, encoding='utf-8', mode=mode, header=header)
    print(f"✅ Data saved to {filename} (mode={mode}, header={header})")

def main():
    # Pre-initialize undetected_chromedriver เพื่อให้ unzip package แล้ว
    temp_driver = init_driver()
    temp_driver.quit()

    # หากมีไฟล์อยู่แล้ว ให้ลบออกเพื่อเริ่มใหม่
    if os.path.exists(output_filename):
        os.remove(output_filename)

    batch_size = 10   # จำนวนหน้าเปิดพร้อมกัน
    max_pages = 7499   # ปรับจำนวนหน้าที่ต้องการดึงข้อมูลตามต้องการ
    all_news = []    # ใช้เก็บข้อมูลใน batch ปัจจุบัน
    is_first_save = True  # ใช้ระบุว่าการบันทึกครั้งแรกควรมี header

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for page in range(1, max_pages + 1):
            futures.append(executor.submit(scrape_page, page))
            
            # เมื่อครบ batch หรืออยู่หน้าสุดท้ายแล้ว
            if len(futures) == batch_size or page == max_pages:
                for future in concurrent.futures.as_completed(futures):
                    all_news.extend(future.result())
                # บันทึกข้อมูลในแต่ละ batch แบบ append
                save_to_csv(all_news, output_filename, write_header=is_first_save)
                is_first_save = False  # หลังจากบันทึกครั้งแรก header จะไม่ถูกเขียนใหม่
                all_news = []  # ล้างข้อมูล batch หลังบันทึก
                futures = []
    print("✅ Scraping complete.")

if __name__ == "__main__":
    main()
