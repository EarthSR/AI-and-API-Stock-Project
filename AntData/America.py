import os
import time
import urllib.parse
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime

def scrape_abc_news():
    """
    ดึงข้อมูลข่าวจาก ABC News โดยใช้ Selenium + BeautifulSoup (ความเร็วสูง)
    """
    # 🚀 ตั้งค่า Selenium เพื่อให้โหลดเร็วที่สุด
    options = Options()
    options.add_argument('--headless')  # ไม่เปิดหน้าต่างเบราว์เซอร์
    options.add_argument('--disable-gpu')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--blink-settings=imagesEnabled=false')  # ปิดการโหลดรูปภาพ
    options.page_load_strategy = 'eager'  # โหลดหน้าไวขึ้น

    # 🚀 เริ่มต้น WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # 🚀 ตั้งค่าพื้นฐาน
    base_url = "https://abcnews.go.com"
    page = 1
    search_url = f'https://abcnews.go.com/search?searchtext=stock technology&sort=date&page={page}'
    driver.get(search_url)

    news_data = []
    file_name = 'D:/StockData/AI-and-API-Stock-Project/news_data/ABCNews.csv'

    # โหลดข่าวที่เคยบันทึก
    existing_titles = set()
    if os.path.exists(file_name):
        try:
            df_existing = pd.read_csv(file_name, usecols=['Title'])
            existing_titles = set(df_existing['Title'].astype(str))
        except Exception:
            pass  # ถ้ามีปัญหาไม่ต้องแสดง log

    try:
        while True:
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'ContentRoll__Headline'))
            )
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            articles = soup.select("section.ContentRoll__Item")

            if not articles:
                break  # ไม่มีข่าวแล้ว หยุด

            for article in articles:
                try:
                    title_tag = article.select_one("div.ContentRoll__Headline a")
                    if not title_tag:
                        continue  # ข้ามถ้าไม่มีลิงก์ข่าว
                    title = title_tag.get_text(strip=True)
                    link = urllib.parse.urljoin(base_url, title_tag['href'])

                    if title in existing_titles or 'PHOTO:' in title or link.endswith('.jpg'):
                        continue  # ข้ามข่าวซ้ำหรือข่าวรูปภาพ

                    date_tag = article.select_one("div.TimeStamp__Date")
                    date = date_tag.get_text(strip=True) if date_tag else 'No Date'

                    if 'hours ago' in date or 'minutes ago' in date:
                        date = datetime.today().strftime('%d %b %Y')
                    else:
                        date_obj = datetime.strptime(date.replace(',', ''), '%B %d %Y')
                        date = date_obj.strftime('%d %b %Y')

                    desc_tag = article.select_one("div.ContentRoll__Desc")
                    description = desc_tag.get_text(strip=True) if desc_tag else 'No Description'

                    news_data.append({
                        "Title": title.replace(',', ''),
                        "Link": link,
                        "Date": date,
                        "Description": description.replace(',', '')
                    })
                    existing_titles.add(title)

                except Exception:
                    continue  # ข้าม error และไปต่อ

            # 💾 บันทึกข่าวทุก ๆ 5 ข่าว
            if len(news_data) >= 5:
                df = pd.DataFrame(news_data)
                df.to_csv(file_name, mode='a', header=not os.path.exists(file_name), index=False)
                print(f"💾 Total News Saved: {len(existing_titles)}")
                news_data = []

            # 🚀 ไปหน้าถัดไป
            page += 1
            next_url = f'https://abcnews.go.com/search?searchtext=Finance&sort=date&page={page}'
            driver.get(next_url)

    except Exception:
        pass  # ไม่ต้องแสดง error log
    finally:
        driver.quit()

        # 💾 บันทึกข่าวที่เหลือ
        if news_data:
            df = pd.DataFrame(news_data)
            df.to_csv(file_name, mode='a', header=not os.path.exists(file_name), index=False)
            print(f"💾 Total News Saved: {len(existing_titles)}")

        print("✅ Scraping Completed.")

scrape_abc_news()
