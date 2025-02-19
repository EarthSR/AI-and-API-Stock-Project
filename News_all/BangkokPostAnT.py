import os
import time
import urllib.parse
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

def scrape_bangkok_post_selenium(query):
    # 🔹 ตั้งค่า Chrome options
    options = Options()
    options.add_argument('--headless')  # รันแบบไม่แสดงหน้าต่างเบราว์เซอร์
    options.add_argument('--disable-gpu')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--blink-settings=imagesEnabled=false')  # ปิดการโหลดรูปภาพ

    # 🔹 เริ่มต้น Chrome driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # 🔹 URL ของหน้าค้นหาข่าว
    #base_url = f'https://search.bangkokpost.com/search/result?publishedDate=&q=&category=news&sort=newest&rows=10&refinementFilter=AQhidXNpbmVzcwxjaGFubmVsYWxpYXMBAV4BJA%3D%3D'
    #base_url = f'https://search.bangkokpost.com/search/result?publishedDate=&q=&category=news&sort=newest&rows=10&refinementFilter=AQppbnZlc3RtZW50DGNoYW5uZWxhbGlhcwEBXgEk'
    #base_url = f'https://search.bangkokpost.com/search/result?publishedDate=&q=&category=news&sort=newest&rows=10&refinementFilter=AQhtb3RvcmluZwxjaGFubmVsYWxpYXMBAV4BJA%3D%3D'
    base_url = f'https://search.bangkokpost.com/search/result?publishedDate=&q=&category=news&sort=newest&rows=10&refinementFilter=AQdnZW5lcmFsDGNoYW5uZWxhbGlhcwEBXgEk'
    driver.get(base_url)

    news_data = []
# 🔹 กำหนดเส้นทางสำหรับไฟล์ CSV
    #file_name = 'D:/Stock_Project/AI-and-API-Stock-Project/news_data/bangkok_post_news.csv'
    #file_name = 'D:/Stock_Project/AI-and-API-Stock-Project/news_data/bangkok_post_news2.csv'
    #file_name = 'D:/Stock_Project/AI-and-API-Stock-Project/news_data/bangkok_post_news3.csv'
    file_name = 'D:/Stock_Project/AI-and-API-Stock-Project/news_data/bangkok_post_news4.csv'

    # 🔹 โหลดข่าวที่เคยบันทึกไปแล้ว
    existing_titles = set()
    if os.path.exists(file_name):
        try:
            df_existing = pd.read_csv(file_name, usecols=['Title'])
            existing_titles = set(df_existing['Title'].astype(str))
        except Exception as e:
            print(f"⚠️ ไม่สามารถโหลดไฟล์ CSV ที่มีอยู่ได้: {e}")

    try:
        while True:
            # 🔹 รอให้เนื้อหาข่าวโหลด (ลด Timeout ลงเหลือ 10 วินาที)
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'mk-listnew--title'))
                )
            except Exception:
                print("❌ Timeout: ไม่สามารถโหลดข่าวในหน้านี้ได้")
                break

            # 🔹 ดึง HTML ของหน้า
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            articles = soup.find_all('div', class_='mk-listnew--title')

            if not articles:
                print("❌ ไม่พบข่าว.")
                break

            # 🔹 ดึงข้อมูลจากแต่ละข่าว
            for article in articles:
                try:
                    title_tag = article.find('h3').find('a')
                    title = title_tag.get_text(strip=True)
                    link = title_tag['href']

                    # 🔹 ข้ามข่าวซ้ำ
                    if title in existing_titles:
                        continue

                    # 🔹 ดึงลิงก์จริง
                    if 'track/visitAndRedirect' in link:
                        url_params = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
                        real_link = url_params['href'][0]
                    else:
                        real_link = link

                    # 🔹 ใช้ requests แทน Selenium เพื่อลดเวลาโหลด
                    try:
                        response = requests.get(real_link, timeout=5)
                        news_soup = BeautifulSoup(response.content, 'html.parser')
                    except requests.exceptions.RequestException:
                        print(f"⚠️ ไม่สามารถโหลดข่าว: {title}")
                        continue

                    # 🔹 ดึงวันที่
                    date_tag = news_soup.find('div', class_='article-info--col')
                    date = date_tag.find('p', string=lambda x: x and 'PUBLISHED :' in x).get_text(strip=True).replace('PUBLISHED :', '').strip() if date_tag else 'No Date'

                    # 🔹 ดึงเนื้อหาข่าว
                    content_div = news_soup.find('div', class_='article-content')
                    paragraphs = content_div.find_all('p') if content_div else []
                    full_content = '\n'.join([p.get_text(strip=True) for p in paragraphs])

                    if not full_content:
                        full_content = 'Content not found'

                    # 🔹 บันทึกข่าวนี้ลงใน dataset
                    news_data.append({
                        "Title": title,
                        "Link": real_link,
                        "Date": date,
                        "Content": full_content
                    })
                    existing_titles.add(title)

                except Exception as e:
                    print(f"⚠️ ข้อผิดพลาดกับข่าว: {e}")
                    continue

            # 🔹 บันทึกข่าวทุก ๆ 5 ข่าว
            if len(news_data) >= 5:
                df = pd.DataFrame(news_data)
                df.to_csv(file_name, mode='a', header=not os.path.exists(file_name), index=False)
                print(f"💾 Total News Saved: {len(existing_titles)}")
                news_data = []

            # 🔹 หาและคลิก Next page
            next_page = soup.find('a', text='Next')
            if next_page and 'href' in next_page.attrs:
                driver.get(next_page['href'])
            else:
                break

        # 🔹 บันทึกข่าวที่เหลือหลังจากวนลูปจบ
        if news_data:
            df = pd.DataFrame(news_data)
            df.to_csv(file_name, mode='a', header=not os.path.exists(file_name), index=False)
            print(f"✅ บันทึกข่าวที่เหลืออีก {len(news_data)} ข่าว")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
    finally:
        driver.quit()

# เรียกใช้งานฟังก์ชัน
query = 'Stock'
scrape_bangkok_post_selenium(query)
