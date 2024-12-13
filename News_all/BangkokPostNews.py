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
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def scrape_bangkok_post_selenium(query):
    # ตั้งค่า Chrome options
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--ignore-certificate-errors')  # ข้ามข้อผิดพลาด SSL certificate
    # options.add_argument('--headless')  # ใช้โหมด headless ถ้าต้องการให้ทำงานเร็วขึ้น
    options.add_argument('--disable-software-rasterizer')  # แก้ปัญหากับ GPU

    # เริ่มต้น Chrome driver ด้วย options ที่ตั้งไว้
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # URL สำหรับการค้นหาข่าว
    base_url = f'https://search.bangkokpost.com/search/result?q={query}&category=news'
    driver.get(base_url)

    news_data = []
    save_counter = 1  # ใช้ counter สำหรับนับการเซฟข้อมูล
    file_name = 'bangkok_post_news.csv'

    try:
        while True:  # Loop ผ่านแต่ละหน้าผลลัพธ์
            # รอให้ผลการค้นหาปรากฏขึ้นบนหน้าเว็บ
            WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'mk-listnew--title'))  # คลาสที่ใช้แสดงผลข่าว
            )

            # เลื่อนหน้าเพื่อโหลดเนื้อหามากขึ้น
            for _ in range(5):
                driver.execute_script("window.scrollBy(0, 1000);")
                time.sleep(2)

            # ดึง HTML ของหน้าเพจที่โหลดแล้ว
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # หาข่าวจากคลาส mk-listnew--title
            articles = soup.find_all('div', class_='mk-listnew--title')

            if not articles:
                print("ไม่พบข่าว.")
                break

            # เก็บข้อมูลจากแต่ละข่าว
            for article in articles:
                try:
                    # ดึงลิงก์ของข่าวจาก <h3><a>
                    title_tag = article.find('h3').find('a')
                    title = title_tag.get_text(strip=True)  # Title extraction
                    link = title_tag['href']

                    # ถ้าลิงก์เป็น redirect ให้ดึงลิงก์จริง
                    if 'track/visitAndRedirect' in link:
                        url_params = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
                        real_link = url_params['href'][0]
                    else:
                        real_link = link

                    # เข้าไปที่ลิงก์ของข่าวเพื่อดึงเนื้อหาและวันที่
                    driver.get(real_link)
                    time.sleep(3)  # Allow time for the article page to load

                    # Extract the content and date from the article page
                    page_soup = BeautifulSoup(driver.page_source, 'html.parser')

                    # หาข้อมูลวันที่จาก <span> หรือ <p> ที่แสดงวันที่
                    # ดึงข้อมูลวันที่จาก <p> ใน <div class="article-info--col">
                    date_tag = page_soup.find('div', class_='article-info--col').find('p', text=lambda x: x and 'PUBLISHED :' in x)
                    if date_tag:
                        date = date_tag.get_text(strip=True).replace('PUBLISHED :', '').strip()  # ตัดคำ 'PUBLISHED :' ออก
                    else:
                        date = 'No Date'


                    # ดึงเนื้อหาภายใน <article>
                    article_content = page_soup.find_all('article')  # หาทุก <article> ในหน้า

                    # กรองเนื้อหาภายใน <p> ของแต่ละ <article>
                    for article in article_content:
                        # ตรวจสอบว่า <article> มี <div class="article-content"> หรือไม่
                        content_div = article.find('div', class_='article-content')  # หาจาก div.article-content

                        if content_div:
                            # ถ้ามี <div class="article-content"> ให้ดึง <p> ภายใน div นี้
                            content_paragraphs = content_div.find_all('p')
                        else:
                            # ถ้าไม่มี <div class="article-content"> ให้ดึง <p> จาก <article> ทั้งหมด
                            content_paragraphs = article.find_all('p')

                        # รวมข้อความจาก <p> ใน <article> หรือ <div class="article-content">
                        full_content = '\n'.join([p.get_text(strip=True) for p in content_paragraphs])

                        if not full_content:
                            full_content = 'Content not found'

                    news_data.append({
                        "Title": title,
                        "Link": real_link,
                        "Published Date": date,
                        "Content": full_content
                    })

                except Exception as e:
                    print(f"เกิดข้อผิดพลาดกับข่าว: {e}")
                    continue

                # Save every 5 rows
                if len(news_data) % 5 == 0:
                    # Save to the same file if it exists, or create new if not
                    if os.path.exists(file_name):
                        news_df = pd.DataFrame(news_data)
                        news_df.to_csv(file_name, mode='a', header=False, index=False)  # Append mode
                    else:
                        news_df = pd.DataFrame(news_data)
                        news_df.to_csv(file_name, index=False)  # Write new file if it doesn't exist
                    save_counter += 1  # Increment the counter after saving
                    news_data = []  # Reset after saving

            # หาและคลิก "Next" page link
            next_page = soup.find('a', text='Next')
            if next_page and 'href' in next_page.attrs:
                next_page_url = next_page['href']
                driver.get(next_page_url)  # ไปที่หน้าถัดไป
                time.sleep(3)  # รอหน้าเพจโหลด
            else:
                break  # ถ้าไม่มี "Next" page link ก็หยุด

        # ปิดเบราว์เซอร์
        driver.quit()

        # Save any remaining data after the loop
        if news_data:
            news_df = pd.DataFrame(news_data)
            news_df.to_csv(file_name, mode='a', header=False, index=False)

        return pd.DataFrame(news_data)

    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
        driver.quit()
        return pd.DataFrame()

# ตัวอย่างการใช้งานฟังก์ชัน
query = 'Stock'
news_df = scrape_bangkok_post_selenium(query)

if not news_df.empty:
    print(news_df)
else:
    print("ไม่พบข้อมูลข่าว.")
