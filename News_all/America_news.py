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
from datetime import datetime, timedelta

def scrape_abc_news(query):
    """
    ดึงข้อมูลข่าวจาก ABC News โดยใช้ Selenium และ BeautifulSoup
    ถ้าลิงก์มาจาก Good Morning America จะใช้วิธีดึงข้อมูลต่างจากเดิม
    """
    print("🔹 Initializing Web Scraper...")
    
    # 🔹 ตั้งค่า Chrome options
    options = Options()
    options.add_argument('--headless')  # รันแบบไม่แสดงหน้าต่างเบราว์เซอร์
    options.add_argument('--disable-gpu')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--blink-settings=imagesEnabled=false')
    
    # 🔹 เริ่มต้น Chrome driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    # 🔹 URL ของหน้าค้นหาข่าว
    page = 1
    base_url = f'https://abcnews.go.com/search?searchtext={query}&sort=date&page={page}'
    driver.get(base_url)
    
    print(f"🔹 Accessing {base_url}")
    
    news_data = []
    file_name = '../news_data/ABCNews.csv'
    
    # 🔹 โหลดข่าวที่เคยบันทึกไปแล้ว
    existing_titles = set()
    if os.path.exists(file_name):
        try:
            df_existing = pd.read_csv(file_name, usecols=['Title'])
            existing_titles = set(df_existing['Title'].astype(str))
            print("✅ Existing news loaded successfully.")
        except Exception as e:
            print(f"⚠️ Error loading existing CSV: {e}")
    
    try:
        while True:
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'ContentRoll__Headline'))
                )
                print("✅ Page loaded successfully.")
            except Exception:
                print("❌ Timeout: Unable to load news on this page.")
                break
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            articles = soup.find_all('div', class_='ContentRoll__Headline')
            
            if not articles:
                print("❌ No news articles found.")
                break
            
            for article in articles:
                try:
                    title_tag = article.find('a', class_='AnchorLink')
                    if not title_tag:
                        continue  # ข้ามถ้าไม่มีลิงก์ข่าว
                    title = title_tag.get_text(strip=True)
                    link = title_tag['href']
                    
                    if title in existing_titles or 'PHOTO:' in title or link.endswith('.jpg'):
                        continue  # ข้ามข่าวที่ซ้ำหรือเป็นข่าวรูปภาพหรือเป็นไฟล์ภาพโดยตรง
                    
                    real_link = urllib.parse.urljoin("https://abcnews.go.com", link)
                    
                    try:
                        if "/video" in real_link:
                            print(f"⚠️ Detected video link: {real_link}")
                            continue  # ข้ามไปยังข่าวถัดไปเมื่อเจอวิดีโอ
                        driver.get(real_link)
                        WebDriverWait(driver, 15).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-testid="prism-article-body"]'))
                        )
                        news_soup = BeautifulSoup(driver.page_source, 'html.parser')
                    except Exception:
                        print(f"⚠️ Unable to fetch news: {title}")
                        continue
                    
                    
                    # ถ้าลิงก์มาจาก Good Morning America ให้ข้ามข่าวนั้นไป
                    if "goodmorningamerica" in real_link:
                        print(f"⚠️ Skipping news from goodmorningamerica: {title}")
                        continue
                    
                    # ถ้าลิงก์ไม่ได้มาจาก goodmorningamerica หรือมีวิดีโอ ให้ดึงข้อมูลจากหน้าอื่นๆ
                    date_tag = news_soup.find('time')  # ค้นหาจาก <time> tag ที่มักมีวันที่
                    if not date_tag:
                        date_tag = soup.find('div', class_='TimeStamp__Date')  # หากไม่พบ, ใช้ div อื่นๆ
                    date_text = date_tag.get_text(strip=True) if date_tag else 'No Date'
                    
                    # จัดการกับ "hours ago" หรือ "minutes ago" และแปลงเป็นวันที่จริง
                    if 'hour' in date_text or 'minute' in date_text:
                        # การคำนวณวันที่จากเวลาที่ผ่านมา
                        time_ago = int(date_text.split()[0])  # ดึงตัวเลขจาก "X hours ago" หรือ "X minutes ago"
                        date = datetime.today() - timedelta(hours=time_ago)
                        date = date.strftime('%d %b %Y')
                    else:
                        try:
                            date_obj = datetime.strptime(date_text.replace(',', ''), '%B %d %Y')
                            date = date_obj.strftime('%d %b %Y')
                        except ValueError:
                            print(f"⚠️ Invalid date format: {date_text}. Setting date to 'No Date'.")
                            date = 'No Date'
                    
                    content_div = news_soup.find('div', {'data-testid': 'prism-article-body'})
                    paragraphs = content_div.find_all('p') if content_div else []
                    full_content = '\n'.join([p.get_text(strip=True).replace(',', '') for p in paragraphs])
                    
                    if not full_content:
                        full_content = 'Content not found'
                    
                    news_data.append({
                        "Title": title.replace(',', ''),
                        "Link": real_link,
                        "Date": date,
                        "Description": full_content
                    })
                    existing_titles.add(title)
                    print(f"✅ Scraped: {title}")
                
                except Exception as e:
                    print(f"⚠️ Error processing article: {e}")
                    continue
            
            # 🔹 บันทึกข่าวทุก ๆ 5 ข่าว
            if len(news_data) >= 5:
                df = pd.DataFrame(news_data)
                df.to_csv(file_name, mode='a', header=not os.path.exists(file_name), index=False)
                print(f"💾 Total News Saved: {len(existing_titles)}")
                news_data = []
            
            # 🔹 ไปหน้าถัดไป
            page += 1
            next_url = f'https://abcnews.go.com/search?searchtext={query}&sort=date&page={page}'
            driver.get(next_url)
            print(f"➡️ Navigating to next page: {next_url}")
        
        # 🔹 บันทึกข่าวที่เหลือ
        if news_data:
            df = pd.DataFrame(news_data)
            df.to_csv(file_name, mode='a', header=not os.path.exists(file_name), index=False)
            print(f"✅ Saved remaining {len(news_data)} news articles.")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
    finally:
        driver.quit()
        print("🛑 Scraper Stopped.")

query = 'stock technology'
scrape_abc_news(query)
