import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# ตั้งค่า Chrome ให้เร็วขึ้น 🚀
chrome_options = uc.ChromeOptions()
chrome_options.add_argument("--headless")  # รันแบบไม่แสดง GUI
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--blink-settings=imagesEnabled=false")  # ปิดการโหลดรูป
chrome_options.add_argument("--disable-gpu")  # ปิด GPU acceleration
chrome_options.add_argument("--disable-extensions")
# ใช้ undetected_chromedriver
driver = uc.Chrome(options=chrome_options)

# เปิดหน้าแรก
base_url = 'https://www.investing.com/news/stock-market-news'
driver.get(base_url)
time.sleep(3)  # รอให้หน้าเว็บโหลด

# ฟังก์ชันปิด popup ถ้ามี
def close_popup():
    try:
        close_button = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.XPATH, "//svg[@data-test='sign-up-dialog-close-button']"))
        )
        close_button.click()
        print("Popup closed.")
    except:
        pass  # ถ้าไม่มี popup ก็ข้ามไป

# ฟังก์ชันดึงข่าวจากหน้าเว็บ
def scrape_news():
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

# ดึงข่าวจากหลายหน้า
all_news = []
max_pages = 7499  # ตั้งค่าจำนวนหน้าที่ต้องการดึง
for page in range(1, max_pages + 1):
    print(f"Scraping page {page}...")

    # โหลดหน้าเว็บ
    if page > 1:
        page_url = f"{base_url}/{page}"
        driver.get(page_url)
        time.sleep(5)  # รอให้หน้าเว็บโหลด

    # ปิด popup ถ้ามี
    close_popup()

    # ดึงข่าวจากหน้านี้
    news = scrape_news()
    all_news.extend(news)

# บันทึกข้อมูลเป็น CSV
df = pd.DataFrame(all_news)
df.to_csv("investing_news_full.csv", index=False, encoding='utf-8')

# ปิด WebDriver
driver.quit()
print("Scraping complete. Data saved to investing_news_full.csv")
