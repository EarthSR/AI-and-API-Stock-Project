from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
from bs4 import BeautifulSoup
import pandas as pd

def scrape_google_news_selenium(query):
    service = Service(ChromeDriverManager().install())
    
    options = Options()
    # options.add_argument('--headless')  # เปิดโหมด headless หากต้องการ
    options.add_argument('--disable-gpu')
    
    driver = webdriver.Chrome(service=service, options=options)

    driver.get(f'https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US%3Aen')

    try:
        # รอโหลดหน้าเว็บให้เสร็จ
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'JtKRv'))  # ใช้คลาสที่ถูกต้อง
        )

        # เลื่อนหน้าเพื่อโหลดข่าวเพิ่มเติม
        for _ in range(5):
            driver.execute_script("window.scrollBy(0, 1000);")
            time.sleep(2)

        # ดึงข้อมูล HTML
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # ดูโครงสร้าง HTML ของหน้า
        print(driver.page_source)  # พิมพ์ HTML ของหน้าเว็บ

        # ค้นหาข่าวจากลิงก์ที่มีคลาส JtKRv
        articles = soup.find_all('a', class_='JtKRv')

        if not articles:
            print("No articles found.")
            driver.quit()
            return pd.DataFrame()

        news_data = []
        for article in articles:
            # ดึงชื่อข่าวจากข้อความในแท็ก <a>
            title = article.get_text().strip()
            link = article['href']
            
            # ถ้าลิงก์เริ่มต้นด้วย './' ให้เพิ่ม 'https://news.google.com' เพื่อสร้างลิงก์เต็ม
            if link.startswith('./'):
                link = f'https://news.google.com{link[1:]}'

            news_data.append({"Title": title, "Link": link})

        driver.quit()
        return pd.DataFrame(news_data)

    except Exception as e:
        print(f"Error occurred: {e}")
        driver.quit()
        return pd.DataFrame()

# ตัวอย่างการดึงข่าว
query = 'Stock'
news_df = scrape_google_news_selenium(query)

if not news_df.empty:
    print(news_df)
    news_df.to_csv('google_news_links.csv', index=False)
else:
    print("No news data found.")
