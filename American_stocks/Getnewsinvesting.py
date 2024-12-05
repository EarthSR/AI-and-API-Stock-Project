from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import pandas as pd  # นำเข้า pandas เพื่อใช้งานในการบันทึก CSV

# ตั้งค่า WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# เปิดเว็บ
url = 'https://www.investing.com/news'
driver.get(url)

# รอให้หน้าเว็บโหลด
time.sleep(3)  # รอหน้าเว็บโหลด (ปรับเวลาได้ตามต้องการ)

# สร้างลูปเลื่อนหน้าเว็บ
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # เลื่อนหน้าลงไปที่ท้ายสุด
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)  # รอให้ข้อมูลโหลด

    # คำนวณความสูงใหม่ของหน้าเว็บ
    new_height = driver.execute_script("return document.body.scrollHeight")
    
    # ถ้าความสูงของหน้าไม่เพิ่มขึ้น แสดงว่าโหลดข้อมูลทั้งหมดแล้ว
    if new_height == last_height:
        break

    last_height = new_height

# แปลง HTML เป็น BeautifulSoup object
soup = BeautifulSoup(driver.page_source, 'html.parser')

# ค้นหาแต่ละบทความข่าว
articles = soup.find_all('article', {'data-test': 'article-item'})

# เก็บข้อมูลข่าว
news_list = []
for article in articles:
    # ดึงชื่อข่าวและลิงก์
    title_tag = article.find('a', {'data-test': 'article-title-link'})
    title = title_tag.get_text(strip=True) if title_tag else 'No Title'
    link = title_tag['href'] if title_tag and 'href' in title_tag.attrs else 'No Link'
    
    # ดึงคำอธิบาย
    description_tag = article.find('p', {'data-test': 'article-description'})
    description = description_tag.get_text(strip=True) if description_tag else 'No Description'
    
    # ดึงวันที่
    date_tag = article.find('time', {'data-test': 'article-publish-date'})
    date = date_tag['datetime'] if date_tag and 'datetime' in date_tag.attrs else 'No Date'
    
    # บันทึกข้อมูล
    news_list.append({
        'title': title,
        'link': link,
        'description': description,
        'date': date
    })

# แปลงข้อมูลเป็น DataFrame ของ pandas
df = pd.DataFrame(news_list)

# กำหนดชื่อไฟล์ CSV
output_file = 'financial_news.csv'

# บันทึกข้อมูลลงใน CSV
df.to_csv(output_file, index=False, encoding='utf-8')

# แสดงข้อความเมื่อบันทึกไฟล์สำเร็จ
print(f"Data saved to {output_file}")

# ปิด WebDriver
driver.quit()
