import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import os
# ตั้งค่า WebDriver
options = Options()
options.add_argument('--ignore-certificate-errors')  # ข้ามการตรวจสอบใบรับรอง SSL
# options.add_argument('--headless')  # ใช้โหมด headless (ไม่แสดงหน้าต่าง)

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# เปิดเว็บ
url = 'https://www.fool.com/tech-stock-news/'
driver.get(url)

# รอให้หน้าเว็บโหลด
time.sleep(3)


def load_all_articles():
    last_height = driver.execute_script("return document.body.scrollHeight")  # ขนาดของหน้าเว็บปัจจุบัน

    while True:
        try:
            # เลื่อนหน้าจอไปข้างล่าง
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 1300);")
            time.sleep(2)  # รอให้หน้าโหลดข้อมูลใหม่

            # ค้นหาปุ่ม "Load More" และคลิกถ้ามี
            load_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//span[text()='Load More']"))
            )
            load_more_button.click()  # คลิกปุ่ม "Load More"
            time.sleep(3)  # รอให้เนื้อหามีการโหลด

            # ตรวจสอบขนาดของหน้าจอหลังจากโหลดข้อมูลใหม่
            new_height = driver.execute_script("return document.body.scrollHeight")


            # อัปเดตขนาดของหน้าเพื่อใช้ในการตรวจสอบรอบถัดไป
            last_height = new_height


        except Exception as e:
            print(f"ไม่พบปุ่ม Load More หรือโหลดจนสุดแล้ว: {e}")
            break  # เมื่อไม่พบปุ่มหรือโหลดจนสุดแล้ว ให้หยุด

# เรียกใช้ฟังก์ชั่นเพื่อโหลดข่าวทั้งหมด
load_all_articles()





# แปลง HTML เป็น BeautifulSoup object
soup = BeautifulSoup(driver.page_source, 'html.parser')

# ค้นหาบทความข่าว
articles = soup.find_all('div', class_='flex py-12px text-gray-1100')

# เก็บข้อมูลข่าว
news_list = []
count = 0  # ตัวนับข่าว

for article in articles:
    title_tag = article.find('a')
    if title_tag:
        title = title_tag.get_text(strip=True)
        link = title_tag['href']
        
        # เช็คว่าลิงก์เป็นแบบเต็มหรือไม่ ถ้าไม่เติม 'https://www.fool.com'
        if link.startswith('/'):
            link = 'https://www.fool.com' + link
            
        img_tag = article.find('img')
        image_url = img_tag['src'] if img_tag else 'No Image'
        
        # คลิกเข้าไปในลิงก์เพื่อดึงเนื้อหาข่าว
        driver.get(link)
        time.sleep(3)  # รอหน้าเว็บใหม่โหลด
        
        # ดึงข้อมูลจากหน้าข่าว
        news_page_soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # ดึงเนื้อหาข่าว
        paragraphs = news_page_soup.find('div', class_='article-body').find_all('p')
        description = " ".join([p.get_text(strip=True) for p in paragraphs]) if paragraphs else 'No Content'
        
        # ดึงวันที่
        date_tag = news_page_soup.find('div', class_='text-lg font-medium text-gray-800 mt-12px md:mt-24px mb-24px')
        date = date_tag.get_text(strip=True).split(' – ')[0] if date_tag else 'No Date'
        
        # เก็บข้อมูลในลิสต์
        news_list.append({
            'title': title,
            'link': link,
            'image_url': image_url,
            'description': description,
            'date': date
        })
        
        count += 1
        # ทุกๆ 5 ข่าวให้บันทึกลง CSV
        
    if count % 5 == 0:
        # แปลงวันที่เป็นรูปแบบที่ต้องการ
        for news in news_list:
            # ลบ "By" ถึง "-" ในวันที่ข่าวและแทนที่จุลภาคด้วยช่องว่าง
            news['date'] = news['date'].split('–')[1].strip().replace(",", "")  # ตัดส่วนของ "By" และข้อความหลังจาก "–" และลบจุลภาค

        # สร้าง DataFrame และบันทึกข้อมูล
        df = pd.DataFrame(news_list)
        
        # ตรวจสอบว่าไฟล์มีอยู่แล้วหรือไม่
        file_exists = os.path.isfile('fool_tech_stock_news.csv')
        
        # หากไฟล์ยังไม่มี ให้เขียน header ด้วย
        df.to_csv('fool_tech_stock_news.csv', mode='a', index=False, header=not file_exists, lineterminator="\n")
        
        news_list.clear()  # ล้างข้อมูลหลังจากบันทึก

        
# ถ้ามีข่าวที่เหลืออยู่หลังจากรอบสุดท้าย ก็ให้บันทึก   
if news_list:
    df = pd.DataFrame(news_list)
    df.to_csv('fool_tech_stock_news.csv', mode='a', index=False, header=not file_exists, lineterminator="\n")

# ปิด WebDriver
driver.quit()
