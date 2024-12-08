from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import pandas as pd
import threading

# ตั้งค่า WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# เปิดเว็บ
url = 'https://www.investing.com/news'
driver.get(url)

# รอให้หน้าเว็บโหลด
time.sleep(3)

def check_and_close_dialog(driver):
    try:
        # ตรวจสอบและรอให้ปุ่มปิดปรากฏ
        close_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//svg[@data-test='sign-up-dialog-close-button']"))
        )
        # คลิกปุ่มปิด
        close_button.click()
        print("Dialog closed successfully.")
    except Exception as e:
        print(f"Error finding or clicking the close button: {e}")

# ฟังก์ชั่นที่จะเช็คปุ่มปิดตลอดเวลา
def continuously_check_for_dialog(driver, check_interval=5):
    while True:
        check_and_close_dialog(driver)
        time.sleep(check_interval)  # รอเวลา 5 วินาที ก่อนที่จะเช็คอีกครั้ง
        
# สร้างฟังก์ชั่นที่ทำงานพร้อมกันกับการดึงข่าว
def start_dialog_checking(driver):
    continuously_check_for_dialog(driver)

# สร้าง Thread สำหรับการตรวจสอบ Dialog
dialog_thread = threading.Thread(target=start_dialog_checking, args=(driver,))
dialog_thread.daemon = True  # ให้ Thread นี้หยุดเมื่อโปรแกรมหลักหยุด
dialog_thread.start()

# รอให้เมนู Stock Markets ปรากฏ
stock_markets_li = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, "//li[@data-test='Stock-Markets']"))
)
stock_markets_li.click()

# คลิกที่ <li> เพื่อเปิดเมนู
stock_markets_li.click()

# รอให้เมนูย่อยแสดง
time.sleep(1)

# เลื่อนหน้าจนข่าวทั้งหมดโหลดครบ
last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    # เลื่อนหน้าลงไปที่ท้ายสุด
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 1000);")

    time.sleep(3)  # รอให้ข้อมูลโหลดเพิ่ม

    # ตรวจสอบความสูงใหม่
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break  # ถ้าความสูงเท่าเดิม แสดงว่าโหลดข่าวครบแล้ว
    last_height = new_height

# แปลง HTML เป็น BeautifulSoup object
soup = BeautifulSoup(driver.page_source, 'html.parser')

# ค้นหาแต่ละบทความข่าว
articles = soup.find_all('article', {'data-test': 'article-item'})

# เก็บข้อมูลลิงก์ของข่าวทั้งหมด
news_list = []
for article in articles:
    title_tag = article.find('a', {'data-test': 'article-title-link'})
    title = title_tag.get_text(strip=True) if title_tag else 'No Title'
    link = title_tag['href'] if title_tag and 'href' in title_tag.attrs else 'No Link'
    description_tag = article.find('p', {'data-test': 'article-description'})
    description = description_tag.get_text(strip=True) if description_tag else 'No Description'
    date_tag = article.find('time', {'data-test': 'article-publish-date'})
    date = date_tag['datetime'] if date_tag and 'datetime' in date_tag.attrs else 'No Date'
    news_list.append({'title': title, 'link': link, 'description': description, 'date': date,'content': 'No Content'})

# สร้างไฟล์ CSV และเขียนหัวข้อ
output_file = 'investing_news_full.csv'
df = pd.DataFrame(news_list)
df.to_csv(output_file, index=False, encoding='utf-8', mode='w', header=True)  # เขียน header ครั้งแรก

# เข้าไปในแต่ละลิงก์เพื่อเก็บข้อมูลเพิ่มเติม
# เข้าไปในแต่ละลิงก์เพื่อเก็บข้อมูลเพิ่มเติม
for idx, news in enumerate(news_list):
    try:
        driver.get(news['link'])  # เข้าไปที่ลิงก์ใหม่
        time.sleep(3)
        article_soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # ดึงเนื้อหาบทความ
        content_tag = article_soup.find('div', {'class': 'article_container'})
        content = ' '.join([p.get_text(strip=True) for p in content_tag.find_all('p')]) if content_tag else 'No Content'
        content = content.replace(',', '')
        # ตรวจสอบว่าเนื้อหาถูกดึงได้หรือไม่
        print(f"Content for article {idx+1}: {content}")  # พิมพ์เนื้อหาบทความย่อส่วนแรก 100 ตัวอักษร
        
        # ลบโฆษณาออก
        ads = article_soup.find_all('div', {'class': 'pt-5'})
        ads += article_soup.find_all('div', {'class': 'mt-3 text-center text-xs text-[#5b616e]'})
        ads += article_soup.find_all('div', {'class': 'mt-5 flex items-center justify-center space-x-12 ad_adgroup__SAKTz before:!-top-3 before:!bg-transparent before:!p-0 before:!normal-case before:tracking-wide before:!text-[#5B616E] after:hidden'})
        ads += article_soup.find_all('div', {'class': 'relative font-sans-v2 before:!text-3xs before:leading-3 ad_ad__II8vw ad_ad__label__2NPqI before:!-top-3 before:!bg-transparent before:!p-0 before:!normal-case before:tracking-wide before:!text-[#5B616E] after:hidden !mt-3 !pt-2'})
        for ad in ads:
            ad.decompose()  # ลบโฆษณาออกจากเนื้อหาบทความ
            
        # Update content field in the news dictionary
        news_list[idx]['content'] = content
        
        # ตรวจสอบว่ามีเนื้อหาหรือไม่
        if content == 'No Content':
            print(f"No content found for article {idx+1}")

        # บันทึกทุกๆ 5 ข่าวในไฟล์เดียวกัน
        if (idx + 1) % 5 == 0:
            temp_df = pd.DataFrame(news_list[:idx+1])
            temp_df.to_csv(output_file, index=False, encoding='utf-8', mode='a', header=False)
            print(f"Data saved to {output_file} after {idx+1} articles")
    except Exception as e:
        print(f"Error processing article {idx+1}: {e}")

# บันทึกข้อมูลทั้งหมดสุดท้ายในไฟล์เดียวกัน
df = pd.DataFrame(news_list)
df.to_csv(output_file, index=False, encoding='utf-8', mode='a', header=False)
print(f"Data saved to {output_file}")



# ปิด WebDriver
driver.quit()