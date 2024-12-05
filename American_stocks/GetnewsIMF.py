from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
from bs4 import BeautifulSoup
import time
import pandas as pd

# ตั้งค่า WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# เปิดเว็บ
url = 'https://www.imf.org/en/News/SearchNews#'
driver.get(url)

# รอให้หน้าเว็บโหลด (รอจนกว่า date input จะพร้อมใช้งาน)
WebDriverWait(driver, 25).until(EC.element_to_be_clickable((By.ID, "dtStartDateDisplay")))

# คลิกที่ช่องป้อนข้อมูล
date_input = driver.find_element(By.ID, "dtStartDateDisplay")
date_input.click()

# รอให้ dropdown ของเดือนโหลด
WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CLASS_NAME, "ui-datepicker-month")))

# เลือกเดือนและปี
month_dropdown = driver.find_element(By.CLASS_NAME, "ui-datepicker-month")
select = Select(month_dropdown)
select.select_by_value("11")  # เลือกเดือน พฤศจิกายน

# รอให้ dropdown ของปีโหลด
WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CLASS_NAME, "ui-datepicker-year")))
year_dropdown = driver.find_element(By.CLASS_NAME, "ui-datepicker-year")
select = Select(year_dropdown)
select.select_by_value("2014")  # เลือกปี 2014

pagination_link = driver.find_element(By.XPATH, "//a[@class='ui-state-default' and text()='5']")
pagination_link.click()

# สร้างลูปเลื่อนหน้าเว็บจนกว่าจะถึงหน้าสุดท้าย
news_list = []
while True:
    # รอให้ข้อมูลในหน้าเว็บโหลด
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'coveo-result-item')))
    
    # แปลง HTML เป็น BeautifulSoup object
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # ค้นหาแต่ละบทความข่าว
    articles = soup.find_all('div', {'class': 'coveo-result-item'})
    
    for article in articles:
        # ดึงวันที่
        date_tag = article.find('div', {'class': 'CoveoFieldValue'})
        date = date_tag.get_text(strip=True) if date_tag else 'No Date'
        
        # ดึงชื่อข่าวและลิงก์
        title_tag = article.find('a', {'class': 'CoveoResultLink'})
        title = title_tag.get_text(strip=True) if title_tag else 'No Title'
        link = title_tag['href'] if title_tag and 'href' in title_tag.attrs else 'No Link'
        
        # ดึงคำอธิบาย
        description_tag = article.find('div', {'class': 'CoveoResultLink'})
        description = description_tag.get_text(strip=True) if description_tag else 'No Description'
        
        # บันทึกข้อมูล
        news_list.append({
            'title': title,
            'link': link,
            'date': date,
            'description': description
        })
    
    # หาปุ่ม "Next" 
    try:
        next_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "coveo-pager-next"))
        )
        # คลิกที่ปุ่ม "Next"
        next_button.click()
        # รอให้หน้าถัดไปโหลด
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'coveo-result-item')))
    except:
        break  # ถ้าไม่มีหน้า "Next" ให้หยุดลูป

# แปลงข้อมูลเป็น DataFrame ของ pandas
df = pd.DataFrame(news_list)

# กำหนดชื่อไฟล์ CSV
output_file = 'imf_news.csv'

# บันทึกข้อมูลลงใน CSV
df.to_csv(output_file, index=False, encoding='utf-8')

# แสดงข้อความเมื่อบันทึกไฟล์สำเร็จ
print(f"Data saved to {output_file}")

# ปิด WebDriver
driver.quit()
