from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
from bs4 import BeautifulSoup
import time
import os
import pandas as pd
from datetime import datetime

# ตั้งค่า WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# เปิดเว็บ
url = 'https://www.imf.org/en/News/SearchNews#'
driver.get(url)

# รอให้หน้าเว็บโหลด
time.sleep(3)

# รอให้หน้าเว็บโหลด
driver.refresh()

# รอให้หน้าเว็บโหลด
time.sleep(3)

folder_path = "../News_all/imf_news_data"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# ดำเนินการสำหรับปีที่กำหนด
start_year = 2024
end_year = 2024

# Get today's date to compare later
today = datetime.today()
today_year = today.year
today_month = today.month
today_day = today.day

news_list = []

for year in range(start_year, end_year + 1):
    try:
        # Check if we are in the current year and date
        if year == today_year:
            # If it's the current year, don't go beyond today's date
            end_month = today_month
            end_day = today_day
        else:
            # If it's not the current year, set the end date to December 30th
            end_month = 12
            end_day = 30

        # รอให้ date input ของ start date พร้อมใช้งาน
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "dtStartDateDisplay")))

        # ตั้งค่า Start Date
        start_date_input = driver.find_element(By.ID, "dtStartDateDisplay")
        start_date_input.click()

        # เลือกเดือน มกราคม
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "ui-datepicker-month")))
        month_dropdown = driver.find_element(By.CLASS_NAME, "ui-datepicker-month")
        Select(month_dropdown).select_by_value("0")  # เดือน 0 คือ มกราคม

        # เลือกปี
        year_dropdown = driver.find_element(By.CLASS_NAME, "ui-datepicker-year")
        Select(year_dropdown).select_by_value(str(year))

        # เลือกวัน
        day_element = driver.find_element(By.XPATH, f"//a[@class='ui-state-default' and text()='1']")
        day_element.click()

        # กำหนด End Date
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "dtEndDateDisplay")))

        # Set the end date based on the current year or the specified end year
        end_date_input = driver.find_element(By.ID, "dtEndDateDisplay")
        end_date_input.click()

        # เลือกเดือน
        month_dropdown = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "ui-datepicker-month")))
        Select(month_dropdown).select_by_value(str(end_month - 1))  # end_month - 1 because month dropdown is 0-indexed

        # เลือกปี
        year_dropdown = driver.find_element(By.CLASS_NAME, "ui-datepicker-year")
        Select(year_dropdown).select_by_value(str(year))

        # เลือกวัน
        day_element = driver.find_element(By.XPATH, f"//a[@class='ui-state-default' and text()='{end_day}']")
        day_element.click()

        # คลิก "Apply"
        filter_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "FilterButton")))

        # Click the button
        filter_button.click()        
        time.sleep(2)

        # ดึงข้อมูลในหน้านี้
        while True:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'coveo-result-item')))
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            articles = soup.find_all('div', {'class': 'coveo-result-item'})

            for article in articles:
                date = article.find('div', {'class': 'CoveoFieldValue'}).get_text(strip=True) if article.find('div', {'class': 'CoveoFieldValue'}) else 'No Date'
                title = article.find('a', {'class': 'CoveoResultLink'}).get_text(strip=True) if article.find('a', {'class': 'CoveoResultLink'}) else 'No Title'
                link = article.find('a', {'class': 'CoveoResultLink'})['href'] if article.find('a', {'class': 'CoveoResultLink'}) else 'No Link'

                news_list.append({'date': date, 'title': title, 'link': link})

            try:
                next_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CLASS_NAME, "coveo-pager-next")))
                next_button.click()
                time.sleep(2)
            except:
                break  # หากไม่มีปุ่มถัดไป

        print(f"ดึงข้อมูลสำหรับวันที่ {year} สำเร็จ")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดสำหรับวันที่ {year}: {e}")
        
    

# บันทึกข้อมูลลงใน CSV หลังจบลูปทั้งหมด
df = pd.DataFrame(news_list)
df.to_csv(os.path.join(folder_path, 'imf_news_links_2024.csv'), index=False, encoding='utf-8')
print("Data saved to imf_news.csv")

df = pd.read_csv(os.path.join(folder_path, 'imf_news_links_2024.csv'))
# List to store full article data
full_news_list = []

# Iterate over each row (article) and visit its link to collect more data
for index, row in df.iterrows():
    try:
        # Visit the article's link
        driver.get(row['link'])
        time.sleep(2)  # Wait for the page to load

        # Parse the page
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Extract full article content
        content_section = soup.find('div', {'class': 'column-padding'})
        # content = content_section.get_text(strip=True) if content_section else 'No Content'
        # Extract the description (main body of the press release)
        description = ' '.join([p.get_text(strip=True) for p in soup.find_all('p')])

        # Append the data to the full_news_list
        full_news_list.append({
            'date': row['date'],
            'title': row['title'],
            'link': row['link'],
            # 'content': content,
            'description': description
        })

        print(f"Collected full content for article: {row['title']}")

    except Exception as e:
        print(f"Error fetching full content for article {row['title']}: {e}")

# Save full article data to CSV
full_df = pd.DataFrame(full_news_list)
full_df.to_csv(os.path.join(folder_path, 'imf_news_full_2024.csv'), index=False, encoding='utf-8')
print("Full article data saved to imf_news_full_2024.csv")
# ปิด WebDriver
driver.quit()
