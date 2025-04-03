import pandas as pd
import os
import mysql.connector


conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="TradeMine",
    autocommit=True
)
cursor = conn.cursor()
print("✅ เชื่อมต่อฐานข้อมูลสำเร็จ!")

query = "SELECT Title as title,URL as link, Content as description, PublishedDate as date FROM News WHERE Source = 'BangkokPost'"
cursor.execute(query)
news_data = cursor.fetchall()

# Convert to DataFrame with column names
columns = [desc[0] for desc in cursor.description]
news_df = pd.DataFrame(news_data, columns=columns)

news_df.to_csv('news_db_thai.csv', index=False)
print("✅ บันทึกข้อมูลฐานข้อมูลเป็นไฟล์ CSV เรียบร้อย")

thai_df = pd.read_csv('./News/Thai_News.csv')
db_df = pd.read_csv('news_db_thai.csv')

thai_df['description'] = thai_df['description'].str.replace(',', ' ')
db_df['description'] = db_df['description'].str.replace(',', ' ')
thai_df['title'] = thai_df['title'].str.replace(',', ' ')
db_df['title'] = db_df['title'].str.replace(',', ' ')


combined_df = pd.concat([db_df, thai_df], ignore_index=True)

# ลบข้อมูลซ้ำ โดยใช้คอลัมน์ 'title'
combined_df = combined_df.drop_duplicates(subset='title', keep='first')

# ตรวจสอบช่วงวันที่
combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
min_date = combined_df['date'].min()
max_date = combined_df['date'].max()

print(f"ข้อมูลรวมมีตั้งแต่วันที่ {min_date.date()} ถึง {max_date.date()}")

# บันทึกผลลัพธ์
combined_df.to_csv('Combined_News_Thai.csv', index=False)

print("รวมไฟล์เสร็จแล้ว และบันทึกเป็น Combined_News_Thai.csv")
