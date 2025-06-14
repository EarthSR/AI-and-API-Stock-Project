import pandas as pd
import mysql.connector

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="TradeMine",
        autocommit=True
    )
    print("✅ เชื่อมต่อฐานข้อมูลสำเร็จ!")
except mysql.connector.Error as e:
    print(f"❌ การเชื่อมต่อฐานข้อมูลล้มเหลว: {e}")
    exit()

cursor = conn.cursor()
query = """
SELECT Title AS title, URL AS link, Content AS description, PublishedDate AS date , Img AS image
FROM News 
WHERE Source = 'BangkokPost' 
AND PublishedDate BETWEEN '2025-05-02' AND CURDATE()
"""
try:
    cursor.execute(query)
    news_data = cursor.fetchall()

    # Get column names from cursor
    columns = [desc[0] for desc in cursor.description]
    news_df = pd.DataFrame(news_data, columns=columns)

    if news_df.empty:
        print("⚠️ ไม่พบข้อมูลสำหรับช่วงวันที่ระบุ")
    else:
        print(news_df.head())  # Preview first few rows
        news_df.to_csv('Thai_News.csv', index=False, encoding='utf-8')
        print("✅ บันทึกข้อมูลฐานข้อมูลเป็นไฟล์ CSV เรียบร้อย")
except mysql.connector.Error as e:
    print(f"❌ เกิดข้อผิดพลาดในการรันคำสั่ง SQL: {e}")
finally:
    cursor.close()
    conn.close()
    print("✅ ปิดการเชื่อมต่อฐานข้อมูลเรียบร้อย")