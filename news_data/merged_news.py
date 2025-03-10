import os
import pandas as pd
from datetime import datetime, timedelta

def merge_and_filter_csv(directory, output_file):
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    all_dfs = []

    # 🔹 คำนวณวันที่ย้อนหลัง 8 ปีจากวันนี้
    cutoff_date = datetime.today() - timedelta(days=8 * 365)

    for file in all_files:
        file_path = os.path.join(directory, file)
        try:
            df = pd.read_csv(file_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"⚠️ ไม่สามารถโหลดไฟล์ {file}: {e}")

    if not all_dfs:
        print("❌ ไม่พบไฟล์ CSV ที่รวมข้อมูลได้")
        return

    # 🔹 รวมข้อมูลทุกไฟล์เข้าด้วยกัน
    merged_df = pd.concat(all_dfs, ignore_index=True)

    # 🔹 จัดการคอลัมน์ Date
    if 'Date' in merged_df.columns:
        # สร้างคอลัมน์ใหม่สำหรับการประมวลผลวันที่
        merged_df['Processed Date'] = pd.to_datetime(
            merged_df['Date'].str.split(' at ').str[0],  # ตัดส่วนเวลาออก
            format='%d %b %Y',  # ระบุรูปแบบวันที่
            errors='coerce'
        )
        # ลบแถวที่ไม่มีวันที่
        merged_df = merged_df.dropna(subset=['Processed Date'])

    # 🔹 กรองข้อมูลที่ไม่เก่ากว่า 8 ปี
    merged_df = merged_df[merged_df['Processed Date'] >= cutoff_date]

    # 🔹 ลบข้อมูลซ้ำ (ใช้ Title และ Link เป็นตัวระบุ)
    merged_df = merged_df.drop_duplicates(subset=['Title', 'Link'], keep='first')

    # 🔹 เรียงลำดับวันที่จากใหม่ -> เก่า โดยยังเก็บคอลัมน์ Date เดิมไว้
    merged_df = merged_df.sort_values(by='Processed Date', ascending=False)

    # 🔹 ลบคอลัมน์ที่ใช้สำหรับประมวลผลวันที่
    merged_df = merged_df.drop(columns=['Processed Date'])

    # 🔹 บันทึกไฟล์ CSV ใหม่
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"✅ รวมข้อมูลและลบข่าวที่เก่ากว่า 8 ปีสำเร็จ! บันทึกที่: {output_file}")

# 🔹 ตั้งค่าโฟลเดอร์ที่เก็บไฟล์ CSV และชื่อไฟล์ผลลัพธ์
csv_directory = "D:/StockData/AI-and-API-Stock-Project/news_data"
filtered_output_csv = "D:/StockData/AI-and-API-Stock-Project/clean_news_csv/BangkokPost_Merge.csv"

# 🔹 เรียกใช้งานฟังก์ชัน
merge_and_filter_csv(csv_directory, filtered_output_csv)
