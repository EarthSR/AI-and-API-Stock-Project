import os
import pandas as pd
from datetime import datetime
import sys

# ✅ ป้องกัน UnicodeEncodeError (ข้ามอีโมจิที่ไม่รองรับ)
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

def clean_and_process_data(input_folder, output_file):
    all_data = []
    csv_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]

    if not csv_files:
        print(f"❌ ไม่พบไฟล์ CSV ในโฟลเดอร์ {input_folder}")
        return

    for file in csv_files:
        try:
            df = pd.read_csv(file, encoding='utf-8')
            df['Source'] = df['link'].apply(lambda x: 'Investing' if 'investing.com' in str(x) else 'BangkokPost' if 'bangkokpost.com' in str(x) else 'Unknown')
            all_data.append(df)
        except Exception as e:
            print(f"⚠️ ไม่สามารถโหลดไฟล์ {file}: {e}")
            continue

    df = pd.concat(all_data, ignore_index=True)

    print(f"\n🔍 [DEBUG] ค่าของ 'Source' หลังรวมไฟล์ {input_folder}:")
    print(df['Source'].value_counts(dropna=False))

    df_cleaned = df.drop_duplicates(subset=['title', 'link'], keep='first')

    # 🔹 ปรับรูปแบบวันที่
    df_cleaned.loc[:, 'date'] = pd.to_datetime(df_cleaned['date'], errors='coerce')
    df_cleaned = df_cleaned.dropna(subset=['date'])

    # 🔹 **ลบข่าวที่เก่ากว่า 01/01/2018**
    cutoff_date = datetime(2018, 1, 1)
    df_cleaned = df_cleaned[df_cleaned['date'] >= cutoff_date]

    print(f"\n🔍 [DEBUG] ค่าของ 'Source' ก่อนบันทึก {output_file}:")
    print(df_cleaned['Source'].value_counts(dropna=False))

    df_cleaned.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✅ [CLEANED] ไฟล์ถูกบันทึกที่: {output_file}\n")

def delete_csv_files_in_folder(folder_path, exclude_filename):
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv') and f != os.path.basename(exclude_filename)]
    for file in csv_files:
        try:
            os.remove(file)
            print(f"🗑️ ลบไฟล์: {file}")
        except Exception as e:
            print(f"⚠️ ไม่สามารถลบไฟล์ {file}: {e}")

# 🔹 ตั้งค่าโฟลเดอร์
bangkokpost_folder = 'D:\\Stock_Project\\AI-and-API-Stock-Project\\BangkokPost_Folder'
investing_folder = 'D:\\Stock_Project\\AI-and-API-Stock-Project\\Investing_Folder'
news_data_folder = 'D:\\Stock_Project\\AI-and-API-Stock-Project\\news_data'

bangkokpost_output_file = os.path.join(news_data_folder, 'Thai_News.csv')
investing_output_file = os.path.join(news_data_folder, 'USA_News.csv')
final_output_file = os.path.join(news_data_folder, 'Final_News.csv')

# 🔹 ลบ Final_News.csv ก่อนเริ่ม
if os.path.exists(final_output_file):
    os.remove(final_output_file)
    print("🗑️ ลบ Final_News.csv เก่าก่อนรวมใหม่")

# 🔹 รวมและทำความสะอาดไฟล์ข่าว
clean_and_process_data(bangkokpost_folder, bangkokpost_output_file)
clean_and_process_data(investing_folder, investing_output_file)

# 🔹 Debug ตรวจสอบไฟล์ใน news_data ก่อนรวม
print("\n🔍 [DEBUG] ตรวจสอบไฟล์ news_data ก่อนรวม:")
print(os.listdir(news_data_folder))

# 🔹 โหลดไฟล์ทั้งหมดใน news_data
final_dataframes = []
for file in os.listdir(news_data_folder):
    if file.endswith('.csv'):
        file_path = os.path.join(news_data_folder, file)
        df_temp = pd.read_csv(file_path, encoding='utf-8')

        print(f"\n🔍 [DEBUG] ค่าของ 'Source' ใน {file}:")
        print(df_temp['Source'].value_counts(dropna=False))

        final_dataframes.append(df_temp)

df_final = pd.concat(final_dataframes, ignore_index=True)

# 🔹 ตรวจสอบค่าซ้ำก่อนบันทึก
df_final = df_final.drop_duplicates(subset=['title', 'link'], keep='first')

# 🔹 **ลบข่าวที่เก่ากว่า 01/01/2018** ใน Final_News.csv
cutoff_date = datetime(2018, 1, 1)  # ✅ แก้ไขตรงนี้
df_final['date'] = pd.to_datetime(df_final['date'], errors='coerce')
df_final = df_final.dropna(subset=['date'])
df_final = df_final[df_final['date'] >= cutoff_date]

print("\n🔍 [DEBUG] ค่าของ 'Source' หลังรวมไฟล์ news_data:")
print(df_final['Source'].value_counts(dropna=False))

# 🔹 **สรุปจำนวนข่าวของแต่ละแหล่งและรวมทั้งหมด**
total_bangkokpost = df_final[df_final['Source'] == 'BangkokPost'].shape[0]
total_investing = df_final[df_final['Source'] == 'Investing'].shape[0]
total_news = df_final.shape[0]

print("\n📊 [SUMMARY REPORT]")
print(f"✅ BangkokPost: {total_bangkokpost} ข่าว")
print(f"✅ Investing: {total_investing} ข่าว")
print(f"✅ รวมทั้งหมด: {total_news} ข่าว")

df_final.to_csv(final_output_file, index=False, encoding='utf-8')

# 🔹 ลบไฟล์ CSV ที่ไม่ใช้
delete_csv_files_in_folder(news_data_folder, final_output_file)

print("\n🎯 [DONE] การรวมและทำความสะอาดข้อมูลเสร็จสมบูรณ์!")
