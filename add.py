import pandas as pd

# โหลดไฟล์ CSV
file_path = "./news_with_sentiment_gpu.csv"

# อ่านข้อมูล
data = pd.read_csv(file_path, header=None)  # อ่านไฟล์โดยไม่มี header

# ตั้งชื่อคอลัมน์ใหม่
data.columns = ['title', 'description', 'Sentiment', 'Confidence']

# บันทึกไฟล์ใหม่พร้อมคอลัมน์
data.to_csv(file_path, index=False, header=True)

print(f"File {file_path} has been updated with columns: ['title', 'description', 'Sentiment', 'Confidence']")
