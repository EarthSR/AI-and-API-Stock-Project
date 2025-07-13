import pandas as pd
import os
import pandas as pd
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ===== STEP 1: Load Dataset =====
file_path = os.path.join(os.path.dirname(__file__), "News", "Thai_News_Hybrid.csv")  # เปลี่ยนเป็น path ที่คุณเก็บไฟล์
df = pd.read_csv(file_path)

# ===== STEP 2: Clean & Prepare =====
# จัดการค่า null และทำความสะอาดข้อมูล
df['MatchedStock'] = df['MatchedStock'].fillna('').str.strip()
df['Sentiment'] = df['Sentiment'].fillna('Neutral')
df['Confidence'] = df['Confidence'].fillna(0.0)
df['date'] = pd.to_datetime(df['date']).dt.date

# แยก MatchedStock และตัดช่องว่าง
df_exploded = df.assign(Stock=df['MatchedStock'].str.split(', ')).explode('Stock')
df_exploded = df_exploded[df_exploded['Stock'] != '']
df_exploded['Stock'] = df_exploded['Stock'].str.strip()

# ตรวจสอบข้อมูลหลัง explode
print("Data after explode (first 20 rows):")
print(df_exploded[['date', 'Stock', 'Sentiment', 'Confidence']].head(20))
print(f"Total rows after explode: {len(df_exploded)}")

# ลบข้อมูลซ้ำโดยใช้ date, Stock, Sentiment, Confidence
df_exploded = df_exploded.drop_duplicates(subset=['date', 'Stock', 'Sentiment', 'Confidence'])

# ตรวจสอบข้อมูลหลังลบซ้ำ
print("Data after deduplication (first 20 rows):")
print(df_exploded[['date', 'Stock', 'Sentiment', 'Confidence']].head(20))
print(f"Total rows after deduplication: {len(df_exploded)}")

# บันทึกข้อมูลกลางเพื่อตรวจสอบ
df_exploded.to_csv('exploded_data.csv', index=False)

# ===== STEP 3: Group & Summarize =====
# รวมข้อมูลสำหรับหุ้นเดียวกันในวันเดียวกัน
daily_sentiment = df_exploded.groupby(['date', 'Stock']).agg(
    positive_news=('Sentiment', lambda x: (x == 'Positive').sum()),
    negative_news=('Sentiment', lambda x: (x == 'Negative').sum()),
    neutral_news=('Sentiment', lambda x: (x == 'Neutral').sum()),
    avg_confidence=('Confidence', 'mean'),  # เปลี่ยนจาก 'weight' เป็น 'mean'
    total_news=('Sentiment', 'count')
).reset_index()

# ตรวจสอบข้อมูลหลังรวม
print("Data after groupby (first 20 rows):")
print(daily_sentiment.head(20))
print(f"Total rows after groupby: {len(daily_sentiment)}")

# ===== STEP 4: Feature Engineering =====
# คำนวณเมตริกเพิ่มเติม
daily_sentiment['positive_ratio'] = daily_sentiment['positive_news'] / daily_sentiment['total_news']
daily_sentiment['negative_ratio'] = daily_sentiment['negative_news'] / daily_sentiment['total_news']

# ปรับ net_sentiment_score ให้เป็น -1, 0, 1
def calculate_net_sentiment(row):
    threshold = 0.4  # ปรับได้ตามความต้องการ
    if row['positive_ratio'] > row['negative_ratio'] and row['positive_ratio'] > threshold:
        return 1
    elif row['negative_ratio'] > row['positive_ratio'] and row['negative_ratio'] > threshold:
        return -1
    else:
        return 0

daily_sentiment['net_sentiment_score'] = daily_sentiment.apply(calculate_net_sentiment, axis=1)
daily_sentiment['has_news'] = daily_sentiment['total_news'].apply(lambda x: 1 if x > 0 else 0)

# กรอง Confidence หลังการรวม (ถ้าต้องการ)
# daily_sentiment = daily_sentiment[daily_sentiment['avg_confidence'] > 0.7]


# ===== STEP 5: Export Summary =====
output_path = os.path.join(os.path.dirname(__file__),"News" ,"daily_sentiment_summary_Thai.csv")
daily_sentiment.to_csv(output_path, index=False)

print("✅ Summary file saved → daily_sentiment_summary_Thai.csv")
