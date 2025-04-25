import pandas as pd
import os
import pandas as pd

# ===== STEP 1: Load Dataset =====
file_path = os.path.join(os.path.dirname(__file__), "News", "Thai_News_Hybrid.csv")  # เปลี่ยนเป็น path ที่คุณเก็บไฟล์
df = pd.read_csv(file_path)

# ===== STEP 2: Clean & Prepare =====
df['MatchedStock'] = df['MatchedStock'].fillna('')
df_exploded = df.assign(Stock=df['MatchedStock'].str.split(', ')).explode('Stock')
df_exploded = df_exploded[df_exploded['Stock'] != '']  # ตัดข่าวที่ไม่มีหุ้น

# ===== STEP 3: Group & Summarize =====
daily_sentiment = df_exploded.groupby(['date', 'Stock']).agg(
    positive_news=('Sentiment', lambda x: (x == 'Positive').sum()),
    negative_news=('Sentiment', lambda x: (x == 'Negative').sum()),
    neutral_news=('Sentiment', lambda x: (x == 'Neutral').sum()),
    avg_confidence=('Confidence', 'mean'),
    total_news=('Sentiment', 'count')
).reset_index()

# ===== STEP 4: Feature Engineering =====
daily_sentiment['positive_ratio'] = daily_sentiment['positive_news'] / daily_sentiment['total_news']
daily_sentiment['negative_ratio'] = daily_sentiment['negative_news'] / daily_sentiment['total_news']
daily_sentiment['net_sentiment_score'] = (daily_sentiment['positive_news'] - daily_sentiment['negative_news']) / daily_sentiment['total_news']
daily_sentiment['has_news'] = daily_sentiment['total_news'].apply(lambda x: 1 if x > 0 else 0)
daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date']).dt.date  # แปลง date เป็น datetime.date

# ===== STEP 5: Export Summary =====
output_path = os.path.join(os.path.dirname(__file__),"News" ,"daily_sentiment_summary_Thai.csv")
daily_sentiment.to_csv(output_path, index=False)

print("✅ Summary file saved → daily_sentiment_summary_Thai.csv")
