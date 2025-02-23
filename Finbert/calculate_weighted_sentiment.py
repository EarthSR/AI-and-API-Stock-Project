import pandas as pd

def calculate_weighted_sentiment(csv_file_path, output_file_path="weighted_sentiment_result.csv", daily_output_file="daily_sentiment_result.csv"):
    # โหลดข้อมูลจากไฟล์ CSV
    df = pd.read_csv(csv_file_path)

    # ตรวจสอบว่ามีคอลัมน์ที่ต้องใช้หรือไม่
    required_columns = {"Sentiment", "Confidence", "date"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV ต้องมีคอลัมน์ {required_columns}")

    # แปลงค่า Sentiment เป็นตัวเลข
    sentiment_mapping = {"Negative": -1, "Neutral": 0, "Positive": 1}
    df["Sentiment Score"] = df["Sentiment"].map(sentiment_mapping)

    # คำนวณ Weighted Sentiment Score
    df["Weighted Sentiment"] = df["Sentiment Score"] * df["Confidence"]

    # คำนวณค่า Final Sentiment Score รวมทั้งหมด
    final_sentiment = df["Weighted Sentiment"].sum() / df["Confidence"].sum()

    # แปลงคอลัมน์วันที่ให้เป็น datetime
    df["date"] = pd.to_datetime(df["date"])

    # คำนวณ Weighted Sentiment Score รายวัน
    daily_sentiment = df.groupby(df["date"].dt.date).apply(
        lambda x: x["Weighted Sentiment"].sum() / x["Confidence"].sum()
    ).reset_index(name="Final Sentiment Score")

    # **Normalize Sentiment Score** (Min-Max Scaling)
    min_score = daily_sentiment["Final Sentiment Score"].min()
    max_score = daily_sentiment["Final Sentiment Score"].max()
    
    if max_score != min_score:  # ป้องกันการหารด้วยศูนย์
        daily_sentiment["Normalized Score"] = (daily_sentiment["Final Sentiment Score"] - min_score) / (max_score - min_score)
    else:
        daily_sentiment["Normalized Score"] = 0.5  # ถ้าค่าทุกวันเหมือนกัน ให้ใช้กลางๆ

    # ปรับช่วงเกณฑ์ใหม่
    def classify_sentiment(score):
        if score > 0.2:
            return "Positive"
        elif score < -0.2:
            return "Negative"
        else:
            return "Neutral"

    daily_sentiment["Sentiment Category"] = daily_sentiment["Final Sentiment Score"].apply(classify_sentiment)

    # บันทึกผลลัพธ์ลงไฟล์ CSV
    df.to_csv(output_file_path, index=False)
    daily_sentiment.to_csv(daily_output_file, index=False)

    print(f"Final Sentiment Score (Overall): {final_sentiment:.4f}")
    print(f"ผลลัพธ์ทั้งหมดถูกบันทึกลงในไฟล์: {output_file_path}")
    print(f"ผลลัพธ์รายวันถูกบันทึกลงในไฟล์: {daily_output_file}")

    return final_sentiment, daily_sentiment

# กำหนดพาธไฟล์ CSV ที่ต้องการวิเคราะห์
csv_file_path = "../Finbert/news_with_sentiment_gpu.csv"  
calculate_weighted_sentiment(csv_file_path)
