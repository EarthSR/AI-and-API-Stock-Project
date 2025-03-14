import pandas as pd
import sys
import os

# ✅ ป้องกัน UnicodeEncodeError (ข้ามอีโมจิที่ไม่รองรับ)
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# ✅ กำหนด BASE_DIR เป็น path หลักของไฟล์ปัจจุบัน
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ ตั้งค่าโฟลเดอร์ปลายทางแบบ dynamic
OUTPUT_FOLDER = os.path.join(BASE_DIR, "Finbert")

def calculate_weighted_sentiment(csv_file_path, output_folder=OUTPUT_FOLDER):
    # ✅ โหลดข้อมูลจากไฟล์ CSV
    df = pd.read_csv(csv_file_path)

    # ✅ ตรวจสอบว่ามีคอลัมน์ที่ต้องใช้หรือไม่
    required_columns = {"Sentiment", "Confidence", "date", "Source"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"❌ CSV ต้องมีคอลัมน์ {required_columns}")

    # ✅ แปลงค่า Sentiment เป็นตัวเลข
    sentiment_mapping = {"Negative": -1, "Neutral": 0, "Positive": 1}
    df["Sentiment Score"] = df["Sentiment"].map(sentiment_mapping)

    # ✅ คำนวณ Weighted Sentiment Score
    df["Weighted Sentiment"] = df["Sentiment Score"] * df["Confidence"]

    # ✅ แปลงคอลัมน์วันที่ให้เป็น datetime
    df["date"] = pd.to_datetime(df["date"])

    # ✅ คำนวณ Weighted Sentiment Score รายวัน (รวมทุกแหล่งข่าว)
    daily_sentiment_all = df.groupby(df["date"].dt.date).apply(
        lambda x: x["Weighted Sentiment"].sum() / x["Confidence"].sum()
    ).reset_index(name="Final Sentiment Score")

    # ✅ คำนวณ Weighted Sentiment Score รายวันตาม `Source`
    sources = df["Source"].unique()
    daily_sentiment_by_source = {}

    for source in sources:
        source_df = df[df["Source"] == source]
        daily_sentiment = source_df.groupby(source_df["date"].dt.date).apply(
            lambda x: x["Weighted Sentiment"].sum() / x["Confidence"].sum()
        ).reset_index(name="Final Sentiment Score")
        daily_sentiment_by_source[source] = daily_sentiment

    # ✅ **Normalize Sentiment Score** (Min-Max Scaling) ทั้งหมด
    def normalize_sentiment(df_sentiment):
        min_score = df_sentiment["Final Sentiment Score"].min()
        max_score = df_sentiment["Final Sentiment Score"].max()
        if max_score != min_score:
            df_sentiment["Normalized Score"] = (df_sentiment["Final Sentiment Score"] - min_score) / (max_score - min_score)
        else:
            df_sentiment["Normalized Score"] = 0.5
        return df_sentiment

    daily_sentiment_all = normalize_sentiment(daily_sentiment_all)
    for source in sources:
        daily_sentiment_by_source[source] = normalize_sentiment(daily_sentiment_by_source[source])

    # ✅ ปรับช่วงเกณฑ์ใหม่
    def classify_sentiment(score):
        if score > 0.2:
            return "Positive"
        elif score < -0.2:
            return "Negative"
        else:
            return "Neutral"

    daily_sentiment_all["Sentiment Category"] = daily_sentiment_all["Final Sentiment Score"].apply(classify_sentiment)
    for source in sources:
        daily_sentiment_by_source[source]["Sentiment Category"] = daily_sentiment_by_source[source]["Final Sentiment Score"].apply(classify_sentiment)

    # ✅ **ป้องกัน Duplicate ก่อนบันทึก**
    def save_to_csv(df_sentiment, file_path):
        if os.path.exists(file_path):
            existing_data = pd.read_csv(file_path)
            combined_data = pd.concat([existing_data, df_sentiment]).drop_duplicates(subset=["date"], keep="last")  # ✅ ลบซ้ำ
        else:
            combined_data = df_sentiment
        
        combined_data.to_csv(file_path, index=False)
        print(f"📁 บันทึกผลลัพธ์ที่: {file_path}")

    # ✅ บันทึกผลลัพธ์รวมทั้งหมด
    all_output_file = os.path.join(output_folder, "daily_sentiment_result.csv")
    save_to_csv(daily_sentiment_all, all_output_file)

    # ✅ บันทึกผลลัพธ์แยกตาม `Source`
    for source in sources:
        source_lower = source.lower()
        if "bangkokpost" in source_lower:
            file_name = "daily_sentiment_result_th.csv"
        elif "investing" in source_lower:
            file_name = "daily_sentiment_result_us.csv"
        else:
            file_name = "daily_sentiment_result_other.csv"

        source_output_file = os.path.join(output_folder, file_name)
        save_to_csv(daily_sentiment_by_source[source], source_output_file)

# ✅ กำหนดพาธไฟล์ CSV ที่ต้องการวิเคราะห์
csv_file_path = "./news_with_sentiment_gpu.csv"
calculate_weighted_sentiment(csv_file_path)
