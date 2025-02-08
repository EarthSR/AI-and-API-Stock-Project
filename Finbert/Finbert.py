from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from tqdm import tqdm
import os
import torch
from multiprocessing import freeze_support

# ตรวจสอบ GPU
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model():
    print("Loading FinBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "yiyanghkust/finbert-tone"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "yiyanghkust/finbert-tone"
    )
    print("✅ Model and tokenizer loaded successfully!")
    return model, tokenizer

def main():
    # โหลดโมเดล
    model, tokenizer = load_model()

    # สร้าง Pipeline
    finbert_sentiment = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=512
    )

    # โหลดข้อมูล
    input_data = pd.read_csv("./bangkok_post_news.csv")
    input_data.dropna(subset=['title', 'description'], how='all', inplace=True)

    # รวมข้อความ
    financial_news = input_data.apply(
        lambda row: f"{row['title']} {row['description']}" if pd.notnull(row['description']) else row['title'], axis=1
    ).tolist()

    partial_results_path = "partial_results_gpu.csv"
    if not os.path.exists(partial_results_path):
        print("No partial results found. Starting fresh.")
        processed_data = pd.DataFrame(columns=['title', 'description', 'date', 'Sentiment', 'Confidence'])
        start_idx = 0
        processed_data.to_csv(partial_results_path, index=False, header=True, mode='w')  # สร้างไฟล์พร้อม Header
    else:
        processed_data = pd.read_csv(partial_results_path)
        start_idx = len(processed_data)
        print(f"Resuming from {start_idx} processed records.")

    # วิเคราะห์ข้อความ
    results = []
    try:
        for idx, news in enumerate(tqdm(financial_news[start_idx:], desc="Processing News")):
            chunks = [news]  # ใช้ข้อความตรงๆ หากไม่แบ่ง chunk
            chunk_results = finbert_sentiment(chunks)
            sentiment = chunk_results[0]['label']
            confidence = chunk_results[0]['score']

            results.append((input_data.iloc[start_idx + idx]['title'],
                            input_data.iloc[start_idx + idx]['description'],
                            input_data.iloc[start_idx + idx]['date'],
                            sentiment, confidence))
            
            if len(results) % 100 == 0:
                temp_df = pd.DataFrame(results, columns=['title', 'description', 'date', 'Sentiment', 'Confidence'])
                temp_df.to_csv(partial_results_path, mode='a', index=False, header=False)  # Append แบบไม่มี Header
                results = []
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if results:
            temp_df = pd.DataFrame(results, columns=['title', 'description', 'date', 'Sentiment', 'Confidence'])
            temp_df.to_csv(partial_results_path, mode='a', index=False, header=False)
        print(f"Saved partial results to {partial_results_path}.")

    # บันทึกผลลัพธ์สุดท้ายพร้อม Header
    final_results = pd.read_csv(partial_results_path)
    final_results.to_csv("news_with_sentiment_gpu.csv", index=False, header=True)
    print("✅ Final results saved to news_with_sentiment_gpu.csv")

# Main entry point
if __name__ == '__main__':
    freeze_support()
    main()
