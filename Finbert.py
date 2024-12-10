from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from tqdm import tqdm
import os
import torch

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
# ตรวจสอบ GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# โหลด Tokenizer และ Model
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").to(device)

# สร้าง Pipeline
finbert_sentiment = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    max_length=512
)

# ฟังก์ชันแบ่งข้อความ
def split_into_chunks(text, max_length=512):
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

# ฟังก์ชันรวมผลลัพธ์
def aggregate_results(chunk_results):
    sentiments = [result['label'] for result in chunk_results]
    confidences = [result['score'] for result in chunk_results]
    max_confidence_idx = confidences.index(max(confidences))
    return sentiments[max_confidence_idx], confidences[max_confidence_idx]

# โหลดข้อมูล
input_data = pd.read_csv("./news_cleaned.csv")
input_data.dropna(subset=['title', 'description'], how='all', inplace=True)

# รวมข้อความ
financial_news = input_data.apply(
    lambda row: f"{row['title']} {row['description']}" if pd.notnull(row['description']) else row['title'], axis=1
).tolist()

# โหลดสถานะ
partial_results_path = "partial_results_gpu.csv"
if not os.path.exists(partial_results_path):
    print("No partial results found. Starting fresh.")
    processed_data = pd.DataFrame(columns=['title', 'description', 'Sentiment', 'Confidence'])
    start_idx = 0
else:
    processed_data = pd.read_csv(partial_results_path)
    start_idx = len(processed_data)
    print(f"Resuming from {start_idx} processed records.")

# วิเคราะห์ข้อความ
results = []
try:
    for idx, news in enumerate(tqdm(financial_news[start_idx:], desc="Processing News")):
        chunks = split_into_chunks(news)
        chunk_results = finbert_sentiment(chunks)
        sentiment, confidence = aggregate_results(chunk_results)
        results.append((input_data.iloc[start_idx + idx]['title'],
                        input_data.iloc[start_idx + idx]['description'],
                        sentiment, confidence))
        if len(results) % 100 == 0:
            temp_df = pd.DataFrame(results, columns=['title', 'description', 'Sentiment', 'Confidence'])
            if not os.path.exists(partial_results_path):
                temp_df.to_csv(partial_results_path, index=False, header=True)
            else:
                temp_df.to_csv(partial_results_path, mode='a', index=False, header=False)
            results = []
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    if results:
        temp_df = pd.DataFrame(results, columns=['title', 'description', 'Sentiment', 'Confidence'])
        temp_df.to_csv(partial_results_path, mode='a', index=False, header=True)
    print(f"Saved partial results to {partial_results_path}.")

# รวมผลลัพธ์สุดท้าย
final_results = pd.read_csv(partial_results_path)
final_results.to_csv("news_with_sentiment_gpu.csv", index=False)
print("Final results saved to news_with_sentiment_gpu.csv")
