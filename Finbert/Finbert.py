from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from tqdm import tqdm
import os
import torch
from multiprocessing import freeze_support
import sys

# ✅ บังคับให้ stdout ใช้ UTF-8 และข้ามอีโมจิโดยอัตโนมัติ
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# ✅ ตรวจสอบระดับของโฟลเดอร์ (ปรับ `..` ตามตำแหน่งของไฟล์)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 

# ✅ ตรวจสอบ GPU
if torch.cuda.is_available():
    print("✅ CUDA Available:", torch.cuda.is_available())
    print("✅ CUDA Version:", torch.version.cuda)
    print("✅ Device Name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    print("⚠️ No GPU detected, using CPU instead.")
    device = torch.device("cpu")

print(f"Using device: {device}")

PARTIAL_RESULTS_PATH = os.path.join(BASE_DIR, "Finbert", "partial_results_gpu.csv")
FINAL_RESULTS_PATH = os.path.join(BASE_DIR, "Finbert", "news_with_sentiment_gpu.csv")

def load_model():
    """ โหลดโมเดล FinBERT """
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
    # ✅ โหลดโมเดล
    model, tokenizer = load_model()

    # ✅ สร้าง Pipeline
    finbert_sentiment = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=512
    )

    # ✅ โหลดข้อมูลข่าว
    input_file_path = os.path.join(BASE_DIR, "news_data", "Final_News.csv")

    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"❌ ไม่พบไฟล์: {input_file_path}")

    input_data = pd.read_csv(input_file_path)

    # ✅ ตรวจสอบ Column
    required_columns = ['title', 'description', 'date', 'link', 'Source']
    for col in required_columns:
        if col not in input_data.columns:
            raise ValueError(f"❌ Column '{col}' is missing from the dataset!")

    # ✅ แทนค่า NaN ด้วย "" เพื่อป้องกันข่าวถูกลบ
    input_data.fillna("", inplace=True)

    # ✅ รวมข้อความ `title` + `description`
    financial_news = input_data.apply(
        lambda row: f"{row['title']} {row['description']}" if row['description'] else row['title'], axis=1
    ).tolist()

    # ✅ บันทึกทับไฟล์ทุกครั้ง (mode='a')
    processed_data = pd.DataFrame(columns=['title', 'description', 'date', 'link', 'Source', 'Sentiment', 'Confidence'])
    processed_data.to_csv(PARTIAL_RESULTS_PATH, index=False, header=True, mode='w')  # ✅ สร้างไฟล์ใหม่ ทุกครั้ง (overwrite)

    results = []
    total_records = len(financial_news)

    with tqdm(total=total_records, desc="Processing News") as pbar:
        try:
            for idx, news in enumerate(financial_news):
                chunks = [news]  # ใช้ข้อความตรงๆ หากไม่แบ่ง chunk
                chunk_results = finbert_sentiment(chunks)
                sentiment = chunk_results[0]['label']
                confidence = chunk_results[0]['score']

                results.append((input_data.iloc[idx]['title'],
                                input_data.iloc[idx]['description'],
                                input_data.iloc[idx]['date'],
                                input_data.iloc[idx]['link'],
                                input_data.iloc[idx]['Source'],
                                sentiment,
                                confidence))

                pbar.update(1)  # ✅ อัปเดต Progress Bar ให้ต่อจากที่ค้างไว้

                # ✅ บันทึกแบบ Append ทีละ 100 ข่าว
                if len(results) % 100 == 0:
                    temp_df = pd.DataFrame(results, columns=['title', 'description', 'date', 'link', 'Source', 'Sentiment', 'Confidence'])
                    temp_df.to_csv(PARTIAL_RESULTS_PATH, mode='a', index=False, header=not os.path.exists(PARTIAL_RESULTS_PATH))
                    results = []

        except Exception as e:
            print(f"❌ Error occurred: {e}")
        finally:
            if results:
                temp_df = pd.DataFrame(results, columns=['title', 'description', 'date', 'link', 'Source', 'Sentiment', 'Confidence'])
                temp_df.to_csv(PARTIAL_RESULTS_PATH, mode='a', index=False, header=False)  # ✅ Append ข้อมูลที่เหลือ
            print(f"✅ Saved partial results to {PARTIAL_RESULTS_PATH}.")

    # ✅ บันทึกผลลัพธ์สุดท้ายพร้อม Header
    if os.path.exists(PARTIAL_RESULTS_PATH):
        final_results = pd.read_csv(PARTIAL_RESULTS_PATH)
        final_results.to_csv(FINAL_RESULTS_PATH, index=False, header=True)
        print(f"✅ Final results saved to {FINAL_RESULTS_PATH}")
    else:
        print(f"⚠️ ไม่มีผลลัพธ์ให้บันทึกที่ {PARTIAL_RESULTS_PATH}")

        print(f"✅ Final results saved to {FINAL_RESULTS_PATH}")

# ✅ Main entry point
if __name__ == '__main__':
    freeze_support()
    main()