from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from tqdm import tqdm
import os
import torch
from multiprocessing import freeze_support
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if torch.cuda.is_available():
    print("✅ CUDA Available:", torch.cuda.is_available())
    print("✅ CUDA Version:", torch.version.cuda)
    print("✅ Device Name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    print("⚠️ No GPU detected, using CPU instead.")
    device = torch.device("cpu")

print(f"Using device: {device}")

PARTIAL_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'thai', 'News', 'Thai_News_Hybrid.csv')
FINAL_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'thai', 'News', 'Thai_News_Sentiment.csv')

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

from urllib.parse import urlparse

def extract_source(link):
    try:
        domain = urlparse(link).netloc
        if domain.startswith("www."):
            domain = domain[4:]
        if domain.endswith(".com"):
            domain = domain[:-4]
        return domain
    except:
        return ""


def prepare_data():
    related_path = os.path.join(CURRENT_DIR, '..', 'thai', 'News', "Related_News_Hybrid.csv")
    unrelated_path = os.path.join(CURRENT_DIR, '..', 'thai', 'News', "Unrelated_News_Hybrid.csv")

    if not os.path.exists(related_path) or not os.path.exists(unrelated_path):
        raise FileNotFoundError("❌ ไม่พบไฟล์ Related หรือ Unrelated News")

    related = pd.read_csv(related_path)
    unrelated = pd.read_csv(unrelated_path)
    print("🔍 กำลังเพิ่ม Source ให้กับข่าวทั้ง Related และ Unrelated")
    related['Source'] = related['link'].apply(extract_source)
    unrelated['Source'] = unrelated['link'].apply(extract_source)
        
    related['Type'] = 'Related'
    unrelated['Type'] = 'Unrelated'

    combined = pd.concat([related, unrelated], ignore_index=True)
    if 'date' in combined.columns:
        combined.dropna(subset=['date'], inplace=True)
    combined.fillna("", inplace=True)
    combined_path = os.path.join(CURRENT_DIR, '..', 'thai', 'News', "Combined_News_Hybrid_Thai.csv")
    combined.to_csv(combined_path, index=False)
    print("✅ รวมข่าวเรียบร้อยแล้ว → Combined_News_Hybrid_Thai.csv")
    return combined

def main():
    model, tokenizer = load_model()

    finbert_sentiment = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=16,
        truncation=True,
        max_length=512
    )

    combined = prepare_data()

    financial_news = combined.apply(
        lambda row: f"{row.get('title', '')} {row.get('description', '')}", axis=1
    ).tolist()

    processed_data = pd.DataFrame(columns=['title', 'description', 'date', 'link', 'Source', 'MatchedStock', 'Type', 'Sentiment', 'Confidence'])
    processed_data.to_csv(PARTIAL_RESULTS_PATH, index=False, header=True, mode='w')

    results = []
    total_records = len(financial_news)

    with tqdm(total=total_records, desc="Processing Combined News") as pbar:
        try:
            for idx, news in enumerate(financial_news):
                chunk_results = finbert_sentiment([news])
                sentiment = chunk_results[0]['label']
                confidence = chunk_results[0]['score']

                results.append((combined.iloc[idx]['title'],
                                combined.iloc[idx]['description'],
                                combined.iloc[idx]['date'],
                                combined.iloc[idx]['link'],
                                combined.iloc[idx]['Source'],
                                combined.iloc[idx]['MatchedStock'],
                                combined.iloc[idx]['Type'],
                                sentiment,
                                confidence))

                pbar.update(1)

                if len(results) % 100 == 0:
                    temp_df = pd.DataFrame(results, columns=['title', 'description', 'date', 'link', 'Source', 'MatchedStock', 'Type', 'Sentiment', 'Confidence'])
                    temp_df.to_csv(PARTIAL_RESULTS_PATH, mode='a', index=False, header=not os.path.exists(PARTIAL_RESULTS_PATH))
                    results = []

        except Exception as e:
            print(f"❌ Error occurred: {e}")
        finally:
            if results:
                temp_df = pd.DataFrame(results, columns=['title', 'description', 'date', 'link', 'Source', 'MatchedStock', 'Type', 'Sentiment', 'Confidence'])
                temp_df.to_csv(PARTIAL_RESULTS_PATH, mode='a', index=False, header=False)
            print(f"✅ Saved partial results to {PARTIAL_RESULTS_PATH}.")

    if os.path.exists(PARTIAL_RESULTS_PATH):
        final_results = pd.read_csv(PARTIAL_RESULTS_PATH)
        final_results.to_csv(FINAL_RESULTS_PATH, index=False, header=True)
        print(f"✅ Final results saved to {FINAL_RESULTS_PATH}")
    else:
        print(f"⚠️ ไม่มีผลลัพธ์ให้บันทึกที่ {PARTIAL_RESULTS_PATH}")

if __name__ == '__main__':
    freeze_support()
    main()
