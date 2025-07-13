import time
import pandas as pd
import mysql.connector
import torch
import os
import numpy as np
from tqdm import tqdm
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import Dataset
from urllib.parse import urlparse

# ตั้งค่าพาธสำหรับบันทึกผลลัพธ์
PARTIAL_RESULTS_PATH = 'partial_results.csv'
FINAL_RESULTS_PATH = 'final_sentiment_results.csv'
DAILY_SUMMARY_PATH = 'daily_sentiment_summary.csv'

# ตั้งค่าฐานข้อมูล
DB_CONFIG = {
    "host": "localhost",
    "user": "root", 
    "password": "1234",
    "database": "TradeMine",
    "autocommit": True,
    "connect_timeout": 60,
    "read_timeout": 300,
    "write_timeout": 300,
    "pool_size": 5,
    "pool_reset_session": False
}

def get_db_connection():
    """สร้างการเชื่อมต่อฐานข้อมูลใหม่"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        print("✅ เชื่อมต่อฐานข้อมูลสำเร็จ!")
        return conn
    except mysql.connector.Error as e:
        print(f"❌ การเชื่อมต่อฐานข้อมูลล้มเหลว: {e}")
        return None

# ✅ รายชื่อหุ้นและชื่อบริษัท
stock_entities = {
    "ADVANC": ["ADVANC", "AIS"],
    "DIF": ["DIF"],
    "DITTO": ["DITTO"],
    "HUMAN": ["HUMAN"],
    "INET": ["INET"],
    "INSET": ["INSET"],
    "INTUCH": ["INTUCH"],
    "JAS": ["JAS"],
    "JMART": ["JMART"],
    "TRUE": ["TRUE"],
    "AAPL": ["Apple"],
    "AMD": ["AMD", "Advanced Micro Devices"],
    "AMZN": ["Amazon"],
    "AVGO": ["Broadcom"],
    "GOOGL": ["Google", "Alphabet"],
    "META": ["Meta", "Facebook"],
    "MSFT": ["Microsoft"],
    "NVDA": ["Nvidia"],
    "TSLA": ["Tesla"],
    "TSM": ["TSMC", "Taiwan Semiconductor"]
}

# ✅ Context Keyword Mapping
context_mapping = [
    {"Keywords": ["mobile", "5G", "network", "internet", "broadband", "AIS", "cellular", "telecom"], "Stocks": ["ADVANC"]},
    {"Keywords": ["infrastructure fund", "telecom assets", "tower lease", "fiber optic"], "Stocks": ["DIF"]},
    {"Keywords": ["digital", "document", "scanner", "workflow", "e-document", "digital solution"], "Stocks": ["DITTO"]},
    {"Keywords": ["human resource", "HR software", "payroll", "recruitment", "employee management"], "Stocks": ["HUMAN"]},
    {"Keywords": ["cloud", "data center", "IT service", "hosting", "cloud solution", "server"], "Stocks": ["INET"]},
    {"Keywords": ["network service", "fiber optic", "infrastructure", "installation service"], "Stocks": ["INSET"]},
    {"Keywords": ["investment", "holding company", "telecom", "AIS", "INTUCH group"], "Stocks": ["INTUCH"]},
    {"Keywords": ["broadband", "fixed internet", "telecom", "Jasmine", "fiber"], "Stocks": ["JAS"]},
    {"Keywords": ["retail", "mobile store", "finance service", "Jaymart", "consumer loan", "mobile retail"], "Stocks": ["JMART"]},
    {"Keywords": ["mobile", "5G", "network", "broadband", "TRUE ID", "telecom", "True Corporation"], "Stocks": ["TRUE"]},
    {"Keywords": ["semiconductor", "chip", "foundry", "wafer", "GPU", "CPU", "manufacturing"], "Stocks": ["TSM", "NVDA", "AMD", "AVGO"]},
    {"Keywords": ["cloud", "AWS", "Azure", "Google Cloud", "data center"], "Stocks": ["AMZN", "MSFT", "GOOGL"]},
    {"Keywords": ["EV", "electric vehicle", "self-driving", "battery", "gigafactory"], "Stocks": ["TSLA", "AAPL", "NVDA"]},
    {"Keywords": ["AI", "machine learning", "chatbot", "language model"], "Stocks": ["NVDA", "MSFT", "META", "GOOGL"]}
]

# ตรวจสอบ GPU
if torch.cuda.is_available():
    print("✅ CUDA Available:", torch.cuda.is_available())
    print("✅ CUDA Version:", torch.version.cuda)
    print("✅ Device Name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    print("⚠️ No GPU detected, using CPU instead.")
    device = torch.device("cpu")

print(f"Using device: {device}")

def load_model():
    """โหลด FinBERT model สำหรับ sentiment analysis"""
    print("Loading FinBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "yiyanghkust/finbert-tone"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "yiyanghkust/finbert-tone"
    )
    print("✅ Model and tokenizer loaded successfully!")
    return model, tokenizer

def extract_source(link):
    """แยกชื่อแหล่งข่าวจาก URL"""
    try:
        domain = urlparse(link).netloc
        if domain.startswith("www."):
            domain = domain[4:]
        if domain.endswith(".com"):
            domain = domain[:-4]
        return domain
    except:
        return ""

def process_news(docs):
    """ประมวลผลข่าวเพื่อหาหุ้นที่เกี่ยวข้อง"""
    results = []
    for doc in docs:
        matched = set()
        # NER
        for ent in doc.ents:
            if ent.label_ == "ORG" and ent.ent_id_:
                matched.add(ent.ent_id_)
        # Context
        text = doc.text.lower()
        for ctx in context_mapping:
            for keyword in ctx["Keywords"]:
                if keyword.lower() in text:
                    matched.update(ctx["Stocks"])
                    break
        results.append(", ".join(sorted(matched)) if matched else None)
    return results

def prepare_data():
    """เตรียมข้อมูลจากไฟล์ CSV ที่มีการ match หุ้นแล้ว"""
    try:
        df = pd.read_csv('News_Matched.csv')
        # เพิ่มคอลัมน์ Source และ Type
        df['Source'] = df['link'].apply(extract_source)
        df['Type'] = 'Database News'
        
        # กรองเฉพาะข่าวที่มีการ match หุ้น
        matched_df = df[df['MatchedStock'].notna() & (df['MatchedStock'] != '')]
        
        print(f"✅ โหลดข้อมูล {len(matched_df)} ข่าวที่เกี่ยวข้องกับหุ้น")
        return matched_df
    except FileNotFoundError:
        print("❌ ไม่พบไฟล์ News_Matched.csv กรุณารันส่วนการ match หุ้นก่อน")
        return pd.DataFrame()

def convert_nan_to_none(data_list):
    """แปลง NaN values เป็น None สำหรับ MySQL"""
    return [[None if (isinstance(x, float) and np.isnan(x)) else x for x in row] for row in data_list]

def update_to_database(daily_sentiment):
    """อัปเดตข้อมูลในฐานข้อมูล StockDetail - เฉพาะข้อมูลที่มีอยู่แล้ว"""
    conn = None
    cursor = None
    try:
        # สร้างการเชื่อมต่อใหม่
        conn = get_db_connection()
        if not conn:
            raise Exception("Cannot connect to database")
        
        cursor = conn.cursor()
        
        # SQL query สำหรับอัปเดต
        update_query = """
        UPDATE StockDetail
        SET
            positive_news = %s,
            negative_news = %s,
            neutral_news = %s,
            Sentiment = %s
        WHERE StockSymbol = %s AND date = %s
        """
        
        # เตรียมข้อมูลสำหรับอัปเดต
        data_to_update = []
        for _, row in daily_sentiment.iterrows():
            data_to_update.append((
                int(row['positive_news']),
                int(row['negative_news']),
                int(row['neutral_news']),
                float(row['Sentiment']),
                row['Stock'],
                row['date']
            ))
        
        # อัปเดตเป็น batch เล็กๆ เพื่อป้องกัน timeout
        batch_size = 100  # ลดขนาด batch
        total_rows = len(data_to_update)
        updated_count = 0
        
        for i in tqdm(range(0, total_rows, batch_size), desc="กำลังอัปเดตข้อมูล", unit="batch"):
            batch = data_to_update[i:i+batch_size]
            
            try:
                # ตรวจสอบการเชื่อมต่อก่อนทำ batch
                if not conn.is_connected():
                    print("⚠️ การเชื่อมต่อขาด กำลังเชื่อมต่อใหม่...")
                    conn = get_db_connection()
                    cursor = conn.cursor()
                
                cursor.executemany(update_query, batch)
                conn.commit()  # commit หลังจาก batch เสร็จ
                updated_count += len(batch)
                print(f"✅ อัปเดตข้อมูล batch {i//batch_size + 1}: {len(batch)} รายการ (รวม {updated_count}/{total_rows})")
                
            except mysql.connector.Error as e:
                print(f"❌ เกิดข้อผิดพลาดใน batch {i//batch_size + 1}: {e}")
                # พยายามเชื่อมต่อใหม่
                try:
                    if conn:
                        conn.close()
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    print("✅ เชื่อมต่อฐานข้อมูลใหม่สำเร็จ")
                except:
                    print("❌ ไม่สามารถเชื่อมต่อฐานข้อมูลใหม่ได้")
                    raise
        
        print(f"✅ อัปเดตข้อมูลทั้งหมด {updated_count} รายการ ในฐานข้อมูลเรียบร้อย")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการอัปเดตฐานข้อมูล: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
            print("✅ ปิดการเชื่อมต่อฐานข้อมูลเรียบร้อย")

def create_daily_summary():
    """สร้าง Daily Sentiment Summary จากไฟล์ผลลัพธ์"""
    print("📈 เริ่มสร้าง Daily Sentiment Summary...")
    
    if not os.path.exists(FINAL_RESULTS_PATH):
        print(f"❌ ไม่พบไฟล์ {FINAL_RESULTS_PATH}")
        return None
    
    df = pd.read_csv(FINAL_RESULTS_PATH)
    
    # ===== STEP 2: Clean & Prepare =====
    df['MatchedStock'] = df['MatchedStock'].fillna('').str.strip()
    df['Sentiment'] = df['Sentiment'].fillna('Neutral')
    df['Confidence'] = df['Confidence'].fillna(0.0)
    df['date'] = pd.to_datetime(df['date']).dt.date

    # แยก MatchedStock และตัดช่องว่าง
    df_exploded = df.assign(Stock=df['MatchedStock'].str.split(', ')).explode('Stock')
    df_exploded = df_exploded[df_exploded['Stock'] != '']
    df_exploded['Stock'] = df_exploded['Stock'].str.strip()

    print(f"Total rows after explode: {len(df_exploded)}")

    # ลบข้อมูลซ้ำ
    df_exploded = df_exploded.drop_duplicates(subset=['date', 'Stock', 'Sentiment', 'Confidence'])
    print(f"Total rows after deduplication: {len(df_exploded)}")

    # บันทึกข้อมูลกลางเพื่อตรวจสอบ
    df_exploded.to_csv('exploded_data.csv', index=False)

    # ===== STEP 3: Group & Summarize =====
    daily_sentiment = df_exploded.groupby(['date', 'Stock']).agg(
        positive_news=('Sentiment', lambda x: (x == 'Positive').sum()),
        negative_news=('Sentiment', lambda x: (x == 'Negative').sum()),
        neutral_news=('Sentiment', lambda x: (x == 'Neutral').sum()),
        avg_confidence=('Confidence', 'mean'),
        total_news=('Sentiment', 'count')
    ).reset_index()

    print(f"Total rows after groupby: {len(daily_sentiment)}")

    # ===== STEP 4: Feature Engineering =====
    daily_sentiment['positive_ratio'] = daily_sentiment['positive_news'] / daily_sentiment['total_news']
    daily_sentiment['negative_ratio'] = daily_sentiment['negative_news'] / daily_sentiment['total_news']

    def calculate_net_sentiment(row):
        threshold = 0.4
        if row['positive_ratio'] > row['negative_ratio'] and row['positive_ratio'] > threshold:
            return 1
        elif row['negative_ratio'] > row['positive_ratio'] and row['negative_ratio'] > threshold:
            return -1
        else:
            return 0

    daily_sentiment['net_sentiment_score'] = daily_sentiment.apply(calculate_net_sentiment, axis=1)
    daily_sentiment['has_news'] = daily_sentiment['total_news'].apply(lambda x: 1 if x > 0 else 0)
    daily_sentiment = daily_sentiment.rename(columns={'net_sentiment_score': 'Sentiment'})
    
    # บันทึกไฟล์
    daily_sentiment.to_csv(DAILY_SUMMARY_PATH, index=False)
    print(f"✅ บันทึก Daily Summary ที่ {DAILY_SUMMARY_PATH}")
    
    return daily_sentiment

def main():
    """ฟังก์ชันหลักสำหรับประมวลผลข่าว"""
    
    # ตรวจสอบว่ามีไฟล์ daily_sentiment_summary.csv อยู่หรือไม่
    if os.path.exists(DAILY_SUMMARY_PATH):
        print(f"✅ พบไฟล์ {DAILY_SUMMARY_PATH} อยู่แล้ว")
        choice = input("ต้องการข้าม sentiment analysis และอัปเดตฐานข้อมูลเลยไหม? (y/n): ")
        if choice.lower() == 'y':
            try:
                daily_sentiment = pd.read_csv(DAILY_SUMMARY_PATH)
                print(f"✅ โหลดข้อมูลจาก {DAILY_SUMMARY_PATH} ({len(daily_sentiment)} รายการ)")
                print("💾 เริ่มอัปเดตฐานข้อมูล...")
                update_to_database(daily_sentiment)
                print("✅ การอัปเดตฐานข้อมูลเสร็จสิ้น")
                return
            except Exception as e:
                print(f"❌ เกิดข้อผิดพลาดในการโหลดไฟล์: {e}")
                print("จะทำการประมวลผลใหม่...")
        else:
            print("จะทำการประมวลผลใหม่...")
    
    # ส่วนที่ 1: ดึงข้อมูลจากฐานข้อมูลและ match หุ้น
    conn = get_db_connection()
    if not conn:
        return
    
    cursor = conn.cursor()
    query = """
    SELECT Title AS title, URL AS link, Content AS description, PublishedDate AS date, Img AS image
    FROM News 
    """
    
    try:
        cursor.execute(query)
        news_data = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        news_df = pd.DataFrame(news_data, columns=columns)

        if news_df.empty:
            print("⚠️ ไม่พบข้อมูลข่าวในฐานข้อมูล")
            return
        
        print(f"✅ ดึงข้อมูลข่าว {len(news_df)} รายการจากฐานข้อมูล")
        
        # บันทึกข้อมูลดิบ
        news_df.to_csv('News.csv', index=False, encoding='utf-8')
        
        # โหลด spaCy model
        try:
            nlp = spacy.load("en_core_web_trf")
            print("✅ โหลด spaCy Model สำเร็จ")
        except OSError:
            print("❌ ไม่พบ spaCy model 'en_core_web_trf'")
            print("กรุณาติดตั้งด้วย: python -m spacy download en_core_web_trf")
            return
        
        # เพิ่ม Entity Ruler
        if "entity_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", before="ner")
            patterns = []
            for stock, keywords in stock_entities.items():
                for keyword in keywords:
                    patterns.append({
                        "label": "ORG",
                        "pattern": keyword,
                        "id": stock
                    })
            ruler.add_patterns(patterns)
        
        # เตรียมข้อความสำหรับการวิเคราะห์
        texts = news_df.apply(lambda row: f"{row.get('title', '')} {row.get('description', '')}", axis=1)
        
        # ประมวลผล NER + Context
        start_time = time.time()
        batch_size = 32
        
        print(f"🔍 เริ่มประมวลผล NER + Context (Batch size = {batch_size})")
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="🔍 Processing batches"):
            batch_texts = texts[i:i+batch_size].tolist()
            docs = list(nlp.pipe(batch_texts))
            batch_results = process_news(docs)
            results.extend(batch_results)
        
        news_df["MatchedStock"] = results
        
        # สรุปผล
        total_news = len(news_df)
        related_df = news_df[news_df["MatchedStock"].notnull()]
        related_news = len(related_df)
        unrelated_news = total_news - related_news
        percentage = (related_news / total_news) * 100
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print("\n📊 Stock Matching Summary")
        print(f"✅ ข่าวทั้งหมด: {total_news}")
        print(f"✅ ข่าวที่เกี่ยวข้อง: {related_news} ({percentage:.2f}%)")
        print(f"✅ ข่าวที่ไม่เกี่ยวข้อง: {unrelated_news} ({100 - percentage:.2f}%)")
        print(f"⏱️ ใช้เวลาในการประมวลผล: {elapsed / 60:.2f} นาที\n")
        
        # บันทึกผลลัพธ์การ match
        news_df.to_csv('News_Matched.csv', index=False, encoding='utf-8')
        
        # ตรวจสอบว่ามีข่าวที่ match หุ้นหรือไม่
        if related_news == 0:
            print("⚠️ ไม่มีข่าวที่เกี่ยวข้องกับหุ้น ข้ามการวิเคราะห์ sentiment")
            return
        
        # ส่วนที่ 2: Sentiment Analysis
        print("🎯 เริ่มการวิเคราะห์ sentiment...")
        
        # โหลด FinBERT model
        model, tokenizer = load_model()
        
        finbert_sentiment = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512
        )
        
        # เตรียมข้อมูลสำหรับ sentiment analysis
        combined = prepare_data()
        
        if combined.empty:
            print("❌ ไม่มีข้อมูลสำหรับการวิเคราะห์ sentiment")
            return
        
        # เตรียมข้อความสำหรับการวิเคราะห์
        financial_news = combined.apply(
            lambda row: f"{row.get('title', '')} {row.get('description', '')}", axis=1
        ).tolist()
        
        # ลบไฟล์ผลลัพธ์เก่า
        if os.path.exists(PARTIAL_RESULTS_PATH):
            os.remove(PARTIAL_RESULTS_PATH)
        
        # ประมวลผล sentiment analysis
        batch_size = 16
        results = []
        total_records = len(financial_news)
        
        print(f"🔍 เริ่มการวิเคราะห์ sentiment (Batch size = {batch_size})")
        
        with tqdm(total=total_records, desc="Processing Sentiment Analysis") as pbar:
            try:
                for i in range(0, total_records, batch_size):
                    batch_texts = financial_news[i:i+batch_size]
                    chunk_results = finbert_sentiment(batch_texts)
                    
                    # รวบรวมผลลัพธ์จาก batch
                    for idx, result in enumerate(chunk_results):
                        global_idx = i + idx
                        sentiment = result['label']
                        confidence = result['score']
                        
                        results.append({
                            'title': combined.iloc[global_idx]['title'],
                            'description': combined.iloc[global_idx]['description'],
                            'date': combined.iloc[global_idx]['date'],
                            'link': combined.iloc[global_idx]['link'],
                            'Source': combined.iloc[global_idx]['Source'],
                            'MatchedStock': combined.iloc[global_idx]['MatchedStock'],
                            'Type': combined.iloc[global_idx]['Type'],
                            'Sentiment': sentiment,
                            'Confidence': confidence,
                            'image': combined.iloc[global_idx]['image']
                        })
                    
                    pbar.update(len(batch_texts))
                    
                    # บันทึกทุก 100 records
                    if len(results) >= 100:
                        temp_df = pd.DataFrame(results)
                        temp_df.to_csv(PARTIAL_RESULTS_PATH, mode='a', index=False, 
                                     header=not os.path.exists(PARTIAL_RESULTS_PATH))
                        results = []
                        
            except Exception as e:
                print(f"❌ เกิดข้อผิดพลาด: {e}")
            finally:
                # บันทึกผลลัพธ์ที่เหลือ
                if results:
                    temp_df = pd.DataFrame(results)
                    temp_df.to_csv(PARTIAL_RESULTS_PATH, mode='a', index=False, 
                                 header=not os.path.exists(PARTIAL_RESULTS_PATH))
        
        # รวบรวมผลลัพธ์สุดท้าย
        if os.path.exists(PARTIAL_RESULTS_PATH):
            final_results = pd.read_csv(PARTIAL_RESULTS_PATH)
            final_results.to_csv(FINAL_RESULTS_PATH, index=False)
            
            # สรุปผล sentiment
            sentiment_summary = final_results['Sentiment'].value_counts()
            print(f"\n📊 Sentiment Analysis Summary")
            print(f"✅ ข่าวทั้งหมดที่วิเคราะห์: {len(final_results)}")
            for sentiment, count in sentiment_summary.items():
                percentage = (count / len(final_results)) * 100
                print(f"✅ {sentiment}: {count} ({percentage:.2f}%)")
            
            print(f"✅ บันทึกผลลัพธ์สุดท้ายที่ {FINAL_RESULTS_PATH}")
            
            # ลบไฟล์ชั่วคราว
            if os.path.exists(PARTIAL_RESULTS_PATH):
                os.remove(PARTIAL_RESULTS_PATH)
                
            # ส่วนที่ 3: สร้าง Daily Summary และบันทึกเข้าฐานข้อมูล
            daily_sentiment = create_daily_summary()
            
            if daily_sentiment is not None:
                print("💾 เริ่มอัปเดตฐานข้อมูล...")
                update_to_database(daily_sentiment)
                
        else:
            print("⚠️ ไม่มีผลลัพธ์ให้บันทึก")
        
        print("✅ การประมวลผลเสร็จสิ้นทั้งหมด")
        
    except mysql.connector.Error as e:
        print(f"❌ เกิดข้อผิดพลาดในการรันคำสั่ง SQL: {e}")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดทั่วไป: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
            print("✅ ปิดการเชื่อมต่อฐานข้อมูลเรียบร้อย")

if __name__ == "__main__":
    main()