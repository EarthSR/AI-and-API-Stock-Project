import spacy
from spacy.tokens import Span
import pandas as pd
from tqdm import tqdm
import time

# ✅ ตรวจสอบและเปิดใช้ GPU
import spacy
if not spacy.prefer_gpu():
    print("⚠️ ไม่พบ GPU หรือ spaCy ไม่ได้ตั้งค่า GPU")
else:
    print("🚀 ใช้ GPU:", spacy.prefer_gpu())

# ✅ รายชื่อหุ้นและชื่อบริษัท
stock_entities = {
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

# ✅ โหลด spaCy Transformer Model
nlp = spacy.load("en_core_web_trf")
print("✅ โหลด Model สำเร็จ")

# ✅ เพิ่ม Entity Ruler
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

# ✅ อ่านข่าว
file_path = "Combined_News.csv"
df = pd.read_csv(file_path)

# ✅ ฟังก์ชัน NER
def extract_entities(row):
    text = f"{row.get('title', '')} {row.get('description', '')}"
    doc = nlp(text)
    matched = set()
    for ent in doc.ents:
        if ent.label_ == "ORG" and ent.ent_id_:
            matched.add(ent.ent_id_)
    return ", ".join(sorted(matched)) if matched else None

# ✅ ประมวลผลและจับเวลา
start_time = time.time()

tqdm.pandas(desc="🔍 NER Processing")
df["NER_MatchedStock"] = df.progress_apply(extract_entities, axis=1)

end_time = time.time()
elapsed = end_time - start_time

# ✅ สรุปผล
total_news = len(df)
related_df = df[df["NER_MatchedStock"].notnull()]
unrelated_df = df[df["NER_MatchedStock"].isnull()]
related_news = len(related_df)
unrelated_news = len(unrelated_df)
percentage = (related_news / total_news) * 100

print("\n📊 NER Model Summary")
print(f"✅ ข่าวทั้งหมด: {total_news}")
print(f"✅ ข่าวที่เกี่ยวข้อง: {related_news} ({percentage:.2f}%)")
print(f"✅ ข่าวที่ไม่เกี่ยวข้อง: {unrelated_news} ({100 - percentage:.2f}%)")
print(f"⏱️ ใช้เวลาในการประมวลผล: {elapsed / 60:.2f} นาที\n")

# ✅ บันทึกไฟล์
related_df.to_csv("Related_News_NER.csv", index=False, encoding='utf-8')
unrelated_df.to_csv("Unrelated_News_NER.csv", index=False, encoding='utf-8')

print(f"💾 บันทึกข่าวที่เกี่ยวข้องที่: Related_News_NER.csv")
print(f"💾 บันทึกข่าวที่ไม่เกี่ยวข้องที่: Unrelated_News_NER.csv")
