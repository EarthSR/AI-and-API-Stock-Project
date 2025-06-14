import spacy
from spacy.tokens import Span
from tqdm import tqdm
import pandas as pd
import time
import os
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

if not spacy.prefer_gpu():
    print("⚠️ ไม่พบ GPU หรือ spaCy ไม่ได้ตั้งค่า GPU")
else:
    print("🚀 ใช้ GPU:", spacy.prefer_gpu())


# ✅ รายชื่อหุ้นและชื่อบริษัท (เปลี่ยนเป็นหุ้นไทย)
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
    "TRUE": ["TRUE"]
}

# ✅ Context Keyword Mapping (ยังไม่มีข้อมูล keyword เฉพาะ สามารถเติมเพิ่มได้)
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
    {"Keywords": ["mobile", "5G", "network", "broadband", "TRUE ID", "telecom", "True Corporation"], "Stocks": ["TRUE"]}
]


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
floder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'thai', 'News')
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'thai', 'News', 'Thai_News.csv')
df = pd.read_csv(file_path)

# ✅ เตรียมข้อความ
texts = df.apply(lambda row: f"{row.get('title', '')} {row.get('description', '')}", axis=1)

# ✅ ฟังก์ชันหาข่าวที่เกี่ยวข้อง
def process_news(docs):
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

# ✅ ประมวลผล
start_time = time.time()
batch_size = 10

print(f"🔍 เริ่มประมวลผล NER + Context (Batch size = {batch_size})")
results = []
for doc_batch in tqdm(nlp.pipe(texts, batch_size=batch_size), total=len(texts), desc="🔍 Hybrid Processing"):
    results.extend(process_news([doc_batch]))

df["MatchedStock"] = results

# ✅ สรุปผล
total_news = len(df)
related_df = df[df["MatchedStock"].notnull()]
unrelated_df = df[df["MatchedStock"].isnull()]
related_news = len(related_df)
unrelated_news = len(unrelated_df)
percentage = (related_news / total_news) * 100

end_time = time.time()
elapsed = end_time - start_time

print("\n📊 Hybrid Model Summary")
print(f"✅ ข่าวทั้งหมด: {total_news}")
print(f"✅ ข่าวที่เกี่ยวข้อง: {related_news} ({percentage:.2f}%)")
print(f"✅ ข่าวที่ไม่เกี่ยวข้อง: {unrelated_news} ({100 - percentage:.2f}%)")
print(f"⏱️ ใช้เวลาในการประมวลผล: {elapsed / 60:.2f} นาที\n")

# ✅ บันทึกไฟล์
related_path = os.path.join(floder_path, 'Related_News_Hybrid.csv')
unrelated_path = os.path.join(floder_path, 'Unrelated_News_Hybrid.csv')
related_df.to_csv(related_path, index=False, encoding='utf-8')
unrelated_df.to_csv(unrelated_path, index=False, encoding='utf-8')

print(f"💾 บันทึกข่าวที่เกี่ยวข้องที่: Related_News_Hybrid.csv")
print(f"💾 บันทึกข่าวที่ไม่เกี่ยวข้องที่: Unrelated_News_Hybrid.csv")
