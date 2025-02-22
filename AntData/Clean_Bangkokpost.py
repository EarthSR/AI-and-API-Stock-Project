import pandas as pd

# 🔹 โหลดข้อมูลจากไฟล์ CSV
file_path = "D:/StockData/AI-and-API-Stock-Project/news_data/news_with_sentiment_gpu.csv"
df = pd.read_csv(file_path)

# 🔹 แปลงชื่อคอลัมน์ให้เป็นตัวพิมพ์เล็กทั้งหมด (ป้องกัน KeyError)
df.columns = df.columns.str.lower()

# 🔹 แปลงคอลัมน์วันที่ให้เป็น datetime และตัดข่าวที่เก่ากว่า 1 ม.ค. 2018
df["date"] = pd.to_datetime(df["date"], errors="coerce")
cutoff_date = pd.Timestamp("2018-01-01")
df = df[df["date"] >= cutoff_date]

# 🔹 คีย์เวิร์ดของหุ้นไทย (ต้องมีอย่างน้อย 1 ตัว)
stock_keywords = {
    "ADVANC": ["ADVANC", "AIS", "Advanced Info Service"],
    "INTUCH": ["INTUCH", "Intouch Holdings"],
    "TRUE": ["TRUE", "True Corporation"],
    "DITTO": ["DITTO", "Ditto Thailand"],
    "DIF": ["DIF", "Digital Infrastructure Fund"],
    "INSET": ["INSET", "Internet Thailand"],
    "JMART": ["JMART", "Jay Mart"],
    "INET": ["INET", "Internet Thailand"],
    "JAS": ["JAS", "Jasmine International"],
    "HUMAN": ["HUMAN", "Humanica"]
}

# 🔹 คีย์เวิร์ดเกี่ยวกับการลงทุน & เทคโนโลยีไทย (ต้องมีอย่างน้อย 1 ตัว)
general_keywords = ["Stock", "Finance", "SET", "Thailand", "ตลาดหุ้น", "หุ้นไทย", "ลงทุน", "ตลาดหลักทรัพย์", "เทคโนโลยี", "ดิจิทัล", "AI", "Digital"]

# 🔹 คีย์เวิร์ดต่างประเทศที่อาจไม่เกี่ยวข้องกับไทย (ถ้ามีไทยเกี่ยวข้อง → เก็บไว้)
foreign_only_keywords = ["US market", "China stock", "Europe economy", "Dow Jones", "NASDAQ", "S&P 500", "India Sensex"]

# 🔹 คีย์เวิร์ดที่เกี่ยวข้องกับ Crypto และ Blockchain (ต้องไม่มีเลย)
crypto_keywords = ["crypto", "bitcoin", "ethereum", "stablecoin", "blockchain", "decentralized finance", "NFT", "Web3"]

# 🔹 ฟังก์ชันช่วยตรวจสอบว่าข่าวผ่านทุกเงื่อนไขหรือไม่
def is_valid_news(title, description):
    title = str(title).lower()
    description = str(description).lower()

    # ✅ 1️⃣ ข่าวต้องเกี่ยวข้องกับหุ้นไทย (ต้องมีอย่างน้อย 1 ตัว)
    relevant_stock = any(
        keyword.lower() in title or keyword.lower() in description
        for keywords in stock_keywords.values()
        for keyword in keywords
    )

    # ✅ 2️⃣ ข่าวต้องเกี่ยวข้องกับการลงทุนหรือเทคโนโลยีในไทย (ต้องมีอย่างน้อย 1 ตัว)
    relevant_general = any(
        keyword.lower() in title or keyword.lower() in description
        for keyword in general_keywords
    )

    # ❌ 3️⃣ ข่าวต้องไม่เกี่ยวข้องกับ Crypto, Blockchain (ห้ามมีแม้แต่ 1 ตัว)
    crypto_news = any(
        keyword.lower() in title or keyword.lower() in description
        for keyword in crypto_keywords
    )

    # ✅ 4️⃣ ถ้ามีคีย์เวิร์ดต่างประเทศ ต้องมี "Thailand" หรือ "หุ้นไทย" เกี่ยวข้องด้วย → เก็บไว้
    has_foreign_keyword = any(
        keyword.lower() in title or keyword.lower() in description
        for keyword in foreign_only_keywords
    )
    has_thailand = "thailand" in title or "thailand" in description or "หุ้นไทย" in title or "หุ้นไทย" in description

    # 🔹 ตัดข่าวที่เป็นต่างประเทศล้วน ๆ (ถ้ามีคำต่างประเทศแต่ไม่มีไทย → ตัดออก)
    foreign_news = has_foreign_keyword and not has_thailand

    # ✅ ✅ ✅ ✅ ต้องผ่านทุกเงื่อนไข → เก็บข่าวนี้ไว้
    return relevant_stock and relevant_general and not crypto_news and not foreign_news

# 🔹 กรองข้อมูล: คัดเฉพาะข่าวที่ผ่านทุกเงื่อนไข
filtered_df = df[df.apply(lambda row: is_valid_news(row["title"], row["description"]), axis=1)]

# 🔹 บันทึกผลลัพธ์ลงไฟล์ใหม่
output_file = "D:/StockData/AI-and-API-Stock-Project/news_data/filtered_news3.csv"
filtered_df.to_csv(output_file, index=False)
print(f"✅ Filtering Complete! Saved {len(filtered_df)} relevant news articles to {output_file}")
