import os
import pandas as pd
import requests
import mysql.connector
from dotenv import load_dotenv
import sys

# ✅ ป้องกัน UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# ✅ โหลดตัวแปรจาก .env
load_dotenv()

# ✅ ฟังก์ชันดึงข้อมูล GDP จาก World Bank API
def fetch_gdp_data(country_codes, start_year=2000, end_year=2023):
    url = f"https://api.worldbank.org/v2/country/{';'.join(country_codes)}/indicator/NY.GDP.MKTP.CD?date={start_year}:{end_year}&format=json&per_page=500"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception("ไม่สามารถดึงข้อมูลจาก World Bank API ได้")
    
    data = response.json()
    if not data or len(data) < 2:
        raise Exception("ไม่มีข้อมูล GDP ที่ดึงมาได้")

    records = []
    for entry in data[1]:  # ข้อมูล GDP อยู่ที่ index 1
        if entry["value"] is not None:
            records.append({
                "Country": entry["country"]["value"],
                "Country Code": entry["country"]["id"],
                "Year": int(entry["date"]),
                "GDP (current US$)": float(entry["value"])
            })

    return pd.DataFrame(records)

# ✅ ดึงข้อมูล GDP ของไทยและอเมริกา
countries = ["THA", "USA"]
df_gdp = fetch_gdp_data(countries, start_year=2000, end_year=2023)

# ✅ แปลงค่า NaN เป็น None
df_gdp = df_gdp.where(pd.notna(df_gdp), None)

# ✅ บันทึกข้อมูลลง CSV
df_gdp.to_csv("GDP_AllData.csv", index=False)
df_gdp[df_gdp["Country Code"] == "THA"].to_csv("GDP_TH.csv", index=False)
df_gdp[df_gdp["Country Code"] == "USA"].to_csv("GDP_US.csv", index=False)
print("✅ บันทึกข้อมูลลงไฟล์ CSV สำเร็จ: GDP_AllData.csv, GDP_TH.csv, GDP_US.csv")

# ✅ เชื่อมต่อฐานข้อมูลและบันทึกข้อมูล
try:
    print("🔗 กำลังเชื่อมต่อกับฐานข้อมูล ...")
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        autocommit=True
    )
    cursor = conn.cursor()
    print("✅ เชื่อมต่อฐานข้อมูลสำเร็จ!")

    # ✅ ตรวจสอบค่าที่มีอยู่ใน Database
    cursor.execute("SELECT Country, Year FROM GDP")
    existing_data = set(cursor.fetchall())

    # ✅ กรองเฉพาะข้อมูลที่ยังไม่มีใน Database
    new_data = [
        (row["Country"], row["Country Code"], row["Year"], row["GDP (current US$)"])
        for _, row in df_gdp.iterrows()
        if (row["Country"], row["Year"]) not in existing_data
    ]

    if new_data:
        insert_gdp_query = """
        INSERT INTO GDP (Country, `Country Code`, Year, `GDP (current US$)`)
        VALUES (%s, %s, %s, %s);
        """
        cursor.executemany(insert_gdp_query, new_data)
        print(f"✅ บันทึกข้อมูลลง GDP: {len(new_data)} รายการ")
    else:
        print("✅ ไม่มีข้อมูลใหม่ที่ต้องเพิ่ม")

except mysql.connector.Error as err:
    print(f"❌ เกิดข้อผิดพลาด: {err}")

finally:
    cursor.close()
    conn.close()
    print("🔹 ปิดการเชื่อมต่อฐานข้อมูลแล้ว")
