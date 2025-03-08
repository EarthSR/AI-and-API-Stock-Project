import os
import pandas as pd
import mysql.connector
from dotenv import load_dotenv
import sys

# ✅ ป้องกัน UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# ✅ โหลดตัวแปรจาก .env
load_dotenv()

# ✅ ตรวจสอบระดับของโฟลเดอร์
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ✅ กำหนดพาธของไฟล์ CSV
GDP_USA_CSV = os.path.join(BASE_DIR, "GDP_USA.csv")
GDP_THAI_CSV = os.path.join(BASE_DIR, "GDP_Thai.csv")

# ✅ อ่านข้อมูล GDP

def load_gdp_data(filepath):
    df = pd.read_csv(filepath)
    
    # ✅ แปลงชื่อคอลัมน์ปีจาก "2000 [YR2000]" -> "2000"
    df.columns = [col.split(" [")[0] if " [YR" in col else col for col in df.columns]
    
    # ✅ แปลงข้อมูลจาก Wide Format เป็น Long Format
    df = df.melt(id_vars=["Series Name", "Series Code", "Country Name", "Country Code"],
                 var_name="Year", value_name="GDP_Value")
    
    # ✅ แปลง Year ให้เป็น int
    df["Year"] = df["Year"].astype(int)
    
    return df

# โหลดข้อมูลจากทั้ง 2 ไฟล์
df_usa = load_gdp_data(GDP_USA_CSV)
df_thai = load_gdp_data(GDP_THAI_CSV)

# ✅ รวมข้อมูลเข้าด้วยกัน
df_gdp = pd.concat([df_usa, df_thai])

# ✅ แปลงค่า NaN เป็น None สำหรับฐานข้อมูล
df_gdp = df_gdp.where(pd.notna(df_gdp), None)

# ✅ เชื่อมต่อฐานข้อมูล
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
    cursor.execute("SELECT Country_Name, Year FROM GDP")
    existing_data = set(cursor.fetchall())
    
    # ✅ กรองเฉพาะข้อมูลที่ยังไม่มีใน Database
    new_data = [tuple(row) for row in df_gdp.values.tolist() if (row[2], row[4]) not in existing_data]
    
    if new_data:
        insert_gdp_query = """
        INSERT INTO GDP (Series_Name, Series_Code, Country_Name, Country_Code, Year, GDP_Value)
        VALUES (%s, %s, %s, %s, %s, %s);
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
