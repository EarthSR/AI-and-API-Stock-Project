import pandas as pd
from sqlalchemy import create_engine, text
import mysql.connector
from concurrent.futures import ThreadPoolExecutor
import threading
import time

DB_CONFIG = {
    "host": "localhost",
    "user": "root", 
    "password": "1234",
    "database": "TradeMine",
    "autocommit": False,
    "connect_timeout": 60,
    "read_timeout": 600,
    "write_timeout": 600,
    "pool_size": 10,  # เพิ่ม pool size
    "pool_reset_session": False
}

def get_db_connection():
    """สร้างการเชื่อมต่อฐานข้อมูลใหม่"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as e:
        print(f"❌ การเชื่อมต่อฐานข้อมูลล้มเหลว: {e}")
        return None

def optimize_mysql_settings(cursor):
    """ปรับตั้งค่า MySQL เพื่อความเร็ว"""
    optimize_queries = [
        "SET SESSION innodb_lock_wait_timeout = 300",
        "SET SESSION lock_wait_timeout = 300", 
        "SET SESSION sql_mode = ''",
        "SET SESSION autocommit = 0",
        "SET SESSION innodb_buffer_pool_size = 2147483648",  # 2GB
        "SET SESSION bulk_insert_buffer_size = 268435456",   # 256MB
        "SET SESSION innodb_flush_log_at_trx_commit = 2",    # ลดการ sync
        "SET SESSION sync_binlog = 0",                       # ปิด binlog sync
        "SET SESSION foreign_key_checks = 0",               # ปิดการตรวจสอบ foreign key
        "SET SESSION unique_checks = 0"                     # ปิดการตรวจสอบ unique ชั่วคราว
    ]
    
    for query in optimize_queries:
        try:
            cursor.execute(query)
        except Exception as e:
            print(f"⚠️ ไม่สามารถตั้งค่า: {query} - {e}")

def merge_backup_optimized():
    """เวอร์ชันที่ปรับปรุงความเร็ว"""
    start_time = time.time()
    
    conn = get_db_connection()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    try:
        print("🚀 เริ่มต้นการปรับปรุงฐานข้อมูล...")
        
        # ปรับตั้งค่า MySQL
        optimize_mysql_settings(cursor)
        
        # ตรวจสอบข้อมูลปัจจุบัน
        cursor.execute("SELECT COUNT(*) FROM News")
        current_count = cursor.fetchone()[0]
        print(f"📊 ข้อมูลปัจจุบัน: {current_count:,} รายการ")
        
        # อ่าน backup file แบบ chunked เพื่อประหยัด memory
        print("📁 กำลังอ่าน backup file...")
        backup_df = pd.read_csv('news_202507172300.csv', low_memory=False)
        backup_df = backup_df.fillna('')
        print(f"📁 ข้อมูล backup: {len(backup_df):,} รายการ")
        
        # ดึง existing URLs แบบ efficient
        print("🔍 ตรวจสอบข้อมูลที่มีอยู่แล้ว...")
        cursor.execute("SELECT URL FROM News WHERE URL IS NOT NULL AND URL != ''")
        existing_urls = set(row[0] for row in cursor.fetchall())
        print(f"🔍 URL ที่มีอยู่แล้ว: {len(existing_urls):,} รายการ")
        
        # กรองข้อมูลใหม่
        new_data = backup_df[
            (~backup_df['URL'].isin(existing_urls)) & 
            (backup_df['URL'] != '') & 
            (backup_df['URL'].notna())
        ]
        print(f"🆕 ข้อมูลใหม่ที่จะแทรก: {len(new_data):,} รายการ")
        
        # === ส่วนที่ 1: แทรกข้อมูลใหม่แบบเร็ว ===
        if len(new_data) > 0:
            insert_new_data_fast(cursor, conn, new_data)
        
        # === ส่วนที่ 2: อัพเดตข้อมูลแบบเร็ว ===
        update_existing_data_fast(cursor, conn, backup_df)
        
        # รีเซ็ตการตั้งค่า
        cursor.execute("SET SESSION foreign_key_checks = 1")
        cursor.execute("SET SESSION unique_checks = 1")
        
        # ตรวจสอบผลลัพธ์
        cursor.execute("SELECT COUNT(*) FROM News")
        final_count = cursor.fetchone()[0]
        
        elapsed_time = time.time() - start_time
        added_count = final_count - current_count
        
        print("\n" + "=" * 60)
        print("🎉 สรุปผลการดำเนินการ")
        print("=" * 60)
        print(f"📊 ข้อมูลก่อนหน้า: {current_count:,} รายการ")
        print(f"📊 ข้อมูลหลังการอัพเดต: {final_count:,} รายการ")
        print(f"🆕 ข้อมูลที่เพิ่มขึ้น: +{added_count:,} รายการ")
        print(f"⏱️ เวลาที่ใช้: {elapsed_time:.2f} วินาที")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        conn.rollback()
        import traceback
        traceback.print_exc()
        
    finally:
        cursor.close()
        conn.close()
        print("✅ ปิดการเชื่อมต่อฐานข้อมูลเรียบร้อย")

def insert_new_data_fast(cursor, conn, new_data):
    """แทรกข้อมูลใหม่แบบเร็ว"""
    print("🚀 เริ่มแทรกข้อมูลใหม่แบบเร็ว...")
    
    # เตรียมข้อมูลสำหรับ bulk insert
    data_tuples = []
    for _, row in new_data.iterrows():
        title = str(row['Title']) if pd.notna(row['Title']) else ''
        url = str(row['URL']) if pd.notna(row['URL']) else ''
        content = str(row['Content']) if pd.notna(row['Content']) else ''
        published_date = str(row['PublishedDate']) if pd.notna(row['PublishedDate']) else None
        img = str(row['Img']) if pd.notna(row['Img']) else ''
        source = str(row['Source']) if pd.notna(row['Source']) else ''
        sentiment = str(row['Sentiment']) if pd.notna(row['Sentiment']) else ''
        confidence_score = float(row['ConfidenceScore']) if pd.notna(row['ConfidenceScore']) and row['ConfidenceScore'] != '' else None
        
        if url and url != 'nan':
            data_tuples.append((title, url, content, published_date, img, source, sentiment, confidence_score))
    
    if len(data_tuples) == 0:
        print("ℹ️ ไม่มีข้อมูลใหม่ที่จะแทรก")
        return
    
    # ใช้ LOAD DATA LOCAL INFILE หรือ bulk insert
    insert_query = """
    INSERT IGNORE INTO News (Title, URL, Content, PublishedDate, Img, Source, Sentiment, ConfidenceScore)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    # แทรกแบบ batch ขนาดใหญ่
    batch_size = 2000  # เพิ่มขนาด batch
    total_inserted = 0
    
    for i in range(0, len(data_tuples), batch_size):
        try:
            batch = data_tuples[i:i + batch_size]
            cursor.executemany(insert_query, batch)
            conn.commit()
            total_inserted += len(batch)
            
            percentage = (total_inserted / len(data_tuples)) * 100
            print(f"📝 INSERT: {total_inserted:,}/{len(data_tuples):,} ({percentage:.1f}%) ✅")
            
        except Exception as e:
            print(f"❌ INSERT batch error: {str(e)[:50]}")
            conn.rollback()
            continue
    
    print(f"✅ แทรกข้อมูลใหม่เสร็จสิ้น: {total_inserted:,} รายการ")

def update_existing_data_fast(cursor, conn, backup_df):
    """อัพเดตข้อมูลแบบเร็ว"""
    print("🚀 เริ่มอัพเดตข้อมูลแบบเร็ว...")
    
    # สร้าง temporary table สำหรับ bulk update
    cursor.execute("""
    CREATE TEMPORARY TABLE temp_updates (
        url VARCHAR(500) PRIMARY KEY,
        source VARCHAR(255),
        sentiment VARCHAR(50),
        confidence_score DECIMAL(5,4)
    )
    """)
    
    # เตรียมข้อมูลสำหรับ temp table
    update_data = backup_df[
        (backup_df['URL'] != '') & 
        (backup_df['URL'].notna()) &
        (backup_df['Source'].notna() | backup_df['Sentiment'].notna() | backup_df['ConfidenceScore'].notna())
    ].copy()
    
    if len(update_data) == 0:
        print("ℹ️ ไม่มีข้อมูลที่จะอัพเดต")
        return
    
    # แทรกข้อมูลลง temp table
    temp_insert_query = """
    INSERT IGNORE INTO temp_updates (url, source, sentiment, confidence_score)
    VALUES (%s, %s, %s, %s)
    """
    
    temp_tuples = []
    for _, row in update_data.iterrows():
        url = str(row['URL']) if pd.notna(row['URL']) else ''
        source = str(row['Source']) if pd.notna(row['Source']) else ''
        sentiment = str(row['Sentiment']) if pd.notna(row['Sentiment']) else ''
        confidence_score = float(row['ConfidenceScore']) if pd.notna(row['ConfidenceScore']) and row['ConfidenceScore'] != '' else None
        
        if url and url != 'nan':
            temp_tuples.append((url, source, sentiment, confidence_score))
    
    print(f"📝 กำลังเตรียมข้อมูลสำหรับอัพเดต: {len(temp_tuples):,} รายการ")
    
    # แทรกข้อมูลลง temp table แบบ batch
    batch_size = 5000
    for i in range(0, len(temp_tuples), batch_size):
        batch = temp_tuples[i:i + batch_size]
        cursor.executemany(temp_insert_query, batch)
        conn.commit()
        
        percentage = ((i + len(batch)) / len(temp_tuples)) * 100
        print(f"📝 TEMP INSERT: {i + len(batch):,}/{len(temp_tuples):,} ({percentage:.1f}%)")
    
    # อัพเดตข้อมูลจาก temp table แบบ bulk
    print("🔄 กำลังอัพเดตข้อมูลจาก temp table...")
    
    bulk_update_query = """
    UPDATE News n
    INNER JOIN temp_updates t ON n.URL = t.url
    SET 
        n.Source = CASE WHEN t.source != '' THEN t.source ELSE n.Source END,
        n.Sentiment = CASE WHEN t.sentiment != '' THEN t.sentiment ELSE n.Sentiment END,
        n.ConfidenceScore = CASE WHEN t.confidence_score IS NOT NULL THEN t.confidence_score ELSE n.ConfidenceScore END
    WHERE 
        (n.Source IS NULL OR n.Source = '' OR 
         n.Sentiment IS NULL OR n.Sentiment = '' OR 
         n.ConfidenceScore IS NULL)
    """
    
    cursor.execute(bulk_update_query)
    updated_rows = cursor.rowcount
    conn.commit()
    
    # ลบ temp table
    cursor.execute("DROP TEMPORARY TABLE temp_updates")
    
    print(f"✅ อัพเดตข้อมูลเสร็จสิ้น: {updated_rows:,} รายการ")

def merge_backup_sqlalchemy_fast():
    """ใช้ SQLAlchemy แบบเร็ว"""
    print("🚀 ใช้ SQLAlchemy แบบเร็ว...")
    
    # สร้าง engine ด้วยการตั้งค่าที่เหมาะสม
    engine = create_engine(
        'mysql+mysqlconnector://root:1234@localhost/TradeMine',
        pool_size=20,
        max_overflow=0,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False,
        connect_args={
            "autocommit": False,
            "connect_timeout": 60,
            "read_timeout": 600,
            "write_timeout": 600
        }
    )
    
    try:
        with engine.connect() as conn:
            # ปรับตั้งค่า
            conn.execute(text("SET SESSION innodb_flush_log_at_trx_commit = 2"))
            conn.execute(text("SET SESSION sync_binlog = 0"))
            conn.execute(text("SET SESSION foreign_key_checks = 0"))
            conn.execute(text("SET SESSION unique_checks = 0"))
            
            # ตรวจสอบข้อมูลปัจจุบัน
            result = conn.execute(text("SELECT COUNT(*) FROM News"))
            current_count = result.scalar()
            print(f"📊 ข้อมูลปัจจุบัน: {current_count:,} รายการ")
            
            # อ่าน backup
            backup_df = pd.read_csv('news_202507172300.csv', low_memory=False)
            backup_df = backup_df.fillna('')
            print(f"📁 ข้อมูล backup: {len(backup_df):,} รายการ")
            
            # ดึง existing URLs
            existing_urls_result = conn.execute(text("SELECT URL FROM News WHERE URL IS NOT NULL AND URL != ''"))
            existing_urls = set(row[0] for row in existing_urls_result)
            
            # กรองข้อมูลใหม่
            new_data = backup_df[
                (~backup_df['URL'].isin(existing_urls)) & 
                (backup_df['URL'] != '') & 
                (backup_df['URL'].notna())
            ]
            
            if len(new_data) > 0:
                print(f"🆕 แทรกข้อมูลใหม่: {len(new_data):,} รายการ")
                # ใช้ to_sql แบบ chunk
                new_data.to_sql('News', engine, if_exists='append', index=False, chunksize=2000, method='multi')
                print("✅ แทรกข้อมูลใหม่เสร็จสิ้น")
            
            # อัพเดตข้อมูล
            update_data = backup_df[
                (backup_df['URL'].isin(existing_urls)) &
                (backup_df['URL'] != '') & 
                (backup_df['URL'].notna())
            ]
            
            if len(update_data) > 0:
                print(f"🔄 อัพเดตข้อมูล: {len(update_data):,} รายการ")
                # สำหรับการอัพเดต ต้องใช้ raw SQL
                for idx, row in update_data.iterrows():
                    if idx % 1000 == 0:
                        print(f"🔄 อัพเดตแล้ว: {idx:,}/{len(update_data):,}")
                    
                    conn.execute(text("""
                    UPDATE News 
                    SET Source = :source, Sentiment = :sentiment, ConfidenceScore = :confidence
                    WHERE URL = :url AND (Source IS NULL OR Source = '' OR 
                                         Sentiment IS NULL OR Sentiment = '' OR 
                                         ConfidenceScore IS NULL)
                    """), {
                        'source': str(row['Source']) if pd.notna(row['Source']) else '',
                        'sentiment': str(row['Sentiment']) if pd.notna(row['Sentiment']) else '',
                        'confidence': float(row['ConfidenceScore']) if pd.notna(row['ConfidenceScore']) and row['ConfidenceScore'] != '' else None,
                        'url': str(row['URL'])
                    })
                
                conn.commit()
                print("✅ อัพเดตข้อมูลเสร็จสิ้น")
            
            # รีเซ็ตการตั้งค่า
            conn.execute(text("SET SESSION foreign_key_checks = 1"))
            conn.execute(text("SET SESSION unique_checks = 1"))
            
            # ตรวจสอบผลลัพธ์
            result = conn.execute(text("SELECT COUNT(*) FROM News"))
            final_count = result.scalar()
            print(f"📊 ข้อมูลหลังการอัพเดต: {final_count:,} รายการ")
            
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("เลือกวิธีการที่ปรับปรุงแล้ว:")
    print("1. MySQL Connector แบบปรับปรุง (แนะนำ)")
    print("2. SQLAlchemy แบบเร็ว")
    print("3. เปรียบเทียบความเร็ว")
    
    choice = input("กรุณาเลือก (1, 2, หรือ 3): ").strip()
    
    if choice == "1":
        merge_backup_optimized()
    elif choice == "2":
        merge_backup_sqlalchemy_fast()
    elif choice == "3":
        print("🏁 เปรียบเทียบความเร็ว...")
        print("=" * 50)
        
        print("🚀 ทดสอบ MySQL Connector แบบปรับปรุง:")
        start = time.time()
        merge_backup_optimized()
        time1 = time.time() - start
        
        print(f"\n⏱️ เวลาที่ใช้: {time1:.2f} วินาที")
        print("=" * 50)
        
    else:
        print("❌ กรุณาเลือก 1, 2, หรือ 3")