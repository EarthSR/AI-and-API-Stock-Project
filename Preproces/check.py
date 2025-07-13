import pandas as pd
import numpy as np  # เพิ่มการนำเข้า numpy เพื่อใช้ np.isclose

# สมมติว่าข้อมูลอยู่ในไฟล์ CSV
# เปลี่ยน path ตามไฟล์ที่คุณมี
file1_path = './merged_stock_sentiment_financial_database.csv'  # ชุดข้อมูลแรก
file2_path = '../GRU_Model/merged_stock_sentiment_financial.csv'  # ชุดข้อมูลที่สอง

# อ่านข้อมูลจากไฟล์ CSV
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# แปลงคอลัมน์ Date เป็น datetime เพื่อให้เปรียบเทียบได้ง่าย
df1['Date'] = pd.to_datetime(df1['Date'])
df2['Date'] = pd.to_datetime(df2['Date'])

# ปัดทศนิยมสำหรับคอลัมน์ตัวเลขใน df2
df2['Open'] = df2['Open'].round(2)
df2['Close'] = df2['Close'].round(2)
df2['High'] = df2['High'].round(2)
df2['Low'] = df2['Low'].round(2)
df2['Volume'] = df2['Volume'].round(2)

# รวมข้อมูลโดยใช้ Date และ Ticker เป็นคีย์
merged_df = pd.merge(df1, df2, on=['Date', 'Ticker'], how='outer', suffixes=('_dataset1', '_dataset2'))

# สร้างรายการคอลัมน์ที่เหมือนกัน (ยกเว้น Date และ Ticker)
common_columns = [col for col in df1.columns if col in df2.columns and col not in ['Date', 'Ticker']]

# สร้าง DataFrame สำหรับเก็บผลลัพธ์การเปรียบเทียบ
comparison_results = []

# วนลูปผ่านแถวใน merged_df เพื่อเปรียบเทียบ
for index, row in merged_df.iterrows():
    date = row['Date']
    ticker = row['Ticker']
    
    # ตรวจสอบว่าข้อมูลมีอยู่ในทั้งสองชุดหรือไม่
    exists_in_df1 = not pd.isna(row[common_columns[0] + '_dataset1'])
    exists_in_df2 = not pd.isna(row[common_columns[0] + '_dataset2'])
    
    result = {
        'Date': date,
        'Ticker': ticker,
        'Exists_in_Dataset1': exists_in_df1,
        'Exists_in_Dataset2': exists_in_df2
    }
    
    # เปรียบเทียบคอลัมน์ที่เหมือนกัน
    for col in common_columns:
        val1 = row.get(col + '_dataset1')
        val2 = row.get(col + '_dataset2')
        
        # ตรวจสอบว่าแตกต่างหรือไม่ (ใช้ np.isclose สำหรับตัวเลขเพื่อป้องกันปัญหาการเปรียบเทียบทศนิยม)
        if pd.isna(val1) and pd.isna(val2):
            result[col + '_match'] = True
        elif pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
            result[col + '_match'] = pd.isna(val1) == pd.isna(val2) or (not pd.isna(val1) and not pd.isna(val2) and np.isclose(val1, val2, rtol=1e-6))
            result[col + '_dataset1'] = val1
            result[col + '_dataset2'] = val2
        else:
            result[col + '_match'] = val1 == val2
            result[col + '_dataset1'] = val1
            result[col + '_dataset2'] = val2
    
    # เพิ่มคอลัมน์ที่เฉพาะใน dataset1
    for col in [c for c in df1.columns if c not in df2.columns and c not in ['Date', 'Ticker']]:
        result[col + '_dataset1'] = row.get(col + '_dataset1')
    
    # เพิ่มคอลัมน์ที่เฉพาะใน dataset2
    for col in [c for c in df2.columns if c not in df1.columns and c not in ['Date', 'Ticker']]:
        result[col + '_dataset2'] = row.get(col + '_dataset2')
    
    comparison_results.append(result)

# สร้าง DataFrame จากผลลัพธ์
comparison_df = pd.DataFrame(comparison_results)

# แสดงผลลัพธ์บางส่วน
print("ตัวอย่างผลลัพธ์การเปรียบเทียบ:")
print(comparison_df.head())

# สรุปจำนวนแถวที่ไม่ตรงกันในคอลัมน์ที่เหมือนกัน
print("\nสรุปจำนวนแถวที่ไม่ตรงกัน:")
for col in common_columns:
    mismatch_count = comparison_df[comparison_df[col + '_match'] == False].shape[0]
    print(f"Column {col}: {mismatch_count} mismatches")

# สร้างไฟล์ CSV สำหรับผลลัพธ์ทั้งหมด
output_file = 'comparison_results.csv'
comparison_df.to_csv(output_file, index=False)
print(f"\nบันทึกผลลัพธ์ทั้งหมดลงในไฟล์: {output_file}")

# สร้างไฟล์ CSV เฉพาะแถวที่ไม่ตรงกัน
mismatch_df = comparison_df[comparison_df[[col + '_match' for col in common_columns]].eq(False).any(axis=1)]
mismatch_output_file = 'mismatch_results.csv'
mismatch_df.to_csv(mismatch_output_file, index=False)
print(f"บันทึกแถวที่ไม่ตรงกันลงในไฟล์: {mismatch_output_file}")

# สรุปจำนวนแถวที่ไม่ตรงกันทั้งหมด
total_mismatches = mismatch_df.shape[0]
print(f"\nจำนวนแถวที่มีข้อมูลไม่ตรงกันทั้งหมด: {total_mismatches}")