import pandas as pd
import numpy as np

# โหลดข้อมูล
file_path = "../American_stocks/stock_data_from_dates.csv"  # เปลี่ยนเป็นไฟล์ของคุณ
df = pd.read_csv(file_path, parse_dates=['Date'])

# เรียงลำดับตามวันที่
df = df.sort_values(by=['Ticker', 'Date'])

# กำหนดช่วงของค่า EMA ที่ต้องการทดสอบ (เริ่มที่ 2 เพิ่มทีละ 2)
ema_pairs = [(short, short * 2) for short in range(2, 26, 2)]  # คู่ของ EMA เช่น (2,4), (4,8), ...

# สร้าง DataFrame สำหรับเก็บผลลัพธ์
results = []

# วนลูปทดสอบ EMA crossover
for short_ema, long_ema in ema_pairs:
    df[f'EMA_{short_ema}'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=short_ema, adjust=False).mean())
    df[f'EMA_{long_ema}'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=long_ema, adjust=False).mean())

    # คำนวณจุดตัด
    df['Signal'] = np.where(df[f'EMA_{short_ema}'] > df[f'EMA_{long_ema}'], 1, 0)  # 1 = Buy, 0 = Sell
    df['Crossover'] = df['Signal'].diff()  # หาจุดเปลี่ยนแปลงของสัญญาณ

    # คำนวณผลตอบแทนจากการใช้กลยุทธ์
    df['Return'] = df['Close'].pct_change()  # คำนวณ % เปลี่ยนแปลงของราคาหุ้น
    df['Strategy_Return'] = df['Return'] * df['Crossover'].shift(1)  # คำนวณผลตอบแทนตามสัญญาณ

    # รวมผลลัพธ์เป็น % สะสม
    performance = df.groupby('Ticker')['Strategy_Return'].sum()

    # บันทึกผลลัพธ์
    for ticker, perf in performance.items():
        results.append({'Ticker': ticker, 'Short_EMA': short_ema, 'Long_EMA': long_ema, 'Total_Return': perf})

# แปลงเป็น DataFrame
results_df = pd.DataFrame(results)

# เลือก EMA ที่ให้ผลตอบแทนสูงสุด
best_ema_crossover = results_df.loc[results_df.groupby('Ticker')['Total_Return'].idxmax()]

# บันทึกผลลัพธ์เป็น CSV
best_ema_crossover.to_csv("best_ema_crossover_results.csv", index=False)

print("✅ บันทึกผลลัพธ์ลงไฟล์ best_ema_crossover_results.csv เรียบร้อย!")
