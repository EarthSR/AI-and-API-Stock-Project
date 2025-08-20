import yfinance as yf
import pandas as pd
import datetime
import sys
import os
from dotenv import load_dotenv
import io
from pandas_market_calendars import get_calendar
import requests
import time
try:
    import mysql.connector
except ImportError:
    print("⚠️ mysql-connector-python not installed. Skipping database operations.")
    mysql = None

# ✅ ป้องกัน UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')

# ✅ ตรวจสอบระดับของโฟลเดอร์
CURRENT_DIR = os.getcwd()
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.env')
load_dotenv(path)

# ✅ ดึงตัวแปรสภาพแวดล้อม
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
FMP_API_KEY = os.getenv("FMP_API_KEY")  # เพิ่ม FMP API Key
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")  # เพิ่ม Alpha Vantage API Key

if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME]) and mysql:
    raise ValueError("❌ ขาดค่าการตั้งค่าฐานข้อมูลในไฟล์ .env")

# ✅ ตรวจสอบโครงสร้างตารางฐานข้อมูล (เพื่อดึงวันที่ล่าสุด)
def check_table_structure():
    if not mysql:
        return False
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = conn.cursor()
        cursor.execute("SHOW COLUMNS FROM StockDetail")
        columns = [col[0] for col in cursor.fetchall()]
        cursor.close()
        conn.close()
        expected_columns = ['Date', 'StockSymbol']
        missing_columns = [col for col in expected_columns if col not in columns]
        if missing_columns:
            print(f"❌ Missing columns in StockDetail: {missing_columns}")
            print("⚠️ Using default start date (2018-01-01) due to table issues.")
            return False
        print("✅ Table structure is sufficient for date checking")
        return True
    except Exception as e:
        print(f"❌ Error checking table structure: {e}")
        print("⚠️ Using default start date (2018-01-01) due to table issues.")
        return False

# ✅ ฟังก์ชันดึงข้อมูลจาก Alpha Vantage
def get_alpha_vantage_data(ticker, api_key):
    """
    ดึงข้อมูลหุ้นจาก Alpha Vantage API
    """
    if not api_key:
        print("⚠️ Alpha Vantage API Key not found in environment variables")
        return None
    
    try:
        # แปลง ticker format สำหรับ Alpha Vantage
        av_symbol = ticker  # ใช้ .BK format
        
        # สร้าง URL สำหรับ Alpha Vantage API
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={av_symbol}&apikey={api_key}&outputsize=compact"
        
        print(f"🔄 กำลังดึงข้อมูล {ticker} จาก Alpha Vantage API...")
        
        # ส่งคำขอ API
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # ตรวจสอบว่ามีข้อมูลหรือไม่
        if 'Time Series (Daily)' not in data:
            if 'Error Message' in data:
                print(f"⚠️ Alpha Vantage Error for {ticker}: {data['Error Message']}")
            elif 'Information' in data:
                print(f"⚠️ Alpha Vantage Info for {ticker}: {data['Information']}")
            else:
                print(f"⚠️ ไม่พบข้อมูลสำหรับ {ticker} จาก Alpha Vantage")
            return None
        
        # แปลงข้อมูลเป็น DataFrame
        time_series = data['Time Series (Daily)']
        df_data = []
        
        for date_str, values in time_series.items():
            df_data.append({
                'Date': pd.to_datetime(date_str),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['5. volume'])
            })
        
        df = pd.DataFrame(df_data)
        df = df.set_index('Date').sort_index()
        
        # ลบ timezone ถ้ามี
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        print(f"✅ ดึงข้อมูล {ticker} จาก Alpha Vantage สำเร็จ: {len(df)} แถว")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error for {ticker} from Alpha Vantage: {e}")
        return None
    except Exception as e:
        print(f"❌ Error fetching {ticker} from Alpha Vantage: {e}")
        return None

def get_fmp_data(ticker, start_date, end_date, api_key):
    """
    ดึงข้อมูลหุ้นจาก Financial Modeling Prep API
    """
    if not api_key:
        print("⚠️ FMP API Key not found in environment variables")
        return None
    
    try:
        # แปลง ticker format สำหรับ FMP (ลบ .BK)
        fmp_symbol = ticker.replace('.BK', '')
        
        # สร้าง URL สำหรับ FMP API
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{fmp_symbol}?apikey={api_key}"
        
        print(f"🔄 กำลังดึงข้อมูล {ticker} จาก FMP API...")
        
        # ส่งคำขอ API
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # ตรวจสอบว่ามีข้อมูลหรือไม่
        if 'historical' not in data or not data['historical']:
            print(f"⚠️ ไม่พบข้อมูลสำหรับ {ticker} จาก FMP")
            return None
        
        # แปลงข้อมูลเป็น DataFrame
        df = pd.DataFrame(data['historical'])
        
        # แปลงรูปแบบวันที่และเรียงลำดับ
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # กรองข้อมูลตามช่วงวันที่
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        
        # เปลี่ยนชื่อคอลัมน์ให้ตรงกับรูปแบบ yfinance
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # เลือกเฉพาะคอลัมน์ที่ต้องการ
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[required_columns]
        
        # ลบ timezone ถ้ามี
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        print(f"✅ ดึงข้อมูล {ticker} จาก FMP สำเร็จ: {len(df)} แถว")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error for {ticker} from FMP: {e}")
        return None
    except Exception as e:
        print(f"❌ Error fetching {ticker} from FMP: {e}")
        return None

# ✅ ฟังก์ชันดึงข้อมูลด้วย fallback mechanism
def get_stock_data_with_fallback(ticker, start_date, end_date, max_retries=3):
    """
    ดึงข้อมูลหุ้นด้วยการลอง yfinance -> FMP -> Alpha Vantage
    """
    data = None
    source_used = None
    
    # ลอง yfinance ก่อน
    for attempt in range(max_retries):
        try:
            print(f"📡 กำลังดึงข้อมูล {ticker} จาก yfinance (ครั้งที่ {attempt + 1})...")
            stock = yf.Ticker(ticker)
            ticker_data = stock.history(start=start_date, end=end_date, interval='1d')
            
            if not ticker_data.empty:
                print(f"✅ ดึงข้อมูล {ticker} จาก yfinance สำเร็จ: {len(ticker_data)} แถว")
                data = ticker_data
                source_used = "yfinance"
                break
            else:
                print(f"⚠️ ไม่พบข้อมูลสำหรับ {ticker} จาก yfinance (ครั้งที่ {attempt + 1})")
                
        except Exception as e:
            print(f"⚠️ Error for {ticker} from yfinance (ครั้งที่ {attempt + 1}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(2)  # รอ 2 วินาทีก่อนลองใหม่
    
    # ถ้า yfinance ล้มเหลว ลอง FMP
    if data is None or data.empty:
        print(f"🔄 yfinance ล้มเหลว กำลังลอง FMP สำหรับ {ticker}...")
        
        for attempt in range(max_retries):
            try:
                fmp_data = get_fmp_data(ticker, start_date, end_date, FMP_API_KEY)
                
                if fmp_data is not None and not fmp_data.empty:
                    data = fmp_data
                    source_used = "FMP"
                    break
                else:
                    print(f"⚠️ ไม่พบข้อมูลสำหรับ {ticker} จาก FMP (ครั้งที่ {attempt + 1})")
                    
            except Exception as e:
                print(f"⚠️ Error for {ticker} from FMP (ครั้งที่ {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(3)  # รอ 3 วินาทีก่อนลองใหม่ (FMP มี rate limit)
    
    # ถ้า FMP ก็ล้มเหลว ลอง Alpha Vantage
    if data is None or data.empty:
        print(f"🔄 FMP ล้มเหลว กำลังลอง Alpha Vantage สำหรับ {ticker}...")
        
        for attempt in range(max_retries):
            try:
                av_data = get_alpha_vantage_data(ticker, ALPHA_VANTAGE_API_KEY)
                
                if av_data is not None and not av_data.empty:
                    # กรองข้อมูลตามช่วงวันที่
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    av_data = av_data[(av_data.index >= start_dt) & (av_data.index <= end_dt)]
                    
                    if not av_data.empty:
                        data = av_data
                        source_used = "Alpha Vantage"
                        break
                    else:
                        print(f"⚠️ ไม่มีข้อมูลในช่วงวันที่ที่ต้องการสำหรับ {ticker} จาก Alpha Vantage (ครั้งที่ {attempt + 1})")
                else:
                    print(f"⚠️ ไม่พบข้อมูลสำหรับ {ticker} จาก Alpha Vantage (ครั้งที่ {attempt + 1})")
                    
            except Exception as e:
                print(f"⚠️ Error for {ticker} from Alpha Vantage (ครั้งที่ {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(15)  # รอ 15 วินาที (Alpha Vantage มี rate limit เข้มงวด)
    
    return data, source_used

# ✅ เพิ่มฟังก์ชันตรวจสอบ Volume anomalies
def check_volume_anomalies(data):
    """ตรวจสอบและรายงาน Volume ที่ผิดปกติ"""
    print("\n🔍 Volume Analysis:")
    
    for ticker in data['Ticker'].unique():
        ticker_data = data[data['Ticker'] == ticker].copy()
        ticker_data['Date'] = pd.to_datetime(ticker_data['Date'])
        
        # หาวันซื้อขายที่ Volume = 0
        zero_volume_trading_days = ticker_data[
            (ticker_data['Volume'] == 0) & 
            (ticker_data['Date'].dt.dayofweek < 5)  # จันทร์-ศุกร์
        ]
        
        if len(zero_volume_trading_days) > 0:
            print(f"⚠️ {ticker}: Found {len(zero_volume_trading_days)} trading days with Volume = 0")
            print(f"   Dates: {zero_volume_trading_days['Date'].dt.strftime('%Y-%m-%d').tolist()}")
        
        # สถิติ Volume
        non_zero_volume = ticker_data[ticker_data['Volume'] > 0]['Volume']
        if len(non_zero_volume) > 0:
            print(f"✅ {ticker}: Avg Volume = {non_zero_volume.mean():,.0f}, "
                  f"Min = {non_zero_volume.min():,.0f}, "
                  f"Max = {non_zero_volume.max():,.0f}")
        else:
            print(f"❌ {ticker}: No valid volume data found!")

# ✅ ตรวจสอบวันที่ล่าสุดจากฐานข้อมูล
latest_dates = {}
tickers = ['ADVANC.BK', 'TRUE.BK', 'DITTO.BK', 'DIF.BK', 
           'INSET.BK', 'JMART.BK', 'INET.BK', 'JAS.BK', 'HUMAN.BK']
has_valid_table = False
today = datetime.datetime.now()
current_date = (today - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

if all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME]) and mysql:
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            autocommit=True
        )
        cursor = conn.cursor()
        print("✅ เชื่อมต่อฐานข้อมูลสำเร็จ!")
        has_valid_table = check_table_structure()
        
        if has_valid_table:
            for ticker in tickers:
                stock_name = ticker.replace('.BK', '')
                try:
                    cursor.execute("SELECT MAX(Date) FROM StockDetail WHERE StockSymbol = %s AND ClosePrice IS NOT NULL" , (stock_name,))
                    result = cursor.fetchone()[0]
                    if result is None:
                        latest_dates[ticker] = "2018-01-01"
                    elif result > today.date():
                        print(f"⚠️ Future date found for {ticker}: {result}. Using default start date (2018-01-01)")
                        latest_dates[ticker] = "2018-01-01"
                    else:
                        latest_dates[ticker] = (result + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                except Exception as e:
                    print(f"⚠️ Error fetching latest date for {ticker}: {e}")
                    latest_dates[ticker] = "2018-01-01"
        else:
            for ticker in tickers:
                latest_dates[ticker] = "2018-01-01"
        
        cursor.close()
        conn.close()
        print("🔹 ปิดการเชื่อมต่อฐานข้อมูลแล้ว")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        for ticker in tickers:
            latest_dates[ticker] = "2018-01-01"
else:
    print("⚠️ Missing database configuration or mysql-connector-python, using default start date (2018-01-01)")
    for ticker in tickers:
        latest_dates[ticker] = "2018-01-01"

# ✅ กำหนดวันที่เริ่มต้นและสิ้นสุด
start_date_db = min(latest_dates.values())
# ขยายช่วงวันที่ให้กว้างขึ้น (ย้อนหลัง 30 วัน สำหรับการทดสอบ)
start_date = (pd.to_datetime(min(latest_dates.values())) - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
end_date = current_date

# ตรวจสอบว่ามีข้อมูลใหม่ให้ดึงหรือไม่
if start_date >= end_date:
    print(f"❌ ไม่มีข้อมูลใหม่ให้ดึง (start_date: {start_date} >= end_date: {end_date})")
    sys.exit(0)

print(f"🔹 ดึงข้อมูลจาก {start_date} ถึง {end_date}")

# แสดงข้อมูล API ที่ใช้ได้
print(f"🔧 API Status:")
print(f"   - yfinance: ✅ Available")
print(f"   - FMP API: {'✅ Available' if FMP_API_KEY else '❌ No API Key'}")

# ✅ ดึงปฏิทิน SET เพื่อระบุวันซื้อขาย
try:
    set_calendar = get_calendar('XBKK')
    trading_days = set_calendar.schedule(start_date=start_date, end_date=end_date).index
    print("✅ ใช้ปฏิทิน SET (XBKK) สำเร็จ")
except Exception as e:
    print(f"⚠️ ไม่สามารถใช้ปฏิทิน SET ได้: {e}")
    # ใช้วันทำการทั่วไปแทน (จันทร์-ศุกร์)
    all_dates_range = pd.date_range(start=start_date, end=end_date, freq='D')
    trading_days = all_dates_range[all_dates_range.weekday < 5]  # 0=จันทร์, 4=ศุกร์
    print("✅ ใช้วันทำการทั่วไป (จันทร์-ศุกร์) แทน")

# ✅ ฟังก์ชันตรวจสอบว่าเป็นวันซื้อขายหรือไม่
def is_trading_day(date, trading_days):
    return pd.Timestamp(date) in trading_days

# ✅ ฟังก์ชันเติมข้อมูลวันหยุดด้วย Forward Fill และ Rolling Mean (ปรับปรุง Volume)
def impute_holiday_data(ticker_data, all_dates, ticker, window=3):
    print(f"🔄 กำลังเติมข้อมูลวันหยุดสำหรับ {ticker}...")
    ticker_data = ticker_data.copy()
    
    # ตรวจสอบคอลัมน์ที่จำเป็น
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in ticker_data.columns for col in required_columns):
        print(f"❌ Missing required columns for {ticker}: {required_columns}")
        return pd.DataFrame()
    
    # แปลง index เป็น datetime และลบ timezone
    ticker_data.index = pd.to_datetime(ticker_data.index).tz_localize(None)
    
    # เก็บข้อมูล Volume เดิมก่อน reindex
    original_volume = ticker_data['Volume'].copy()
    original_volume = original_volume[original_volume > 0]  # เก็บเฉพาะค่าที่มากกว่า 0
    
    print(f"📊 {ticker} - ข้อมูลดิบ: {len(ticker_data)} แถว")
    print(f"📊 {ticker} - ช่วงวันที่ดิบ: {ticker_data.index.min()} ถึง {ticker_data.index.max()}")
    
    # Reindex เพื่อรวมวันหยุด
    ticker_data = ticker_data.reindex(all_dates, method=None)
    print(f"📊 {ticker} - หลัง reindex: {len(ticker_data)} แถว")
    
    # คำนวณเปอร์เซ็นต์ข้อมูลที่หายไป
    missing_percentage = ticker_data[required_columns].isnull().mean() * 100
    print(f"🔍 {ticker} - ข้อมูลที่หายไป: Open: {missing_percentage['Open']:.1f}%, High: {missing_percentage['High']:.1f}%, Low: {missing_percentage['Low']:.1f}%, Close: {missing_percentage['Close']:.1f}%, Volume: {missing_percentage['Volume']:.1f}%")
    
    if missing_percentage.sum() > 50:
        print(f"⚠️ Warning: ข้อมูลหายไปมากสำหรับ {ticker} ({missing_percentage.sum():.2f}%)")
    
    # เติมข้อมูลราคาด้วย Forward Fill, Backward Fill และ Rolling Mean
    print(f"🔄 {ticker} - เติมข้อมูลราคาด้วย Forward Fill...")
    ticker_data[['Open', 'High', 'Low', 'Close']] = ticker_data[['Open', 'High', 'Low', 'Close']].ffill(limit=5)
    
    print(f"🔄 {ticker} - เติมข้อมูลราคาด้วย Backward Fill...")
    ticker_data[['Open', 'High', 'Low', 'Close']] = ticker_data[['Open', 'High', 'Low', 'Close']].bfill(limit=5)
    
    print(f"🔄 {ticker} - เติมข้อมูลราคาด้วย Rolling Mean (window={window})...")
    # ใช้ rolling mean สำหรับข้อมูลที่ยังหายไปอยู่
    for col in ['Open', 'High', 'Low', 'Close']:
        missing_mask = ticker_data[col].isnull()
        if missing_mask.any():
            # คำนวณ rolling mean และเติมเฉพาะตำแหน่งที่หายไป
            rolling_mean = ticker_data[col].rolling(window=window, min_periods=1, center=True).mean()
            ticker_data.loc[missing_mask, col] = rolling_mean.loc[missing_mask]
    
    # จัดการ Volume แยกต่างหาก - ปรับปรุงการจัดการ Volume
    print(f"🔄 {ticker} - เติมข้อมูล Volume...")
    
    # ใช้ SET trading calendar หรือ weekdays ถ้าไม่มี
    try:
        set_cal = get_calendar('XBKK')
        trading_schedule = set_cal.schedule(start_date=all_dates[0], end_date=all_dates[-1])
        trading_days_set = set(trading_schedule.index.normalize())
    except:
        # ใช้วันจันทร์-ศุกร์แทน
        trading_days_set = set(all_dates[all_dates.weekday < 5])
    
    # คำนวณ Volume เฉลี่ยจากข้อมูลเดิม
    avg_volume = original_volume.mean() if len(original_volume) > 0 else 1000000
    
    for date in ticker_data.index:
        date_normalized = pd.Timestamp(date).normalize()
        
        if date_normalized in trading_days_set:
            # วันซื้อขาย: จัดการ Volume ที่หายไป
            if pd.isna(ticker_data.loc[date, 'Volume']) or ticker_data.loc[date, 'Volume'] == 0:
                # ใช้ forward fill จากข้อมูลก่อนหน้า
                prev_volumes = ticker_data.loc[:date, 'Volume'].dropna()
                prev_volumes = prev_volumes[prev_volumes > 0]
                
                if len(prev_volumes) > 0:
                    # ใช้ค่าเฉลี่ยของ 3 วันล่าสุด
                    recent_avg = prev_volumes.tail(3).mean()
                    ticker_data.loc[date, 'Volume'] = recent_avg
                else:
                    # ใช้ค่าเฉลี่ยโดยรวม
                    ticker_data.loc[date, 'Volume'] = avg_volume
                    
                print(f"🔄 Imputed volume for {ticker} on {date.strftime('%Y-%m-%d')}: {ticker_data.loc[date, 'Volume']:,.0f}")
        else:
            # วันหยุด: ตั้งเป็น 0
            ticker_data.loc[date, 'Volume'] = 0
    
    # คำนวณ Changepercent
    print(f"🔄 {ticker} - คำนวณ Changepercent...")
    ticker_data['Changepercent'] = ((ticker_data['Close'] - ticker_data['Open']) / ticker_data['Open'] * 100).fillna(0)
    
    # ตรวจสอบผลลัพธ์หลังเติมข้อมูล
    final_missing = ticker_data[required_columns].isnull().mean() * 100
    print(f"✅ {ticker} - ข้อมูลหายไปหลังเติม: Open: {final_missing['Open']:.1f}%, High: {final_missing['High']:.1f}%, Low: {final_missing['Low']:.1f}%, Close: {final_missing['Close']:.1f}%, Volume: {final_missing['Volume']:.1f}%")
    
    # นับจำนวนวันที่เติมข้อมูล
    original_dates = set(ticker_data.dropna(subset=['Open']).index)
    all_dates_set = set(ticker_data.index)
    imputed_dates = all_dates_set - original_dates
    print(f"📈 {ticker} - เติมข้อมูลแล้ว {len(imputed_dates)} วัน จากทั้งหมด {len(all_dates_set)} วัน")

    return ticker_data

# ✅ สร้างช่วงวันที่ทั้งหมด (รวมวันหยุด)
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
print(f"📅 สร้างช่วงวันที่: {len(all_dates)} วัน (จาก {start_date} ถึง {end_date})")

# ✅ ดึงข้อมูลด้วย fallback mechanism (yfinance -> FMP)
data_dict = {}
sources_used = {}

print("🔄 เริ่มดึงข้อมูลด้วย fallback mechanism...")
for ticker in tickers:
    ticker_data, source = get_stock_data_with_fallback(ticker, start_date, end_date, max_retries=3)
    
    if ticker_data is not None and not ticker_data.empty:
        print(f"📋 ตัวอย่างข้อมูล {ticker} (จาก {source}):")
        print(ticker_data.head(3).to_string())
        
        # แสดงข้อมูล Volume ก่อนการประมวลผล
        original_volumes = ticker_data['Volume'][ticker_data['Volume'] > 0]
        if len(original_volumes) > 0:
            print(f"🔵 Original volume stats for {ticker}: Mean={original_volumes.mean():,.0f}, Min={original_volumes.min():,.0f}, Max={original_volumes.max():,.0f}")
        
        # เติมข้อมูลวันหยุด
        ticker_data = impute_holiday_data(ticker_data, all_dates, ticker, window=3)  
        stock_name = ticker.replace('.BK', '')
        ticker_data['Ticker'] = stock_name
        data_dict[ticker] = ticker_data
        sources_used[ticker] = source
        print(f"✅ เติมข้อมูลวันหยุดสำหรับ {ticker} เสร็จสิ้น (แหล่งข้อมูล: {source})")
    else:
        print(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้จากทั้ง yfinance และ FMP")

if not data_dict:
    print("⚠️ ไม่มีข้อมูลใหม่จากทุกแหล่ง")
    sys.exit(0)

# แสดงสรุปแหล่งข้อมูลที่ใช้
print(f"\n📊 สรุปแหล่งข้อมูลที่ใช้:")
for ticker, source in sources_used.items():
    print(f"🔹 {ticker}: {source}")

# ✅ ประมวลผลข้อมูลและแสดงสถิติการเติมข้อมูล
data_list = []
total_imputed_days = 0

print("\n🔍 ตรวจสอบข้อมูลที่เติมในวันหยุด:")
for ticker, ticker_data in data_dict.items():
    if ticker_data.empty:
        print(f"⚠️ ไม่มีข้อมูลสำหรับ {ticker}")
        continue
    
    # นับวันที่เติมข้อมูล
    imputed_count = 0
    for date in ticker_data.index:
        if not is_trading_day(date, trading_days):
            if not ticker_data.loc[date, ['Open', 'High', 'Low', 'Close']].isnull().any():
                imputed_count += 1
                if imputed_count <= 3:  # แสดงเฉพาะ 3 วันแรก
                    print(f"📝 {ticker} - {date.strftime('%Y-%m-%d')} (วันหยุด): เติมข้อมูลด้วยค่าเฉลี่ย")
    
    if imputed_count > 3:
        print(f"📝 {ticker} - เติมข้อมูลวันหยุดทั้งหมด {imputed_count} วัน")
    
    total_imputed_days += imputed_count
    data_list.append(ticker_data)

print(f"\n✅ เติมข้อมูลวันหยุดทั้งหมด: {total_imputed_days} วัน")

if not data_list:
    print("❌ ไม่มีข้อมูลใด ๆ ที่ดึงมาได้")
    sys.exit(1)

# รวมข้อมูลทั้งหมดเป็น DataFrame เดียว
print("🔄 รวมข้อมูลทั้งหมด...")
cleaned_data = pd.concat(data_list).reset_index()

# ✅ เปลี่ยนชื่อคอลัมน์ให้ตรงกับ CSV
cleaned_data = cleaned_data.rename(columns={
    'index': 'Date',
    'Ticker': 'Ticker',
    'Open': 'Open',
    'High': 'High',
    'Low': 'Low',
    'Close': 'Close',
    'Volume': 'Volume',
    'Changepercent': 'Changepercent'
})

cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date']).dt.strftime('%Y-%m-%d')

# ✅ จัดเรียงคอลัมน์
columns_to_keep = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Changepercent']
cleaned_data = cleaned_data[columns_to_keep]

# ✅ กรองข้อมูลที่ไม่ถูกต้องและตัดวันที่
print("🔹 กรองข้อมูลที่ไม่ถูกต้องออก...")
before_filter = len(cleaned_data)
cleaned_data = cleaned_data[
    (cleaned_data['Open'].notna()) &
    (cleaned_data['High'].notna()) &
    (cleaned_data['Low'].notna()) &
    (cleaned_data['Close'].notna()) &
    (cleaned_data['Date'] >= start_date_db) &
    (cleaned_data['Date'] <= end_date)
]
after_filter = len(cleaned_data)
print(f"🔹 กรองข้อมูลแล้ว: {before_filter} -> {after_filter} แถว")

# ✅ เรียงลำดับและลบข้อมูลซ้ำ
cleaned_data = cleaned_data.sort_values(['Date', 'Ticker']).reset_index(drop=True)
cleaned_data = cleaned_data.drop_duplicates(subset=['Date', 'Ticker'], keep='first')

# ✅ เพิ่มการตรวจสอบ Volume หลังประมวลผลข้อมูล
check_volume_anomalies(cleaned_data)

# ✅ เพิ่มการตรวจสอบ Volume = 0 ในวันซื้อขาย
print("\n🔧 Checking for zero volume on trading days...")
cleaned_data['Date_dt'] = pd.to_datetime(cleaned_data['Date'])
cleaned_data['is_weekday'] = cleaned_data['Date_dt'].dt.dayofweek < 5

# นับจำนวนแถวที่มี Volume = 0 ในวันซื้อขาย
zero_volume_weekdays = cleaned_data[(cleaned_data['Volume'] == 0) & (cleaned_data['is_weekday'])]
if len(zero_volume_weekdays) > 0:
    print(f"⚠️ Found {len(zero_volume_weekdays)} records with zero volume on weekdays")
    
    # แสดงรายละเอียด
    for ticker in zero_volume_weekdays['Ticker'].unique():
        ticker_zeros = zero_volume_weekdays[zero_volume_weekdays['Ticker'] == ticker]
        print(f"   {ticker}: {len(ticker_zeros)} days - {ticker_zeros['Date'].tolist()}")

# ลบคอลัมน์ช่วยเหลือ
cleaned_data = cleaned_data.drop(['Date_dt', 'is_weekday'], axis=1)

# ✅ ลบข้อมูลของวันนี้ (ถ้าต้องการ)
today = datetime.datetime.today().strftime('%Y-%m-%d')
initial_rows = len(cleaned_data)
cleaned_data = cleaned_data[cleaned_data['Date'].astype(str) != today]
if len(cleaned_data) < initial_rows:
    print(f"🔹 ลบข้อมูลของวันนี้ ({today}) แล้ว")

# ✅ บันทึกข้อมูลเป็นไฟล์ CSV
csv_path = os.path.join(os.path.dirname(__file__), "Stock", "stock_data_thai.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
cleaned_data.to_csv(csv_path, index=False)
print(f"✅ บันทึกไฟล์ CSV สำเร็จที่: {csv_path}")

# ✅ แสดงสถิติข้อมูล
print(f"\n📊 สรุปผลลัพธ์:")
print(f"🔹 จำนวนข้อมูลทั้งหมด: {len(cleaned_data)} แถว")
print(f"🔹 วันที่ที่มีข้อมูล: {cleaned_data['Date'].nunique()} วัน")
if not cleaned_data.empty:
    print(f"🔹 ช่วงวันที่: {cleaned_data['Date'].min()} ถึง {cleaned_data['Date'].max()}")
    for ticker in cleaned_data['Ticker'].unique():
        ticker_data = cleaned_data[cleaned_data['Ticker'] == ticker]
        trading_days_count = len([d for d in ticker_data['Date'] if is_trading_day(pd.to_datetime(d), trading_days)])
        holiday_days_count = len(ticker_data) - trading_days_count
        source = sources_used.get(f"{ticker}.BK", "Unknown")
        
        # คำนวณ Average Volume
        avg_volume = ticker_data[ticker_data['Volume'] > 0]['Volume'].mean()
        avg_volume_str = f"{avg_volume:,.0f}" if not pd.isna(avg_volume) else "N/A"
        
        print(f"🔹 {ticker}: {len(ticker_data)} แถว (วันซื้อขาย: {trading_days_count}, วันหยุดที่เติม: {holiday_days_count}), วันที่ล่าสุด {ticker_data['Date'].max()}, Avg Volume: {avg_volume_str} [แหล่ง: {source}]")
    
    print("\n📋 ตัวอย่างข้อมูลที่เติมวันหยุด:")
    # แสดงตัวอย่างข้อมูลวันหยุดที่เติม
    sample_data = cleaned_data.head(15)
    for _, row in sample_data.iterrows():
        date_status = "วันซื้อขาย" if is_trading_day(pd.to_datetime(row['Date']), trading_days) else "วันหยุด(เติม)"
        print(f"  {row['Date']} | {row['Ticker']} | Close: {row['Close']:.2f} | Volume: {row['Volume']:,.0f} | ({date_status})")
else:
    print("❌ ไม่มีข้อมูลที่ถูกต้องในช่วงวันที่ที่กำหนด")