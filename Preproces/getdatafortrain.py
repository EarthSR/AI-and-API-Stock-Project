import pandas as pd
import mysql.connector
from datetime import datetime
import os
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataExtractor:
    def __init__(self, host="localhost", user="root", password="1234", database="TradeMine"):
        self.connection_config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database,
            'autocommit': True
        }
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = mysql.connector.connect(**self.connection_config)
            self.cursor = self.conn.cursor()
            logger.info("✅ เชื่อมต่อฐานข้อมูลสำเร็จ!")
            return True
        except mysql.connector.Error as e:
            logger.error(f"❌ การเชื่อมต่อฐานข้อมูลล้มเหลว: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("✅ ปิดการเชื่อมต่อฐานข้อมูลเรียบร้อย")
    
    def rename_dataframe_headers(self, df):
        """Rename DataFrame headers to more descriptive names"""
        column_mapping = {
            'TotalRevenue': 'Total Revenue',
            'QoQGrowth': 'QoQ Growth (%)',
            'EPS': 'Earnings Per Share (EPS)',
            'ROE': 'ROE (%)',
            'YoYGrowth': 'YoY Growth (%)',
            'NetProfitMargin': 'Net Profit Margin (%)',
            'DebtToEquity': 'Debt to Equity',
            'PERatio': 'P/E Ratio',
            'P_BV_Ratio': 'P/BV Ratio',
            'Dividend_Yield': 'Dividend Yield (%)',
            'Changepercen': 'Change Percent (%)',
            'NetProfit': 'Net Profit',
            'StockSymbol': 'Ticker',
            'OpenPrice': 'Open',
            'HighPrice': 'High',
            'LowPrice': 'Low',
            'ClosePrice': 'Close',
            'Volume': 'Volume',
            'Sentiment': 'Sentiment',
            'positive_news': 'positive_news',
            'negative_news': 'negative_news',
            'neutral_news': 'neutral_news'
        }
        return df.rename(columns=column_mapping)

    def clean_data(self, df):
        """Clean data: remove rows with zero prices, null ClosePrice, duplicate rows, weekend days, and replace zeros with NaN in numeric columns"""
        # Log initial row count
        initial_rows = len(df)
        
        # ตรวจสอบว่าคอลัมน์ 'ClosePrice' มีอยู่ใน DataFrame หรือไม่
        if 'ClosePrice' not in df.columns:
            logger.warning("⚠️ คอลัมน์ 'ClosePrice' ไม่มีอยู่ใน DataFrame กำลังเพิ่มคอลัมน์ด้วยค่า NaN")
            df['ClosePrice'] = np.nan
        
        # ลบแถวที่มีค่า ClosePrice เป็น NaN
        if 'ClosePrice' in df.columns:
            initial_rows_close = len(df)
            df = df.dropna(subset=['ClosePrice'])
            logger.info(f"🗑️ ลบ {initial_rows_close - len(df)} แถวที่มีค่า ClosePrice เป็น null")
        
        # Remove rows where any price column is zero
        price_columns = ['OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice']
        for col in price_columns:
            if col in df.columns:
                initial_rows_price = len(df)
                df = df[df[col] != 0]
                logger.info(f"🗑️ ลบ {initial_rows_price - len(df)} แถวที่มีค่า 0 ในคอลัมน์ {col}")
        
        # Remove duplicate rows based on Date and Ticker
        initial_rows = len(df)
        df = df.drop_duplicates(subset=['Date', 'StockSymbol'], keep='first')
        logger.info(f"🗑️ ลบ {initial_rows - len(df)} แถวที่ซ้ำกัน")
        
        # Replace zeros with NaN in numeric columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_columns = [col for col in numeric_columns if col not in ['positive_news', 'negative_news', 'neutral_news']]
        for col in numeric_columns:
            df[col] = df[col].replace(0, np.nan)
        logger.info("✅ แทนที่ค่า 0 ด้วย NaN ในคอลัมน์ตัวเลข")
        
        logger.info(f"✅ การทำความสะอาดข้อมูลเสร็จสิ้น: เหลือ {len(df)} แถวจาก {initial_rows} แถวเริ่มต้น")
        return df
    
    def build_query(self, stock_symbols=None, date_range=None, limit=None):
        """Build dynamic SQL query based on parameters"""
        base_query = """
        SELECT 
            StockDetail.Date, 
            StockDetail.StockSymbol , 
            StockDetail.OpenPrice , 
            StockDetail.HighPrice , 
            StockDetail.LowPrice , 
            StockDetail.ClosePrice , 
            StockDetail.Volume , 
            StockDetail.P_BV_Ratio,
            StockDetail.Sentiment, 
            StockDetail.TotalRevenue,
            StockDetail.YoYGrowth, 
            StockDetail.QoQGrowth, 
            StockDetail.EPS, 
            StockDetail.ROE, 
            StockDetail.NetProfitMargin, 
            StockDetail.DebtToEquity, 
            StockDetail.PERatio,
            StockDetail.NetProfit, 
            StockDetail.Dividend_Yield, 
            StockDetail.positive_news, 
            StockDetail.negative_news, 
            StockDetail.neutral_news
        FROM StockDetail
        LEFT JOIN Stock ON StockDetail.StockSymbol = Stock.StockSymbol
        """
        return base_query
    
    def analyze_data_quality(self, df):
        """Comprehensive data quality analysis for ML training"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE DATA QUALITY ANALYSIS")
        print("="*60)
        
        # Convert Decimal columns to float first
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Try to convert to numeric, errors='ignore' will keep original if can't convert
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        # Basic Information
        print(f"\n📈 Basic Information:")
        print(f"  • Total Records: {len(df):,}")
        print(f"  • Total Columns: {len(df.columns)}")
        print(f"  • Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Date Range Analysis
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            print(f"\n📅 Date Range Analysis:")
            print(f"  • Start Date: {df['Date'].min()}")
            print(f"  • End Date: {df['Date'].max()}")
            print(f"  • Date Range: {(df['Date'].max() - df['Date'].min()).days} days")
            print(f"  • Unique Dates: {df['Date'].nunique():,}")
        
        # Ticker Analysis
        if 'Ticker' in df.columns:
            ticker_counts = df['Ticker'].value_counts()
            print(f"\n🏢 Ticker Analysis:")
            print(f"  • Unique Tickers: {df['Ticker'].nunique()}")
            print(f"  • Records per Ticker:")
            print(f"    - Min: {ticker_counts.min():,}")
            print(f"    - Max: {ticker_counts.max():,}")
            print(f"    - Mean: {ticker_counts.mean():.0f}")
            print(f"    - Median: {ticker_counts.median():.0f}")
            
            print(f"\n  • Top 10 Tickers by Record Count:")
            for ticker, count in ticker_counts.head(10).items():
                print(f"    - {ticker}: {count:,} records")
        
        # Missing Values Analysis
        print(f"\n❓ Missing Values Analysis:")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        print(f"  • Columns with Missing Values:")
        for col in missing_data[missing_data > 0].index:
            print(f"    - {col}: {missing_data[col]:,} ({missing_percent[col]:.1f}%)")
        
        # Data Type Analysis
        print(f"\n🔧 Data Type Analysis:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  • {dtype}: {count} columns")
        
        # Price Analysis (Critical for Stock Data) - แก้ไข error handling
        price_columns = ['Open', 'High', 'Low', 'Close']
        available_price_cols = [col for col in price_columns if col in df.columns]
        
        if available_price_cols:
            print(f"\n💰 Price Data Analysis:")
            for col in available_price_cols:
                if col in df.columns:
                    try:
                        # Convert to numeric if needed
                        numeric_col = pd.to_numeric(df[col], errors='coerce')
                        print(f"  • {col}:")
                        print(f"    - Min: ${numeric_col.min():.2f}")
                        print(f"    - Max: ${numeric_col.max():.2f}")
                        print(f"    - Mean: ${numeric_col.mean():.2f}")
                        print(f"    - Std: ${numeric_col.std():.2f}")
                        print(f"    - Missing: {numeric_col.isnull().sum():,} ({numeric_col.isnull().sum()/len(df)*100:.1f}%)")
                    except Exception as e:
                        print(f"  • {col}: Error analyzing - {str(e)}")
        
        # Volume Analysis - แก้ไข error handling
        if 'Volume' in df.columns:
            try:
                volume_numeric = pd.to_numeric(df['Volume'], errors='coerce')
                print(f"\n📊 Volume Analysis:")
                print(f"  • Min Volume: {volume_numeric.min():,.0f}")
                print(f"  • Max Volume: {volume_numeric.max():,.0f}")
                print(f"  • Mean Volume: {volume_numeric.mean():,.0f}")
                print(f"  • Missing Volume: {volume_numeric.isnull().sum():,}")
            except Exception as e:
                print(f"\n📊 Volume Analysis: Error - {str(e)}")
        
        # Sentiment Analysis
        if 'Sentiment' in df.columns:
            sentiment_counts = df['Sentiment'].value_counts()
            print(f"\n😊 Sentiment Analysis:")
            for sentiment, count in sentiment_counts.items():
                print(f"  • {sentiment}: {count:,} ({count/len(df)*100:.1f}%)")
        
        # Financial Metrics Analysis - แก้ไข error handling
        financial_cols = ['Total Revenue', 'P/E Ratio', 'ROE (%)', 'Earnings Per Share (EPS)']
        available_financial = [col for col in financial_cols if col in df.columns]
        
        if available_financial:
            print(f"\n💼 Financial Metrics Analysis:")
            for col in available_financial:
                try:
                    financial_numeric = pd.to_numeric(df[col], errors='coerce')
                    non_null_count = financial_numeric.notna().sum()
                    print(f"  • {col}: {non_null_count:,} valid records ({non_null_count/len(df)*100:.1f}%)")
                except Exception as e:
                    print(f"  • {col}: Error analyzing - {str(e)}")
        
        # News Sentiment Analysis
        news_cols = ['positive_news', 'negative_news', 'neutral_news']
        available_news = [col for col in news_cols if col in df.columns]
        
        if available_news:
            print(f"\n📰 News Sentiment Analysis:")
            for col in available_news:
                try:
                    news_numeric = pd.to_numeric(df[col], errors='coerce')
                    print(f"  • {col}:")
                    print(f"    - Total: {news_numeric.sum():,}")
                    print(f"    - Mean per day: {news_numeric.mean():.1f}")
                    print(f"    - Max per day: {news_numeric.max()}")
                except Exception as e:
                    print(f"  • {col}: Error analyzing - {str(e)}")
        
        # Data Quality Score
        print(f"\n🎯 Data Quality Score:")
        
        # Calculate completeness score
        completeness_score = ((len(df.columns) - missing_data.count()) / len(df.columns)) * 100
        
        # Calculate price data completeness
        price_completeness = 0
        if available_price_cols:
            try:
                price_missing = sum([pd.to_numeric(df[col], errors='coerce').isnull().sum() for col in available_price_cols])
                total_price_cells = len(available_price_cols) * len(df)
                price_completeness = ((total_price_cells - price_missing) / total_price_cells) * 100
            except:
                price_completeness = 0
        
        # Calculate ticker balance score
        ticker_balance = 0
        if 'Ticker' in df.columns:
            ticker_counts = df['Ticker'].value_counts()
            ticker_std = ticker_counts.std()
            ticker_mean = ticker_counts.mean()
            ticker_balance = max(0, 100 - (ticker_std / ticker_mean * 100))
        
        print(f"  • Overall Completeness: {completeness_score:.1f}%")
        print(f"  • Price Data Completeness: {price_completeness:.1f}%")
        print(f"  • Ticker Balance Score: {ticker_balance:.1f}%")
        
        # Recommendations
        print(f"\n💡 Recommendations for ML Training:")
        
        if price_completeness < 95:
            print("  ⚠️  Price data has missing values - consider interpolation")
        
        if ticker_balance < 80:
            print("  ⚠️  Unbalanced ticker distribution - may cause bias")
        
        missing_critical = missing_percent[missing_percent > 20]
        if len(missing_critical) > 0:
            print("  ⚠️  Critical missing data in:", ", ".join(missing_critical.index.tolist()))
        
        if len(df) < 10000:
            print("  ⚠️  Small dataset - consider data augmentation")
        
        if df['Ticker'].nunique() < 10:
            print("  ⚠️  Few tickers - limited diversification")
        
        print("  ✅ Ready for feature engineering and model training" if completeness_score > 85 else "  ❌ Data needs significant preprocessing")
        
        print("="*60)
        
        return {
            'completeness_score': completeness_score,
            'price_completeness': price_completeness,
            'ticker_balance': ticker_balance,
            'total_records': len(df),
            'unique_tickers': df['Ticker'].nunique() if 'Ticker' in df.columns else 0,
            'missing_data': missing_data.to_dict()
        }
    
    def extract_data(self, stock_symbols=None, date_range=None, limit=None):
        """Extract stock data with optional filters"""
        if not self.conn:
            logger.error("❌ ไม่มีการเชื่อมต่อฐานข้อมูล")
            return None

        query = self.build_query(stock_symbols, date_range, limit)

        try:
            logger.info(f"🔍 กำลังดึงข้อมูล... (Query: {len(query)} characters)")
            self.cursor.execute(query)
            data = self.cursor.fetchall()
            
            # Get column names
            columns = [desc[0] for desc in self.cursor.description]
            df = pd.DataFrame(data, columns=columns)
            if df.empty:
                logger.warning("⚠️ ไม่พบข้อมูลสำหรับเงื่อนไขที่ระบุ")
                return None
            
            # Clean data: remove rows with zero prices and replace zeros with NaN
            df = self.clean_data(df)
            
            logger.info(f"✅ ดึงข้อมูลได้ {len(df)} แถว")
            return df
            
        except mysql.connector.Error as e:
            logger.error(f"❌ เกิดข้อผิดพลาดในการรันคำสั่ง SQL: {e}")
            return None
    
    def save_to_csv(self, df, filename=None, include_timestamp=True):
        """Save DataFrame to CSV file"""
        if df is None or df.empty:
            logger.warning("⚠️ ไม่มีข้อมูลให้บันทึก")
            return False
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
            filename = f"stock_data_{timestamp}.csv" if timestamp else "stock_data.csv"
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"✅ บันทึกข้อมูลเป็นไฟล์ CSV เรียบร้อย: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ เกิดข้อผิดพลาดในการบันทึกไฟล์: {e}")
            return False
    
    def get_data_summary(self, df):
        """Get summary statistics of the data"""
        if df is None or df.empty:
            return None
        
        summary = {
            'total_records': len(df),
            'unique_stocks': df['Ticker'].nunique(),
            'date_range': (df['Date'].min(), df['Date'].max()),
            'columns': df.columns.tolist()
        }
        return summary

# Usage example
def main():
    extractor = StockDataExtractor()
    
    if not extractor.connect():
        return
    
    try:
        # Extract all data
        df = extractor.extract_data()
        
        if df is not None:
            # Rename headers first
            renamed_df = extractor.rename_dataframe_headers(df)
            
            # Fix YoY Growth column issue
            if 'YoY Growth (%)' in renamed_df.columns:
                if renamed_df['YoY Growth (%)'].eq(0).any():
                    renamed_df['YoY Growth (%)'] = np.nan

            # Show preview
            print("\n📊 ตัวอย่างข้อมูล:")
            print(renamed_df.head())
            
            # ✅ เพิ่มการวิเคราะห์ข้อมูลแบบละเอียด
            analysis_result = extractor.analyze_data_quality(renamed_df)
            
            # Show basic summary
            summary = extractor.get_data_summary(renamed_df)
            if summary:
                print(f"\n📈 สรุปข้อมูลพื้นฐาน:")
                print(f"- จำนวนรายการทั้งหมด: {summary['total_records']:,}")
                print(f"- จำนวนหุ้นที่ไม่ซ้ำ: {summary['unique_stocks']}")
                print(f"- ช่วงวันที่: {summary['date_range'][0]} ถึง {summary['date_range'][1]}")
            
            # Save to CSV
            extractor.save_to_csv(renamed_df, 'merged_stock_sentiment_financial_database.csv')
            
            # Save analysis results
            with open('data_quality_analysis.txt', 'w', encoding='utf-8') as f:
                f.write("Data Quality Analysis Results\n")
                f.write("="*50 + "\n")
                f.write(f"Completeness Score: {analysis_result['completeness_score']:.1f}%\n")
                f.write(f"Price Completeness: {analysis_result['price_completeness']:.1f}%\n")
                f.write(f"Ticker Balance: {analysis_result['ticker_balance']:.1f}%\n")
                f.write(f"Total Records: {analysis_result['total_records']:,}\n")
                f.write(f"Unique Tickers: {analysis_result['unique_tickers']}\n")
            
            print("✅ บันทึกผลการวิเคราะห์ใน 'data_quality_analysis.txt'")
            
    finally:
        extractor.disconnect()

if __name__ == "__main__":
    main()