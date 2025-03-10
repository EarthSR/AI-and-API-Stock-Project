import pandas as pd
from datetime import datetime

def clean_and_process_data(input_files, output_file):
    all_data = []

    for input_file in input_files:
        try:
            df = pd.read_csv(input_file, encoding='utf-8')
            all_data.append(df)
        except FileNotFoundError:
            print(f"Error: File {input_file} not found.")
            continue

    df = pd.concat(all_data, ignore_index=True)

    initial_rows = len(df)
    
    df_cleaned = df.copy()
    
    df_cleaned = df_cleaned.iloc[::-1]
    
    df_cleaned = df_cleaned[
        (~df_cleaned['date'].isin(['ไม่มีข้อมูล', 'No Data', 'No Date'])) &
        (~df_cleaned['description'].isin(['ไม่มีข้อมูล', 'No Data'])) &
        (~df_cleaned['link'].isin(['ไม่มีข้อมูล', 'No Data'])) &
        (~df_cleaned['title'].isin(['ไม่มีข้อมูล', 'No Data']))
    ]
    
    df_cleaned = df_cleaned.drop_duplicates(subset=['title', 'link'], keep='first')

    duplicate_removed = initial_rows - len(df_cleaned)

    def parse_and_format_date(date_str):
        date_formats = [
            "%d %b %Y at %H:%M",
            "%a, %b %d, %Y, %I:%M %p",
            "%Y-%m-%d %H:%M:%S",
            "%b %d, %Y at %H:%M",
            "%d/%m/%Y %H:%M:%S",
            "%B %d, %Y"
        ]
    
        for date_format in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, date_format)
                return parsed_date.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                continue
        print(f"Error parsing date: {date_str}")
        return None

    df_cleaned['date'] = df_cleaned['date'].apply(parse_and_format_date)
    df_cleaned = df_cleaned.dropna(subset=['date'])

    df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])
    df_cleaned = df_cleaned.sort_values(by='date', ascending=True)
    
    # แทนที่เครื่องหมาย "," ใน 'description' ด้วยช่องว่าง
    df_cleaned['description'] = df_cleaned['description'].str.replace(',', ' ', regex=False)

    columns_order = ['title', 'date', 'link', 'description']
    df_cleaned = df_cleaned[columns_order]

    final_rows = len(df_cleaned)

    batch_size = 100
    for start_idx in range(0, final_rows, batch_size):
        end_idx = min(start_idx + batch_size, final_rows)
        batch = df_cleaned.iloc[start_idx:end_idx]
        if start_idx == 0:
            batch.to_csv(output_file, index=False, encoding='utf-8', mode='w', header=True)
        else:
            batch.to_csv(output_file, index=False, encoding='utf-8', mode='a', header=False)
        
        print(f"Processed rows {start_idx + 1} to {end_idx} of {final_rows}")

    print("\n[Summary of Data Cleaning]")
    print(f"- Rows before removing duplicates: {initial_rows}")
    print(f"- Rows removed due to duplicates: {duplicate_removed}")
    print(f"- Rows remaining after cleaning: {final_rows}")
    print(f"- Cleaned data saved to: {output_file}\n")

input_files = [
    '../News_all/investing_news.csv'
]

output_file = 'investing_news_cleaned.csv'
clean_and_process_data(input_files, output_file)

try:
    df = pd.read_csv(output_file, encoding='utf-8')
    print("\nCleaned Data (5 rows):")
    print(df.head())
except FileNotFoundError:
    print(f"Error: File {output_file} not found.")
