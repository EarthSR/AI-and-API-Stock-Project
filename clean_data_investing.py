import pandas as pd
from datetime import datetime

def clean_and_process_data(input_files, output_file):
    # Initialize an empty list to hold dataframes
    all_data = []

    # Loop through all input files
    for input_file in input_files:
        try:
            # Read data from the CSV file
            df = pd.read_csv(input_file, encoding='utf-8')
            all_data.append(df)
        except FileNotFoundError:
            print(f"Error: File {input_file} not found.")
            continue

    # Concatenate all data into one dataframe
    df = pd.concat(all_data, ignore_index=True)

    # Number of rows before removing duplicates
    initial_rows = len(df)
    
    df_cleaned = df.copy()
    
    df_cleaned = df_cleaned.iloc[::-1]
    
    # Remove rows where 'date' is 'ไม่มีข้อมูล' or 'No Data'
    df_cleaned = df_cleaned[
        (~df_cleaned['date'].isin(['ไม่มีข้อมูล', 'No Data', 'No Date'])) &
        (~df_cleaned['description'].isin(['ไม่มีข้อมูล', 'No Data'])) &
        (~df_cleaned['link'].isin(['ไม่มีข้อมูล', 'No Data'])) &
        (~df_cleaned['title'].isin(['ไม่มีข้อมูล', 'No Data']))
    ]
    
    
    description_replaced = df_cleaned['content'].apply(lambda x: x != 'No Content').sum()
    df_cleaned['description'] = df_cleaned.apply(
        lambda row: row['content'] if row['content'] != 'No Content' else row['description'], axis=1
    )
    
    df_cleaned = df_cleaned.drop(columns=['content'])
    
    # Remove duplicate rows based on 'title' and 'link'
    df_cleaned = df_cleaned.drop_duplicates(subset=['title', 'link'], keep='first')
   
    # Number of rows removed due to duplicates
    duplicate_removed = initial_rows - len(df_cleaned)
    

    # Define a function to parse and format the date
    def parse_and_format_date(date_str):
        # List of possible date formats
        date_formats = [
            "%d %b %Y at %H:%M",  # Example: 6 Dec 2024 at 08:38
            "%a, %b %d, %Y, %I:%M %p",  # Example: Fri, Dec 6, 2024, 2:57 PM
            "%Y-%m-%d %H:%M:%S",  # Example: 2024-12-06 08:38:00
            "%b %d, %Y at %H:%M",  # Example: Dec 6, 2024 at 08:38
            "%d/%m/%Y %H:%M:%S",  # Example: 06/12/2024 08:38:00
            "%B %d, %Y"  # Example: December 6, 2024
        ]
    
        for date_format in date_formats:
            try:
                # Try parsing the date with each format
                parsed_date = datetime.strptime(date_str, date_format)
                # Format the date string into desired format
                return parsed_date.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                continue  # If parsing fails, try the next format
        # If none of the formats work, return None
        print(f"Error parsing date: {date_str}")
        return None

    # Apply the function to the 'date' column to clean and format it
    df_cleaned['date'] = df_cleaned['date'].apply(parse_and_format_date)

    # Drop rows with invalid 'date'
    df_cleaned = df_cleaned.dropna(subset=['date'])

    # Sort the data by 'date'
    df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])
    df_cleaned = df_cleaned.sort_values(by='date', ascending=True)
    
    columns_order = ['title', 'date', 'link', 'description']  # Modify this as per your preference
    df_cleaned = df_cleaned[columns_order]

    # Final row count
    final_rows = len(df_cleaned)

    # Save cleaned data in batches
    batch_size = 100
    for start_idx in range(0, final_rows, batch_size):
        end_idx = min(start_idx + batch_size, final_rows)
        batch = df_cleaned.iloc[start_idx:end_idx]
        if start_idx == 0:
            # Write header on the first batch
            batch.to_csv(output_file, index=False, encoding='utf-8', mode='w', header=True)
        else:
            # Append data in subsequent batches
            batch.to_csv(output_file, index=False, encoding='utf-8', mode='a', header=False)
        
        print(f"Processed rows {start_idx + 1} to {end_idx} of {final_rows}")

    # Summary report
    print("\n[Summary of Data Cleaning]")
    print(f"- Rows before removing duplicates: {initial_rows}")
    print(f"- Rows removed due to duplicates: {duplicate_removed}")
    print(f"- Rows remaining after cleaning: {final_rows}")
    print(f"- แถวที่ description ถูกแทนด้วย content: {description_replaced}")
    print(f"- Cleaned data saved to: {output_file}\n")

# Input and output files
input_files = [
    './News_all/investing_news_full.csv'
]  # List of input CSV files

output_file = 'investing_news_cleaned.csv'  # Output CSV file
clean_and_process_data(input_files, output_file)

# Display first 5 rows of the result
try:
    df = pd.read_csv(output_file, encoding='utf-8')
    print("\nCleaned Data (5 rows):")
    print(df.head())
except FileNotFoundError:
    print(f"Error: File {output_file} not found.")
