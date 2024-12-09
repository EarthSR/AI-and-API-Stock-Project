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

    # Remove duplicate rows based on 'title' and 'link'
    df_cleaned = df.drop_duplicates(subset=['title', 'link'], keep='first')

    # Number of rows removed due to duplicates
    duplicate_removed = initial_rows - len(df_cleaned)

    # Define a function to parse and format the date
    def parse_and_format_date(date_str):
        # List of possible date formats
        date_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d"
        ]
    
        for date_format in date_formats:
            try:
                # Try parsing the date with each format
                parsed_date = datetime.strptime(date_str, date_format)
                # Format the date string into desired format
                return parsed_date.strftime("%Y-%m-%d")
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
    print(f"- Cleaned data saved to: {output_file}\n")

# Input and output files
input_files = [
    './bangkok_post_news_cleaned.csv','./imf_news_cleaned.csv', './investing_news_cleaned.csv', './yahoo_finance_news_cleaned.csv'
]  # List of input CSV files

output_file = 'news_cleaned.csv'  # Output CSV file
clean_and_process_data(input_files, output_file)

# Display first 5 rows of the result
try:
    df = pd.read_csv(output_file, encoding='utf-8')
    print("\nCleaned Data (5 rows):")
    print(df.head())
except FileNotFoundError:
    print(f"Error: File {output_file} not found.")
