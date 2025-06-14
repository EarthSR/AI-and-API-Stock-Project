import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text

# Load environment variables
path = os.path.join(os.path.dirname('__file__'), '..', 'Preproces', 'config.env')
load_dotenv(path)

# Get database connection string from environment variables
DB_CONNECTION = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"

def get_db_connection():
    """
    Create a database connection using the connection string.
    """
    engine = create_engine(DB_CONNECTION)
    return engine.connect()

def save_predictions_to_db(predictions_df):
    engine = create_engine(DB_CONNECTION)

    try:
        with engine.begin() as connection:
            print(predictions_df.columns)

            for _, row in predictions_df.iterrows():
                date = row['Date']
                symbol = row['Ticker']  # เพิ่ม symbol

                check_query = text("""
                    SELECT COUNT(*) FROM stockdetail WHERE date = :date AND StockSymbol = :symbol;
                """)
                result = connection.execute(check_query, {"date": date, "symbol": symbol}).fetchone()

                if result[0] > 0:
                    update_query = text("""
                        UPDATE stockdetail
                        SET PredictionClose_LSTM = :Predicted_Price_LSTM, 
                            PredictionClose_GRU = :Predicted_Price_GRU, 
                            PredictionTrend_LSTM = :Predicted_Dir_LSTM, 
                            PredictionTrend_GRU = :Predicted_Dir_GRU
                        WHERE date = :date AND StockSymbol = :symbol;
                    """)
                    connection.execute(update_query, {
                        "Predicted_Price_LSTM": row['Predicted_Price_LSTM'],
                        "Predicted_Price_GRU": row['Predicted_Price_GRU'],
                        "Predicted_Dir_LSTM": row['Predicted_Dir_LSTM'],
                        "Predicted_Dir_GRU": row['Predicted_Dir_GRU'],
                        "date": date,
                        "symbol": symbol
                    })
                    print(f"✅ Updated record for {symbol} on {date}")
                else:
                    print(f"⚠️ No record found for {symbol} on {date}, skipping update.")
        print("✅ All predictions updated in the database successfully.")
    except Exception as e:
        print(f"❌ Error updating predictions in database: {e}")

def load_predictions(file):
    """
    Load predictions from the CSV file.
    """
    return pd.read_csv(file)

# File path to the predictions CSV file
file = './ensemble_predictions.csv'

# Load predictions from the CSV file
predictions_df = load_predictions(file)

# Update predictions in the database
if predictions_df is not None:
    save_predictions_to_db(predictions_df)
else:
    print("No predictions loaded. Skipping update operation.")
