# Import required libraries
import pandas as pd
import ta
from sklearn.preprocessing import LabelEncoder, RobustScaler

# Load and preprocess data
# Load stock data and sort by Ticker and Date
df_stock = pd.read_csv("./GRU_Model/cleaned_data.csv", parse_dates=["Date"]).sort_values(by=["Ticker", "Date"])
df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')

# Load news data
df_news = pd.read_csv("./GRU_Model/news_with_sentiment_gpu.csv")
df_news['Date'] = pd.to_datetime(df_news['Date'], errors='coerce')

# Map Sentiment values to numerical representations
df_news['Sentiment'] = df_news['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
df_news = df_news[['Date', 'Sentiment', 'Confidence']]

# Merge stock data and news data on Date
df = pd.merge(df_stock, df_news, on='Date', how='left')

# Fill missing values
df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

# Add features
df['Change'] = df['Close'] - df['Open']
df['Change (%)'] = (df['Change'] / df['Open']) * 100
df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
df['RSI'].fillna(method='ffill', inplace=True)
df['RSI'].fillna(0, inplace=True)
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].rolling(window=9).mean()
bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
df['Bollinger_High'] = bollinger.bollinger_hband()
df['Bollinger_Low'] = bollinger.bollinger_lband()
df.fillna(0, inplace=True)

# Label Encode Ticker
ticker_encoder = LabelEncoder()
df['Ticker_ID'] = ticker_encoder.fit_transform(df['Ticker'])
num_tickers = len(ticker_encoder.classes_)
print(f"Number of tickers: {num_tickers}")

# Split data into train and test based on date
sorted_dates = df['Date'].unique()
train_cutoff = sorted_dates[int(len(sorted_dates) * 6 / 7)]  # 6/7 of the data as training
train_df = df[df['Date'] <= train_cutoff].copy()
test_df = df[df['Date'] > train_cutoff].copy()

print("Train cutoff:", train_cutoff)
print("First date in train set:", train_df['Date'].min())
print("Last date in train set:", train_df['Date'].max())

# Create target variable by shifting 'Close' prices
train_targets_price = train_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
train_df = train_df.iloc[:-1]
test_targets_price = test_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
test_df = test_df.iloc[:-1]

# Define feature columns
feature_columns = ['Open', 'Close', 'High', 'Low', 'Volume', 'RSI', 'SMA_5', 'SMA_10', 
                   'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'Sentiment', 'Confidence']

train_features = train_df[feature_columns].values
test_features = test_df[feature_columns].values
train_ticker_id = train_df['Ticker_ID'].values
test_ticker_id = test_df['Ticker_ID'].values

# Scale numeric features using RobustScaler
scaler = RobustScaler()
numeric_columns_to_scale = ['Open', 'Close', 'High', 'Low', 'Volume']
df_stock[numeric_columns_to_scale] = scaler.fit_transform(df_stock[numeric_columns_to_scale])

# Scale features and targets
scaler_features = RobustScaler()
train_features_scaled = scaler_features.fit_transform(train_features)
test_features_scaled = scaler_features.transform(test_features)

scaler_target = RobustScaler()
train_targets_scaled = scaler_target.fit_transform(train_targets_price)
test_targets_scaled = scaler_target.transform(test_targets_price)

# Save processed data (optional)
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("Data preprocessing completed successfully!")
