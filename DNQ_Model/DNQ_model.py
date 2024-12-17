import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from collections import deque
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import joblib
import ta
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, filename='training.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')


# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=2000)  # Memory to store state-action-reward sequences
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            tf.keras.Input(shape=(10, 15)), # ระบุ input shape ด้วย Input Layer
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')  # Output layer
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        state = np.reshape(state, (1, 10, 15))  # เปลี่ยนรูปแบบ state ให้เป็น (1, 10, 15)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # การสำรวจ (Exploration)
        act_values = self.model.predict(state)  # ทำนาย Q-values
        return np.argmax(act_values[0])  # คืนค่าการกระทำที่มี Q-value สูงสุด

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Ensure state has the correct shape (1, 10, 15)
            state = np.reshape(state, (1, 10, 15))  # แก้ไขให้ตรงกับที่โมเดลต้องการ
            next_state = np.reshape(next_state, (1, 10, 15))  # แก้ไขให้ตรงกับที่โมเดลต้องการ

            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])  # Update Q-value for non-terminal state
            target_f = self.model.predict(state)  # Get current Q-values for the state
            target_f[0][action] = target  # Update the Q-value for the chosen action
            self.model.fit(state, target_f, epochs=1, verbose=0)  # Train the model with the updated target

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Decay epsilon to reduce exploration over time



# สร้าง sequences สำหรับ DQN
def create_sequences_for_dqn(features, targets, seq_length=10):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length, :])  # Ensure proper slicing
        y.append(targets[i+seq_length])  # Prediction will be the next closing price
    X = np.array(X)
    y = np.array(y)
    print("X shape (after sequence creation):", X.shape)  # Debugging: Check shape of X and y
    print("y shape (after sequence creation):", y.shape)
    return X, y  # Return X and y without np.array again as it's already an array.

# ฟังก์ชันสำหรับแสดงผลกราฟ
def plot_predictions(y_true, y_pred, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='red', alpha=0.7)
    plt.title(f'True vs Predicted Prices for {ticker}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# ตรวจสอบ GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logging.info(f"Memory growth enabled for GPU: {physical_devices[0]}")
        print("Memory growth enabled for GPU:", physical_devices[0])
    except Exception as e:
        logging.error(f"Failed to set memory growth: {e}")
        print(f"Error setting memory growth: {e}")
else:
    logging.info("GPU not found, using CPU")
    print("GPU not found, using CPU")

# โหลดข้อมูล
df_stock = pd.read_csv("cleaned_data.csv", parse_dates=["Date"]).sort_values(by=["Ticker", "Date"])
df_news = pd.read_csv("news_with_sentiment_gpu.csv")
df_news['Date'] = pd.to_datetime(df_news['Date'], errors='coerce')
df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')
print(df_news['Date'].dtype)
print(df_stock['Date'].dtype)
df_news['Sentiment'] = df_news['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
df_news['Confidence'] = df_news['Confidence']
df = pd.merge(df_stock, df_news[['Date', 'Sentiment', 'Confidence']], on='Date', how='left')

# เติมค่าที่ขาดหายไป
df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

# เพิ่มฟีเจอร์
df['Change'] = df['Close'] - df['Open']
df['Change (%)'] = (df['Change'] / df['Open']) * 100

df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
df['RSI'].fillna(method='ffill', inplace=True)  # เติมค่า NaN โดยใช้การเติมแบบ forward fill
df['RSI'].fillna(0, inplace=True)  # เติมค่า NaN ที่เหลือด้วย 0

df['SMA_5'] = df['Close'].rolling(window=5).mean()  # SMA 50 วัน
df['SMA_10'] = df['Close'].rolling(window=10).mean()  # SMA 200 วัน
# คำนวณ MACD ด้วย EMA 12 และ EMA 26
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
# คำนวณ MACD = EMA(12) - EMA(26)
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].rolling(window=9).mean()  

bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
df['Bollinger_High'] = bollinger.bollinger_hband()
df['Bollinger_Low'] = bollinger.bollinger_lband()

df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment', 'Confidence', 
                   'RSI', 'SMA_10', 'SMA_5', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low']

# Label Encode Ticker
ticker_encoder = LabelEncoder()
df['Ticker_ID'] = ticker_encoder.fit_transform(df['Ticker'])
num_tickers = len(ticker_encoder.classes_)

sorted_dates = df['Date'].unique()
train_cutoff = sorted_dates[int(len(sorted_dates) * 6 / 7)]  # ขอบเขตที่ 6 ปี

# ข้อมูล train, test
train_df = df[df['Date'] <= train_cutoff].copy()
test_df = df[df['Date'] > train_cutoff].copy()

scaler = RobustScaler()
numeric_columns_to_scale = ['Open', 'Close', 'High', 'Low', 'Volume']
df_stock[numeric_columns_to_scale] = scaler.fit_transform(df_stock[numeric_columns_to_scale])

# Feature Scaling
scaler = RobustScaler()
train_features = scaler.fit_transform(train_df[feature_columns])
test_features = scaler.transform(test_df[feature_columns])

train_labels = train_df['Close'].values
test_labels = test_df['Close'].values

# Create sequences for DQN
X_train, y_train = create_sequences_for_dqn(train_features, train_labels)
X_test, y_test = create_sequences_for_dqn(test_features, test_labels)

# Initialize DQN agent
state_size = X_train.shape[1]  # Features (number of columns)
action_size = 3  # Actions: Buy, Hold, Sell
agent = DQNAgent(state_size=state_size, action_size=action_size)

# Train model with early stopping and dynamic epsilon
epochs = 200
batch_size = 64
for epoch in range(epochs):
    state = X_train[0]  # Initial state
    total_reward = 0
    for time in range(len(X_train) - 1):
        action = agent.act(state)  # Select action (Buy/Hold/Sell)
        reward = 0
        if action == 0:  # Buy
            reward = y_train[time + 1] - y_train[time]  # Reward is price change
        elif action == 1:  # Hold
            reward = 0
        elif action == 2:  # Sell
            reward = y_train[time] - y_train[time + 1]  # Reward is negative of price change

        next_state = X_train[time + 1]
        done = time == len(X_train) - 2  # Done when reaching the last time step
        agent.remember(state, action, reward, next_state, done)  # Store experience
        agent.replay(batch_size)  # Update model from experience replay
        state = next_state  # Update state
        total_reward += reward  # Add reward for this step

    logging.info(f"Epoch {epoch + 1}/{epochs} - Total reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")
    
agent.model.save('best_price_model_full.keras')
agent.model.save('best_price_model_full.h5')

def predict_next_day_with_dqn(agent, test_df, feature_columns, seq_length=10):
    """
    Predict next day's action (Buy, Hold, Sell) and retrain DQN agent with actual reward using only test_df.
    """
    predictions = []
    actual_values = []
    total_reward = 0
    scaler_features = MinMaxScaler()

    # Sort data by date to ensure sequential prediction
    test_df = test_df.sort_values('Date')

    for i in range(len(test_df) - seq_length - 1):
        current_ticker = test_df.iloc[i]['Ticker']
        
        # Extract historical data up to the current point
        historical_data = test_df.iloc[i:i+seq_length]
        historical_data = historical_data[historical_data['Ticker'] == current_ticker]
        
         # Print progress
        if i % 100 == 0:  # Print every 100 iterations
            print(f"Processing: {i}/{len(test_df)-1}")

        if len(historical_data) < seq_length:
            continue

        # Prepare the state for the DQN agent
        features = historical_data[feature_columns].values
        state = scaler_features.fit_transform(features).flatten()  # Flatten for DQN input

        # Select action using the DQN agent
        action = agent.act(state.reshape(1, -1))  # Action: 0 = Buy, 1 = Hold, 2 = Sell
        current_price = test_df.iloc[i + seq_length]['Close']
        next_price = test_df.iloc[i + seq_length + 1]['Close']

        # Calculate reward
        if action == 0:  # Buy
            reward = next_price - current_price
        elif action == 2:  # Sell
            reward = current_price - next_price
        else:  # Hold
            reward = 0

        total_reward += reward

        # Prepare next state
        next_historical_data = test_df.iloc[i+1:i+1+seq_length]
        next_features = next_historical_data[feature_columns].values
        next_state = scaler_features.transform(next_features).flatten()

        done = (i == len(test_df) - seq_length - 2)  # Terminal state at the end of the data

        # Store experience in agent memory
        agent.remember(state.reshape(1, -1), action, reward, next_state.reshape(1, -1), done)

        # Retrain agent (experience replay)
        agent.replay(batch_size=32)

        # Save prediction and actual value
        predictions.append(action)  # Storing the action chosen (Buy/Hold/Sell)
        actual_values.append(next_price)

    print(f"Total Reward: {total_reward}")
    return predictions, actual_values


# Initialize the DQN agent (assuming feature_columns and seq_length are defined)
state_size = len(feature_columns) * 10  # Flattened state size
action_size = 3  # 0 = Buy, 1 = Hold, 2 = Sell
agent = DQNAgent(state_size, action_size)

# Execute the prediction with retraining using test_df only
predictions, actual_values = predict_next_day_with_dqn(
    agent,  # Use DQN agent
    test_df, 
    feature_columns
)
# Calculate metrics
mae = mean_absolute_error(actual_values, predictions)
mse = mean_squared_error(actual_values, predictions)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(actual_values, predictions)
r2 = r2_score(actual_values, predictions)

print("\nTest Metrics with Retraining:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"R2 Score: {r2:.4f}")

# Plot results (Actual vs Predicted)
plt.figure(figsize=(15, 6))
plt.plot(actual_values, label='Actual', color='blue')
plt.plot(predictions, label='Predicted', color='red', alpha=0.7)
plt.title('Next Day Predictions with Retraining')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Save the plot to a file
plt.savefig('next_day_predictions_with_retraining.png')

# Plot residuals (Actual - Predicted)
residuals = np.array(actual_values) - np.array(predictions)
plt.figure(figsize=(15, 6))
plt.scatter(range(len(residuals)), residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')  # Line at y = 0
plt.title('Prediction Residuals with Retraining')
plt.xlabel('Time')
plt.ylabel('Residual')
plt.show()

# Save residuals plot to a file
plt.savefig('prediction_residuals_with_retraining.png')

agent.model.save('price_prediction_DNQ_model_embedding_aftertest.keras')
