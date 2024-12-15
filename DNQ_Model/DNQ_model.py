import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from collections import deque
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import ta
import logging

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
        model = Sequential()
        model.add(Dense(64, input_shape=(self.state_size,), activation='relu'))  # แก้ไขตรงนี้
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # Output layer: Q-values for each action
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):  
        state = np.reshape(state, (1, self.state_size))  # แปลงให้เป็น (1, state_size)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action (exploration)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# สร้าง sequences สำหรับ DQN
def create_sequences_for_dqn(features, targets, seq_length=10):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(targets[i+seq_length])  # Prediction will be the next closing price
    return np.array(X), np.array(y)

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
# ตรวจสอบ GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    # Set visible devices to the first GPU (or any other specific one)
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    # Enable memory growth for the first GPU
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
df_news['Sentiment'] = df_news['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
df_news['Confidence'] = df_news['Confidence'] / 100
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

df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
df['Bollinger_High'] = bollinger.bollinger_hband()
df['Bollinger_Low'] = bollinger.bollinger_lband()

df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment', 'Confidence', 
                   'RSI', 'SMA_10', 'SMA_200', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low']

# Label Encode Ticker
ticker_encoder = LabelEncoder()
df['Ticker_ID'] = ticker_encoder.fit_transform(df['Ticker'])
num_tickers = len(ticker_encoder.classes_)

# แบ่งข้อมูล Train/Val/Test ตามเวลา
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

sorted_dates = df['Date'].sort_values().unique()
train_cutoff = sorted_dates[int(len(sorted_dates)*train_ratio)]
val_cutoff = sorted_dates[int(len(sorted_dates)*(train_ratio+val_ratio))]

train_df = df[df['Date'] <= train_cutoff].copy()
val_df = df[(df['Date'] > train_cutoff) & (df['Date'] <= val_cutoff)].copy()
test_df = df[df['Date'] > val_cutoff].copy()

train_targets_price = train_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
train_df = train_df.iloc[:-1]

val_targets_price = val_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
val_df = val_df.iloc[:-1]

test_targets_price = test_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
test_df = test_df.iloc[:-1]

train_features = train_df[feature_columns].values
val_features = val_df[feature_columns].values
test_features = test_df[feature_columns].values

train_ticker_id = train_df['Ticker_ID'].values
val_ticker_id = val_df['Ticker_ID'].values
test_ticker_id = test_df['Ticker_ID'].values

# สเกลข้อมูลจากเทรนเท่านั้น
scaler_features = MinMaxScaler()
train_features_scaled = scaler_features.fit_transform(train_features)
val_features_scaled = scaler_features.transform(val_features)
test_features_scaled = scaler_features.transform(test_features)

scaler_target = MinMaxScaler()
train_targets_scaled = scaler_target.fit_transform(train_targets_price)
val_targets_scaled = scaler_target.transform(val_targets_price)
test_targets_scaled = scaler_target.transform(test_targets_price)

joblib.dump(scaler_features, 'scaler_features_full.pkl')  # บันทึก scaler ฟีเจอร์
joblib.dump(scaler_target, 'scaler_target_full.pkl')     # บันทึก scaler เป้าหมาย

# สร้าง sequences สำหรับ DQN
seq_length = 10
X_train, y_train = create_sequences_for_dqn(train_features_scaled, train_targets_scaled, seq_length)
X_val, y_val = create_sequences_for_dqn(val_features_scaled, val_targets_scaled, seq_length)
X_test, y_test = create_sequences_for_dqn(test_features_scaled, test_targets_scaled, seq_length)

# สร้างและฝึกโมเดล DQN
state_size = len(feature_columns) * seq_length  # ขนาดของ state
action_size = 3  # สามารถใช้ "Buy", "Hold", "Sell"
agent = DQNAgent(state_size, action_size)

# ฝึก agent
for e in range(30):  # จำนวนรอบในการฝึก
    state = X_train[0]  # เริ่มต้นที่ข้อมูลตัวแรก
    state = np.reshape(state, [1, state_size])
    
    for time in range(len(X_train) - 1):
        action = agent.act(state)  # เลือก action
        next_state = X_train[time + 1]  # สถานะถัดไป
        next_state = np.reshape(next_state, [1, state_size])

        # คำนวณ reward (อาจจะมาจากการเปรียบเทียบราคาหุ้น)
        reward = 0
        if action == 0:  # ซื้อ
            reward = next_state[0][3] - state[0][3]
        elif action == 1:  # ขาย
            reward = state[0][3] - next_state[0][3]
        else:  # ถือ
            reward = 0

        done = time == len(X_train) - 2  # จบเมื่อถึงข้อมูลสุดท้าย
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            logging.info(f"Episode {e}/{1000} finished")
            break

    # Replay สำหรับประสบการณ์
    agent.replay(batch_size=32)

# ทำนายผล
y_pred_scaled = agent.model.predict(X_test)
y_pred = scaler_target.inverse_transform(y_pred_scaled)

# แสดงกราฟ
plot_predictions(y_test, y_pred, "Stock Prediction")

# ผลการทำนาย
mse = np.mean((y_test - y_pred) ** 2)
logging.info(f'Mean Squared Error: {mse}')
