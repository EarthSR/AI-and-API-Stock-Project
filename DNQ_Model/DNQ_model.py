import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
from collections import deque
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import ta
import logging
import os
import psutil
import gc

# Configuration
CONFIG = {
    'BATCH_SIZE': 32,
    'EPOCHS': 50,
    'SEQUENCE_LENGTH': 10,
    'LEARNING_RATE': 0.0001,
    'GAMMA': 0.95,
    'EPSILON': 1.0,
    'EPSILON_DECAY': 0.997,
    'EPSILON_MIN': 0.01,
    'MEMORY_SIZE': 1000,
    'VALIDATION_SPLIT': 0.2
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    filename='training.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = CONFIG['LEARNING_RATE']
        self.gamma = CONFIG['GAMMA']
        self.epsilon = CONFIG['EPSILON']
        self.epsilon_decay = CONFIG['EPSILON_DECAY']
        self.epsilon_min = CONFIG['EPSILON_MIN']
        self.memory = deque(maxlen=CONFIG['MEMORY_SIZE'])
        self.model = self._build_model()
        logging.info(f"Model input shape: {self.model.input_shape}")
        
    def _build_model(self):
        model = Sequential([
            tf.keras.Input(shape=self.state_shape),
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model

    def act(self, state):
        state = np.reshape(state, (1,) + self.state_shape)  # (1, 10, 15)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state, verbose=0)  # (1, 3)
        action = np.argmax(act_values[0])
        
        return min(action, self.action_size - 1)

    def remember(self, state, action, reward, next_state, done):
        if state.shape == self.state_shape and next_state.shape == self.state_shape:
            self.memory.append((state, action, reward, next_state, done))
        else:
            logging.warning(f"Skipping storing state with invalid shapes: {state.shape}, {next_state.shape}")
            
    def replay(self, batch_size):
        logging.info("Starting replay method")
        if len(self.memory) < batch_size:
            logging.info("Not enough memories to replay")
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([m[0] for m in minibatch]).astype(np.float32)
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch]).astype(np.float32)
        dones = np.array([m[4] for m in minibatch])
        
        # ล็อกรูปร่างและชนิดข้อมูล
        logging.info(f"States shape: {states.shape}, dtype: {states.dtype}")
        logging.info(f"Next_states shape: {next_states.shape}, dtype: {next_states.dtype}")
        
        # ตรวจสอบ NaNs หรือ Infs
        if np.isnan(states).any() or np.isinf(states).any():
            logging.error("States contain NaN or Inf values.")
            return
        if np.isnan(next_states).any() or np.isinf(next_states).any():
            logging.error("Next_states contain NaN or Inf values.")
            return
        
        try:
            target = self.model.predict(states, verbose=0)
            target_next = self.model.predict(next_states, verbose=0)
            logging.info("Prediction successful")
        except ValueError as e:
            logging.error(f"Prediction error: {e}", exc_info=True)
            return
        
        indices = np.arange(batch_size)
        target[indices, actions] = rewards + (1 - dones) * self.gamma * np.max(target_next, axis=1)
        
        try:
            history = self.model.fit(states, target, epochs=1, verbose=0, batch_size=batch_size)
            loss = history.history['loss'][0]
            logging.info(f"Training loss: {loss}")
        except ValueError as e:
            logging.error(f"Fit error: {e}", exc_info=True)
            return
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        logging.info("Replay method completed")


def create_sequences(features, targets, seq_length=CONFIG['SEQUENCE_LENGTH']):
    X, y = [], []
    for i in range(len(features) - seq_length):
        seq = features[i:i+seq_length]
        if seq.shape != (seq_length, features.shape[1]):
            logging.warning(f"Invalid sequence shape at index {i}: {seq.shape}")
        X.append(seq)
        y.append(targets[i+seq_length])
    X = np.array(X)
    y = np.array(y)
    logging.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
    return X, y

def prepare_data():
    logging.info("Starting data preparation")
    
    # Load data
    df_stock = pd.read_csv("cleaned_data.csv", parse_dates=["Date"]).sort_values(by=["Ticker", "Date"])
    df_news = pd.read_csv("news_with_sentiment_gpu.csv")
    df_news['Date'] = pd.to_datetime(df_news['Date'], errors='coerce')
    df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')
    
    # Process news sentiment
    df_news['Sentiment'] = df_news['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
    
    # Label Encode Ticker
    ticker_encoder = LabelEncoder()
    df_stock['Ticker_ID'] = ticker_encoder.fit_transform(df_stock['Ticker'])
    
    # Merge stock and news data
    df = pd.merge(df_stock, df_news[['Date', 'Sentiment', 'Confidence']], on='Date', how='left')
    
    # Handle missing values efficiently
    df = df.ffill().bfill().fillna(0)
    
    # Replace infinities and handle
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    
    # Add technical indicators BEFORE scaling
    df = add_technical_indicators(df)
    
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 
        'Sentiment', 'Confidence', 'RSI', 'SMA_10', 'SMA_5', 
        'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low'
    ]
    
    # Split train/test based on date across all stocks
    sorted_dates = df['Date'].unique()
    train_cutoff = sorted_dates[int(len(sorted_dates) * 6 / 7)]
    
    train_mask = df['Date'] <= train_cutoff
    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()
    
    # Create targets
    train_targets = train_df.groupby('Ticker')['Close'].shift(-1).iloc[:-1]
    test_targets = test_df.groupby('Ticker')['Close'].shift(-1).iloc[:-1]
    
    # Prepare features
    train_features = train_df[feature_columns].iloc[:-1].values
    test_features = test_df[feature_columns].iloc[:-1].values
    
    # Scale features and targets AFTER creating technical indicators
    feature_scaler = RobustScaler()
    target_scaler = RobustScaler()
    
    train_features_scaled = feature_scaler.fit_transform(train_features)
    test_features_scaled = feature_scaler.transform(test_features)
    
    train_targets_scaled = target_scaler.fit_transform(train_targets.values.reshape(-1, 1))
    test_targets_scaled = target_scaler.transform(test_targets.values.reshape(-1, 1))
    
    joblib.dump(feature_scaler, 'feature_scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')
    
    logging.info("Data preparation completed")
    logging.info(f"train_features_scaled shape: {train_features_scaled.shape}")
    logging.info(f"test_features_scaled shape: {test_features_scaled.shape}")
    logging.info(f"train_targets_scaled shape: {train_targets_scaled.shape}")
    logging.info(f"test_targets_scaled shape: {test_targets_scaled.shape}")
    
    return (train_df, test_df, feature_columns, train_features_scaled, 
            test_features_scaled, train_targets_scaled, test_targets_scaled)

def add_technical_indicators(df):
    # Calculate basic indicators
    df['Change'] = df['Close'] - df['Open']
    df['Open_safe'] = df['Open'].replace(0, 1e-8)
    df['Change (%)'] = (df['Change'] / df['Open_safe']) * 100
    df.drop('Open_safe', axis=1, inplace=True)
    
    # Technical indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].rolling(window=9).mean()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    
    return df.ffill().bfill().fillna(0)

def calculate_reward(action, current_price, next_price):
    if action == 0:  # Buy
        return (next_price - current_price) / current_price
    elif action == 2:  # Sell
        return (current_price - next_price) / current_price
    return 0  # Hold

def train_model(X_train, y_train, agent):
    total_rewards = []

    # สร้าง EarlyStopping callback
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    # กำหนดโมเดลให้กับ early_stopping ด้วยตนเอง
    early_stopping.set_model(agent.model)
    
    early_stopping.on_train_begin()
    for epoch in range(CONFIG['EPOCHS']):
        if epoch % 5 == 0:
            process = psutil.Process(os.getpid())
            logging.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        logging.info(f"Starting Epoch {epoch + 1}/{CONFIG['EPOCHS']}")
        state = X_train[0]
        total_reward = 0
        
        for time in range(len(X_train) - 1):
            if time % 100 == 0:
                logging.info(f"Epoch {epoch + 1}: Processing time step {time}/{len(X_train) - 1}")
            
            action = agent.act(state)
            reward = calculate_reward(action, y_train[time], y_train[time + 1])
            next_state = X_train[time + 1]
            done = time == len(X_train) - 2
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay(CONFIG['BATCH_SIZE'])
            
            state = next_state
            total_reward += reward
            
            if time % 1000 == 0:
                logging.info(f"Epoch {epoch + 1}: Step {time}/{len(X_train) - 1}")
                gc.collect()
        
        total_rewards.append(total_reward)
        logging.info(f"Epoch {epoch + 1} - Total reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")
        
        # เรียกใช้ early_stopping.on_epoch_end ด้วย logs={'loss': total_reward}
        if early_stopping.on_epoch_end(epoch, logs={'loss': total_reward}):
            logging.info("Early stopping triggered")
            break

        if (epoch + 1) % 10 == 0:
            plot_training_progress(total_rewards)
            agent.model.save(f'model_checkpoint_epoch_{epoch+1}.keras')

    logging.info("Training completed successfully")
    return agent, total_rewards

def evaluate_model(agent, X_test, y_test):
    predictions = []
    actual_values = []
    
    for i in range(len(X_test)):
        state = X_test[i]
        action = agent.act(state)
        predictions.append(action)
        actual_values.append(y_test[i])
        
        if i % 1000 == 0:
            gc.collect()
    
    metrics = {
        'mae': mean_absolute_error(actual_values, predictions),
        'mse': mean_squared_error(actual_values, predictions),
        'rmse': np.sqrt(mean_squared_error(actual_values, predictions)),
        'r2': r2_score(actual_values, predictions)
    }
    
    logging.info("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        logging.info(f"{metric.upper()}: {value:.4f}")
    
    return predictions, actual_values, metrics

def plot_training_progress(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.savefig('training_progress.png')
    plt.close()

def predict_and_retrain_sequential(agent, test_df, feature_columns, scaler_features, scaler_target, initial_balance=100000, window_size=10):
    """
    ทำนายราคาและประเมินโมเดลแบบ sequential ด้วยข้อมูล test สำหรับหุ้นหนึ่งหุ้น
    
    Parameters:
    - agent: DQN agent ที่ถูกฝึกฝนแล้ว
    - test_df: DataFrame ของข้อมูลทดสอบสำหรับหุ้นหนึ่งหุ้น
    - feature_columns: รายชื่อ features ที่ใช้
    - scaler_features: RobustScaler ที่ใช้สเกลฟีเจอร์
    - scaler_target: RobustScaler ที่ใช้สเกลเป้าหมาย
    - initial_balance: เงินทุนเริ่มต้น
    - window_size: ขนาดของ sequence สำหรับการทำนาย
    """
    predictions = []
    actual_prices = []
    portfolio_values = [initial_balance]
    balance = initial_balance
    position = 0
    daily_returns = []
    transaction_fee = 0.001

    # เตรียมข้อมูลสำหรับการ evaluate
    trade_history = []
    performance_metrics = []

    # จัดเรียงข้อมูลตามวันที่
    test_df = test_df.sort_values('Date').reset_index(drop=True)

    for i in range(window_size, len(test_df)-1):
        try:
            # ดึงข้อมูลสำหรับ sequence ปัจจุบัน
            current_sequence = test_df.iloc[i-window_size:i]
            current_features = current_sequence[feature_columns].values

            # Log shape
            logging.info(f"Current sequence shape: {current_features.shape}")

            # Scale features
            scaled_features = scaler_features.transform(current_features)

            # ทำนายการกระทำ
            state = np.reshape(scaled_features, (1, window_size, len(feature_columns)))
            logging.info(f"State shape for prediction: {state.shape}")
            action = agent.act(state)

            # ราคาปัจจุบันและราคาถัดไป (แปลงกลับจากการสเกล)
            # สมมติว่า 'Close' เป็นคอลัมน์ที่ 4 ใน feature_columns
            # ดึงค่าปิดที่สเกลแล้วและทำ inverse_transform
            current_price_scaled = test_df.iloc[i]['Close']
            next_price_scaled = test_df.iloc[i+1]['Close']
            # ทำ inverse_transform เฉพาะราคาปิด
            current_price = scaler_target.inverse_transform([[current_price_scaled]])[0][0]
            next_price = scaler_target.inverse_transform([[next_price_scaled]])[0][0]

            # บันทึกการทำนาย
            predictions.append(action)
            actual_prices.append(next_price)

            # คำนวณผลตอบแทน
            trade_return = 0

            # ดำเนินการตามการทำนาย
            if action == 0:  # Buy
                if position == 0:
                    shares_to_buy = int(balance / (current_price * (1 + transaction_fee)))
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price * (1 + transaction_fee)
                        balance -= cost
                        position += shares_to_buy
                        trade_history.append({
                            'Date': test_df.iloc[i]['Date'],
                            'Action': 'Buy',
                            'Price': current_price,
                            'Shares': shares_to_buy,
                            'Balance': balance
                        })

            elif action == 2:  # Sell
                if position > 0:
                    sell_value = position * current_price * (1 - transaction_fee)
                    trade_return = sell_value - (position * current_price)
                    balance += sell_value
                    trade_history.append({
                        'Date': test_df.iloc[i]['Date'],
                        'Action': 'Sell',
                        'Price': current_price,
                        'Shares': position,
                        'Balance': balance,
                        'Return': trade_return
                    })
                    position = 0

            # คำนวณมูลค่าพอร์ตโฟลิโอ
            portfolio_value = balance + (position * current_price)
            portfolio_values.append(portfolio_value)
            daily_returns.append((portfolio_value - portfolio_values[-2]) / portfolio_values[-2])

            # Log progress
            if i % 10 == 0:
                logging.info(f"""
                Processing day {i}/{len(test_df)-1}
                Portfolio Value: {portfolio_value:.2f}
                Current Price: {current_price:.2f}
                Predicted Action: {['Buy', 'Hold', 'Sell'][action]}
                Position: {position}
                Balance: {balance:.2f}
                """)

        except Exception as e:
            logging.error(f"Error on day {i}: {str(e)}")
            continue

    # คำนวณ final metrics
    final_metrics = calculate_final_metrics(
        portfolio_values, 
        daily_returns, 
        predictions, 
        actual_prices,
        trade_history
    )

    # Plot results
    plot_results(portfolio_values, predictions, actual_prices, test_df['Date'].iloc[window_size:])

    return final_metrics, trade_history, performance_metrics

def process_multiple_stocks(agent, test_df, feature_columns, scaler_features, scaler_target, initial_balance=100000, window_size=10):
    """
    ทดสอบโมเดลเดียวกันสำหรับแต่ละหุ้นในชุดข้อมูลทดสอบ
    
    Parameters:
    - agent: DQN agent ที่ถูกฝึกฝนแล้ว
    - test_df: DataFrame ของข้อมูลทดสอบรวมหลายหุ้น
    - feature_columns: รายชื่อ features ที่ใช้
    - scaler_features: RobustScaler ที่ใช้สเกลฟีเจอร์
    - scaler_target: RobustScaler ที่ใช้สเกลเป้าหมาย
    - initial_balance: เงินทุนเริ่มต้น
    - window_size: ขนาดของ sequence สำหรับการทำนาย
    """
    results = {}
    trade_histories = {}
    performance_metrics_all = {}
    
    # สมมติว่ามีคอลัมน์ 'Ticker' ใน test_df
    tickers = test_df['Ticker'].unique()
    
    for ticker in tickers:
        logging.info(f"Testing ticker: {ticker}")
        
        # กรองข้อมูลสำหรับหุ้นตัวนี้
        stock_df = test_df[test_df['Ticker'] == ticker].copy()
        
        # ตรวจสอบว่ามีข้อมูลเพียงพอสำหรับ window_size
        if len(stock_df) < window_size + 1:
            logging.warning(f"Not enough data for ticker {ticker}. Skipping.")
            continue
        
        # โหลด scaler เฉพาะหุ้นนี้
        # เนื่องจากเราใช้ scaler เดียวกันสำหรับทุกหุ้นในการฝึกฝน
        # ดังนั้นไม่ต้องโหลด scaler แยก
        # scaler_features = joblib.load(f'scaler_features_{ticker}.pkl')
        # scaler_target = joblib.load(f'scaler_target_{ticker}.pkl')
        
        # เรียกใช้ฟังก์ชันทดสอบสำหรับหุ้นนี้
        metrics, trade_history, performance_metrics = predict_and_retrain_sequential(
            agent=agent,
            test_df=stock_df,
            feature_columns=feature_columns,
            scaler_features=scaler_features,
            scaler_target=scaler_target,
            initial_balance=initial_balance,
            window_size=window_size
        )
        
        # เก็บผลลัพธ์
        results[ticker] = metrics
        trade_histories[ticker] = trade_history
        performance_metrics_all[ticker] = performance_metrics
        
        logging.info(f"Completed testing for ticker: {ticker}")
    
    return results, trade_histories, performance_metrics_all



def calculate_final_metrics(portfolio_values, daily_returns, predictions, actual_prices, trade_history):
    """คำนวณ metrics สุดท้าย"""
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
    sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if len(daily_returns) > 0 else 0
    max_drawdown = calculate_max_drawdown(portfolio_values)
    
    # คำนวณ trading metrics
    total_trades = len([t for t in trade_history if t['Action'] in ['Buy', 'Sell']])
    profitable_trades = len([t for t in trade_history if t.get('Return', 0) > 0])
    win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
    
    return {
        'Total Return (%)': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown,
        'Total Trades': total_trades,
        'Win Rate (%)': win_rate,
        'Final Portfolio Value': portfolio_values[-1]
    }

def calculate_max_drawdown(portfolio_values):
    cumulative = np.maximum.accumulate(portfolio_values)
    drawdown = (cumulative - portfolio_values) / cumulative
    return np.max(drawdown) * 100

def plot_results(portfolio_values, predictions, actual_prices, dates):
    """สร้างกราฟแสดงผลลัพธ์"""
    # Plot portfolio value
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(dates, portfolio_values[1:], label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    
    # Plot predictions vs actual prices
    plt.subplot(2, 1, 2)
    actions = ['Buy', 'Hold', 'Sell']
    action_colors = ['green', 'gray', 'red']
    for action in [0, 1, 2]:
        idx = [i for i, a in enumerate(predictions) if a == action]
        plt.scatter(np.array(dates)[idx], np.array(actual_prices)[idx], c=action_colors[action], label=actions[action], alpha=0.5)
    plt.plot(dates, actual_prices, 'b-', label='Actual Prices', alpha=0.5)
    plt.title('Predicted Actions vs Actual Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('sequential_prediction_results.png')
    plt.close()

def calculate_max_drawdown(portfolio_values):
    cumulative = np.maximum.accumulate(portfolio_values)
    drawdown = (cumulative - portfolio_values) / cumulative
    return np.max(drawdown) * 100

def main():
    # Setup GPU
    logging.info("Starting program")
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            logging.info(f"Using GPU: {physical_devices[0]}")
        except Exception as e:
            logging.error(f"GPU setup failed: {e}")
    else:
        logging.info("Using CPU")
    
    # Prepare data
    train_df, test_df, feature_columns, train_features_scaled, test_features_scaled, \
    train_targets_scaled, test_targets_scaled = prepare_data()
    
    # Create sequences for training
    X_train, y_train = create_sequences(train_features_scaled, train_targets_scaled.flatten(), seq_length=CONFIG['SEQUENCE_LENGTH'])
    
    logging.info(f"X_train shape: {X_train.shape}")
    logging.info(f"y_train shape: {y_train.shape}")
    
    # Initialize agent
    state_shape = X_train.shape[1:]
    action_size = 3  # Buy, Hold, Sell
    logging.info(f"State shape for agent: {state_shape}")
    agent = DQNAgent(state_shape, action_size)
    
    try:
        # Train model
        agent, total_rewards = train_model(X_train, y_train, agent)
        
        # Save trained model
        agent.model.save('final_model_DQN.keras')
        
        # Plot training progress
        plot_training_progress(total_rewards)
        
        # Load scalers
        scaler_features = joblib.load('feature_scaler.pkl')
        scaler_target = joblib.load('target_scaler.pkl')
        
        # ทดสอบโมเดลสำหรับแต่ละหุ้น
        results, trade_histories, performance_metrics_all = process_multiple_stocks(
            agent=agent,
            test_df=test_df,
            feature_columns=feature_columns,
            scaler_features=scaler_features,
            scaler_target=scaler_target,
            initial_balance=100000,
            window_size=CONFIG['SEQUENCE_LENGTH']
        )
        
        # แสดงผลลัพธ์
        logging.info("\nFinal Results:")
        for ticker, metrics in results.items():
            logging.info(f"Metrics for {ticker}:")
            for key, value in metrics.items():
                logging.info(f"  {key}: {value}")
            logging.info("\n")
        
        # บันทึกผลลัพธ์ทั้งหมด
        pd.DataFrame(results).transpose().to_csv('final_metrics_all_stocks.csv')
        pd.concat({k: pd.DataFrame(v) for k, v in trade_histories.items()}, axis=0).to_csv('trade_histories_all_stocks.csv')
        pd.concat({k: pd.DataFrame(v) for k, v in performance_metrics_all.items()}, axis=0).to_csv('performance_metrics_all_stocks.csv')
        
    except Exception as e:
        logging.error(f"Error during training/testing: {str(e)}")
        raise
    
    finally:
        # Cleanup
        gc.collect()

if __name__ == "__main__":
    main()
