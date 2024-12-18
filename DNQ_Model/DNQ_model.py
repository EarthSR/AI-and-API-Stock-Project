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

# Configuration
CONFIG = {
    'BATCH_SIZE': 64,
    'EPOCHS': 100,
    'SEQUENCE_LENGTH': 10,
    'LEARNING_RATE': 0.001,
    'GAMMA': 0.95,
    'EPSILON': 1.0,
    'EPSILON_DECAY': 0.995,
    'EPSILON_MIN': 0.01,
    'MEMORY_SIZE': 2000,
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
        self.state_shape = state_shape  # e.g., (10, 15)
        self.action_size = action_size
        self.learning_rate = CONFIG['LEARNING_RATE']
        self.gamma = CONFIG['GAMMA']
        self.epsilon = CONFIG['EPSILON']
        self.epsilon_decay = CONFIG['EPSILON_DECAY']
        self.epsilon_min = CONFIG['EPSILON_MIN']
        self.memory = deque(maxlen=CONFIG['MEMORY_SIZE'])
        self.model = self._build_model()
        
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
        state = np.reshape(state, (1,) + self.state_shape)
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            act_values = self.model.predict(state, verbose=0)
            action = np.argmax(act_values[0])
            
        if action >= self.action_size:
            logging.warning(f"Invalid action: {action}, resetting to valid range.")
            action = random.randrange(self.action_size)
        
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])
        
        # Predict Q-values for current states and next states
        target = self.model.predict(states, verbose=0)
        target_next = self.model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])
        
        # Train the model
        self.model.fit(states, target, epochs=1, verbose=0, batch_size=batch_size)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def clear_memory(self):
        self.memory.clear()

def create_sequences_for_dqn(features, targets, seq_length=CONFIG['SEQUENCE_LENGTH']):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length, :])
        y.append(targets[i+seq_length])
    return np.array(X), np.array(y)

def prepare_data():
    # Load data
    df_stock = pd.read_csv("cleaned_data.csv", parse_dates=["Date"]).sort_values(by=["Ticker", "Date"])
    df_news = pd.read_csv("news_with_sentiment_gpu.csv")
    df_news['Date'] = pd.to_datetime(df_news['Date'], errors='coerce')
    df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')
    
    # ตรวจสอบชนิดข้อมูลวันที่
    print(df_news['Date'].dtype)
    print(df_stock['Date'].dtype)
    
    # Process news sentiment
    df_news['Sentiment'] = df_news['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
    
    # Label Encode Ticker
    ticker_encoder = LabelEncoder()
    df_stock['Ticker_ID'] = ticker_encoder.fit_transform(df_stock['Ticker'])
    num_tickers = len(ticker_encoder.classes_)
    
    # Merge stock and news data
    df = pd.merge(df_stock, df_news[['Date', 'Sentiment', 'Confidence']], on='Date', how='left')
    
    # Fill missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    # แทนที่ค่า Inf ด้วย NaN และเติม NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    # Scale numerical features
    scaler = RobustScaler()
    numeric_columns = ['Open', 'Close', 'High', 'Low', 'Volume']
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    joblib.dump(scaler, 'price_scaler.pkl')
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # แทนที่ค่า Inf และ NaN อีกครั้งหลังการเพิ่มฟีเจอร์
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    # Prepare features
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 
        'Sentiment', 'Confidence', 'RSI', 'SMA_10', 'SMA_5', 
        'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low'
    ]
    
    # Fill any remaining missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    # Split train/test
    sorted_dates = df['Date'].unique()
    train_cutoff = sorted_dates[int(len(sorted_dates) * 6 / 7)]
    
    train_df = df[df['Date'] <= train_cutoff].copy()
    test_df = df[df['Date'] > train_cutoff].copy()
    
    print("Train cutoff:", train_cutoff)
    print("First date in train set:", train_df['Date'].min())
    print("Last date in train set:", train_df['Date'].max())
    
    # สร้าง target โดย shift(-1)
    train_targets_price = train_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
    train_df = train_df.iloc[:-1]

    test_targets_price = test_df['Close'].shift(-1).dropna().values.reshape(-1, 1)
    test_df = test_df.iloc[:-1]

    train_features = train_df[feature_columns].values
    test_features = test_df[feature_columns].values

    train_ticker_id = train_df['Ticker_ID'].values
    test_ticker_id = test_df['Ticker_ID'].values

    # สเกลข้อมูลด้วย RobustScaler
    # สเกลข้อมูลจากชุดฝึก (train) เท่านั้น
    scaler_features = RobustScaler()
    train_features_scaled = scaler_features.fit_transform(train_features)  # ใช้ fit_transform กับชุดฝึก
    test_features_scaled = scaler_features.transform(test_features)  # ใช้ transform กับชุดทดสอบ

    scaler_target = RobustScaler()
    train_targets_scaled = scaler_target.fit_transform(train_targets_price)
    test_targets_scaled = scaler_target.transform(test_targets_price)

    joblib.dump(scaler_features, 'scaler_features.pkl')  # บันทึก scaler ฟีเจอร์
    joblib.dump(scaler_target, 'scaler_target.pkl')     # บันทึก scaler เป้าหมาย
    
    
    return train_df, test_df, feature_columns, train_features_scaled, test_features_scaled, train_targets_scaled, test_targets_scaled

def add_technical_indicators(df):
    df['Change'] = df['Close'] - df['Open']
    # ป้องกันการหารด้วยศูนย์โดยแทนค่า Open ที่เป็นศูนย์ด้วยค่าเล็กน้อย (เช่น 1e-8)
    df['Open_safe'] = df['Open'].replace(0, 1e-8)
    df['Change (%)'] = (df['Change'] / df['Open_safe']) * 100
    df.drop('Open_safe', axis=1, inplace=True)
    
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    # Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()  # SMA 5 วัน
    df['SMA_10'] = df['Close'].rolling(window=10).mean()  # SMA 10 วัน
    
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].rolling(window=9).mean()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df


def calculate_reward(action, current_price, next_price):
    if action == 0:  # Buy
        return next_price - current_price
    elif action == 2:  # Sell
        return current_price - next_price
    return 0  # Hold

def plot_training_progress(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.savefig('training_progress.png')
    plt.close()

def train_model(X_train, y_train, agent):
    total_rewards = []
    for epoch in range(CONFIG['EPOCHS']):
        logging.info(f"เริ่มต้น Epoch {epoch + 1}/{CONFIG['EPOCHS']}")
        state = X_train[0]
        total_reward = 0
        
        for time in range(len(X_train) - 1):
            if time % 1000 == 0:
                logging.info(f"Epoch {epoch + 1}: Processing time step {time}/{len(X_train) - 1}")
                
            action = agent.act(state)
            reward = calculate_reward(action, y_train[time], y_train[time + 1])
            next_state = X_train[time + 1]
            done = time == len(X_train) - 2
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay(CONFIG['BATCH_SIZE'])
            
            state = next_state
            total_reward += reward
            
        total_rewards.append(total_reward)
        logging.info(f"Epoch {epoch + 1}/{CONFIG['EPOCHS']} - Total reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")
        
        if (epoch + 1) % 10 == 0:
            plot_training_progress(total_rewards)
    
    return agent


def evaluate_model(agent, X_test, y_test, scaler_target):
    predictions = []
    actual_values = []
    
    for i in range(len(X_test)):
        state = X_test[i]
        action = agent.act(state)
        predictions.append(action)
        actual_values.append(y_test[i])
    
    # เนื่องจาก actions เป็น discrete actions (Buy, Hold, Sell) การคำนวณ MAE, MSE อาจไม่เหมาะสม
    # แนะนำให้ประเมินด้วยการวัดผลการเทรด เช่น ผลตอบแทน, Sharpe Ratio, ฯลฯ
    
    # อย่างไรก็ตาม หากต้องการคำนวณ metrics แบบเดิม:
    mae = mean_absolute_error(actual_values, predictions)
    mse = mean_squared_error(actual_values, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_values, predictions)
    
    logging.info(f"""
    Evaluation Metrics:
    MAE: {mae:.4f}
    MSE: {mse:.4f}
    RMSE: {rmse:.4f}
    R2 Score: {r2:.4f}
    """)
    
    return predictions, actual_values

def calculate_max_drawdown(portfolio_values):
    cumulative = np.maximum.accumulate(portfolio_values)
    drawdown = (cumulative - portfolio_values) / cumulative
    return np.max(drawdown) * 100

def predict_and_retrain_sequential(agent, test_df, feature_columns, scaler_features, scaler_target, initial_balance=100000, window_size=10):
    """
    ทำนายราคาและ retrain โมเดลแบบ sequential ด้วยข้อมูล test
    
    Parameters:
    - agent: DQN agent
    - test_df: DataFrame ของข้อมูลทดสอบ
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
            
            # Scale features
            scaled_features = scaler_features.transform(current_features)
            
            # ทำนายการกระทำ
            state = np.reshape(scaled_features, (1, window_size, len(feature_columns)))
            action = agent.act(state)
            
            # ราคาปัจจุบันและราคาถัดไป
            current_price = test_df.iloc[i]['Close']
            next_price = test_df.iloc[i+1]['Close']
            
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
            
            # Retrain model with new data
            if i % 1 == 0:  # Retrain ทุก 1 วัน
                # สร้าง sequence ใหม่สำหรับ training
                train_sequence = test_df.iloc[max(0, i-10):i]  # ใช้ข้อมูล 10 วันล่าสุด
                train_features_new = train_sequence[feature_columns].values
                scaled_train_features_new = scaler_features.transform(train_features_new)
                
                # สร้าง sequences สำหรับ training
                if len(scaled_train_features_new) >= window_size + 1:
                    X_train_new, y_train_new = create_sequences_for_dqn(
                        scaled_train_features_new,
                        train_sequence['Close'].values,
                        seq_length=window_size
                    )
                    
                    # เพิ่มข้อมูลใหม่ลงใน memory
                    for j in range(len(X_train_new)):
                        state_new = X_train_new[j]
                        # สมมติการกระทำแบบสุ่มสำหรับการฝึกเพิ่มเติม (สามารถปรับปรุงได้)
                        action_new = random.randrange(agent.action_size)
                        reward_new = calculate_reward(action_new, y_train_new[j], y_train_new[j])
                        next_state_new = X_train_new[j]  # Placeholder
                        done_new = False  # Placeholder
                        agent.remember(state_new, action_new, reward_new, next_state_new, done_new)
                    
                    # Retrain model
                    agent.replay(min(CONFIG['BATCH_SIZE'], len(agent.memory)))
                    
                    # บันทึก metrics หลัง retrain
                    current_metrics = {
                        'Date': test_df.iloc[i]['Date'],
                        'Portfolio Value': portfolio_value,
                        'Return': daily_returns[-1] if daily_returns else 0,
                        'Epsilon': agent.epsilon
                    }
                    performance_metrics.append(current_metrics)
            
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

def main():
    # Setup GPU
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
    train_df, test_df, feature_columns, train_features_scaled, test_features_scaled, train_targets_scaled, test_targets_scaled = prepare_data()
    
    # Create sequences
    X_train, y_train = create_sequences_for_dqn(
        train_features_scaled,
        train_targets_scaled.flatten(),
        seq_length=CONFIG['SEQUENCE_LENGTH']
    )
    X_test, y_test = create_sequences_for_dqn(
        test_features_scaled,
        test_targets_scaled.flatten(),
        seq_length=CONFIG['SEQUENCE_LENGTH']
    )
    
    # Define state shape
    state_shape = X_train.shape[1:]  # (sequence_length, num_features)
    
    # Initialize agent
    action_size = 3  # Buy, Hold, Sell
    agent = DQNAgent(state_shape, action_size)
    
    # Train model
    agent = train_model(X_train, y_train, agent)
    
    # Evaluate model
    predictions, actual_values = evaluate_model(agent, X_test, y_test, scaler_target=None)  # scaler_target ไม่ได้ใช้ในการคำนวณ
    
    # Save final model
    agent.model.save('final_model_DQN.h5')
    
    # Plot evaluation results
    plt.figure(figsize=(15, 6))
    plt.plot(actual_values, label='Actual', color='blue')
    plt.plot(predictions, label='Predicted Actions', color='red', alpha=0.7)
    plt.title('Trading Decisions vs Actual Price Movement')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.savefig('final_results.png')
    plt.close()
    
    # Predict and retrain sequentially
    metrics, trade_history, performance_metrics = predict_and_retrain_sequential(
        agent,
        test_df,
        feature_columns,
        scaler_features,
        scaler_target
    )
    
    # แสดงผลลัพธ์
    logging.info("\nFinal Results:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.2f}")
    
    # บันทึกผลลัพธ์
    pd.DataFrame(trade_history).to_csv('trade_history.csv', index=False)
    pd.DataFrame(performance_metrics).to_csv('performance_metrics.csv', index=False)

if __name__ == "__main__":
    main()
