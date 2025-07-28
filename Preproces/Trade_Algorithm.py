import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TradingAlgorithm:
    def __init__(self, db_connection_string: str, initial_capital: float = 100000):
        """
        Initialize Trading Algorithm
        
        Args:
            db_connection_string: Database connection string
            initial_capital: Initial trading capital
        """
        self.db_connection = db_connection_string
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # {stock_symbol: {'quantity': int, 'avg_price': float}}
        self.transaction_history = []
        
        # Trading parameters
        self.max_position_size = 0.1  # Maximum 10% of capital per position
        self.stop_loss_pct = 0.05     # 5% stop loss
        self.take_profit_pct = 0.15   # 15% take profit
        self.min_confidence_threshold = 0.6  # Minimum confidence for trading
        
    def get_prediction_data(self) -> pd.DataFrame:
        """
        Fetch prediction data from database
        """
        query = """
        SELECT 
            StockSymbol,
            StockDetail.PredictionTrend_Ensemble,
            CurrentPrice,
            Volume,
            Timestamp
        FROM StockDetail
        WHERE PredictionTrend_Ensemble IS NOT NULL
        ORDER BY Timestamp DESC
        """
        
        try:
            conn = sqlite3.connect(self.db_connection)
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def interpret_prediction(self, prediction_value) -> Dict:
        """
        Interpret prediction trend ensemble value
        
        Args:
            prediction_value: The ensemble prediction value
            
        Returns:
            Dict with signal, confidence, and action
        """
        if isinstance(prediction_value, str):
            prediction_value = prediction_value.lower()
            
            if 'bullish' in prediction_value or 'buy' in prediction_value:
                return {'signal': 'BUY', 'confidence': 0.8, 'action': 'LONG'}
            elif 'bearish' in prediction_value or 'sell' in prediction_value:
                return {'signal': 'SELL', 'confidence': 0.8, 'action': 'SHORT'}
            elif 'neutral' in prediction_value or 'hold' in prediction_value:
                return {'signal': 'HOLD', 'confidence': 0.5, 'action': 'HOLD'}
        
        elif isinstance(prediction_value, (int, float)):
            # Assume numeric values: > 0.6 = BUY, < 0.4 = SELL, else HOLD
            if prediction_value > 0.6:
                confidence = min(prediction_value, 1.0)
                return {'signal': 'BUY', 'confidence': confidence, 'action': 'LONG'}
            elif prediction_value < 0.4:
                confidence = min(1 - prediction_value, 1.0)
                return {'signal': 'SELL', 'confidence': confidence, 'action': 'SHORT'}
            else:
                return {'signal': 'HOLD', 'confidence': 0.5, 'action': 'HOLD'}
        
        return {'signal': 'HOLD', 'confidence': 0.0, 'action': 'HOLD'}
    
    def calculate_position_size(self, stock_symbol: str, current_price: float, 
                              confidence: float) -> int:
        """
        Calculate optimal position size based on capital and risk management
        """
        # Base position size as percentage of capital
        base_position_value = self.current_capital * self.max_position_size
        
        # Adjust based on confidence
        adjusted_position_value = base_position_value * confidence
        
        # Calculate number of shares
        shares = int(adjusted_position_value / current_price)
        
        return max(shares, 0)
    
    def check_risk_management(self, stock_symbol: str, current_price: float) -> str:
        """
        Check if position needs to be closed due to stop loss or take profit
        """
        if stock_symbol not in self.positions:
            return 'HOLD'
        
        position = self.positions[stock_symbol]
        avg_price = position['avg_price']
        quantity = position['quantity']
        
        if quantity > 0:  # Long position
            pnl_pct = (current_price - avg_price) / avg_price
            
            if pnl_pct <= -self.stop_loss_pct:
                return 'STOP_LOSS'
            elif pnl_pct >= self.take_profit_pct:
                return 'TAKE_PROFIT'
        
        elif quantity < 0:  # Short position
            pnl_pct = (avg_price - current_price) / avg_price
            
            if pnl_pct <= -self.stop_loss_pct:
                return 'STOP_LOSS'
            elif pnl_pct >= self.take_profit_pct:
                return 'TAKE_PROFIT'
        
        return 'HOLD'
    
    def execute_trade(self, stock_symbol: str, action: str, quantity: int, 
                     price: float, reason: str = ''):
        """
        Execute a trade and update positions
        """
        timestamp = datetime.now()
        
        # Calculate trade value
        trade_value = quantity * price
        
        if action == 'BUY':
            # Check if we have enough capital
            if trade_value > self.current_capital:
                print(f"Insufficient capital for {stock_symbol}: Need {trade_value}, Have {self.current_capital}")
                return False
            
            # Update capital
            self.current_capital -= trade_value
            
            # Update position
            if stock_symbol in self.positions:
                old_qty = self.positions[stock_symbol]['quantity']
                old_avg = self.positions[stock_symbol]['avg_price']
                new_qty = old_qty + quantity
                new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
                self.positions[stock_symbol] = {'quantity': new_qty, 'avg_price': new_avg}
            else:
                self.positions[stock_symbol] = {'quantity': quantity, 'avg_price': price}
        
        elif action == 'SELL':
            # Update capital
            self.current_capital += trade_value
            
            # Update position
            if stock_symbol in self.positions:
                self.positions[stock_symbol]['quantity'] -= quantity
                if self.positions[stock_symbol]['quantity'] == 0:
                    del self.positions[stock_symbol]
        
        # Record transaction
        transaction = {
            'timestamp': timestamp,
            'symbol': stock_symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'value': trade_value,
            'reason': reason,
            'capital_after': self.current_capital
        }
        
        self.transaction_history.append(transaction)
        
        print(f"{timestamp}: {action} {quantity} shares of {stock_symbol} at {price:.2f} - Reason: {reason}")
        return True
    
    def run_algorithm(self) -> Dict:
        """
        Main algorithm execution
        """
        print("Starting Trading Algorithm...")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        
        # Get prediction data
        df = self.get_prediction_data()
        
        if df.empty:
            print("No prediction data available")
            return self.get_portfolio_summary()
        
        print(f"Processing {len(df)} stocks with predictions...")
        
        for _, row in df.iterrows():
            stock_symbol = row['StockSymbol']
            prediction = row['PredictionTrend_Ensemble']
            current_price = row['CurrentPrice']
            
            if pd.isna(current_price) or current_price <= 0:
                continue
            
            # Interpret prediction
            prediction_info = self.interpret_prediction(prediction)
            signal = prediction_info['signal']
            confidence = prediction_info['confidence']
            action = prediction_info['action']
            
            # Skip if confidence is too low
            if confidence < self.min_confidence_threshold:
                continue
            
            # Check risk management first
            risk_action = self.check_risk_management(stock_symbol, current_price)
            
            if risk_action in ['STOP_LOSS', 'TAKE_PROFIT']:
                # Close position
                if stock_symbol in self.positions:
                    quantity = abs(self.positions[stock_symbol]['quantity'])
                    if quantity > 0:
                        self.execute_trade(stock_symbol, 'SELL', quantity, 
                                         current_price, risk_action)
                continue
            
            # Execute trading logic based on signal
            if signal == 'BUY' and action == 'LONG':
                # Calculate position size
                position_size = self.calculate_position_size(stock_symbol, current_price, confidence)
                
                if position_size > 0:
                    self.execute_trade(stock_symbol, 'BUY', position_size, 
                                     current_price, f'Prediction: {prediction}')
            
            elif signal == 'SELL':
                # Close long position if exists
                if stock_symbol in self.positions and self.positions[stock_symbol]['quantity'] > 0:
                    quantity = self.positions[stock_symbol]['quantity']
                    self.execute_trade(stock_symbol, 'SELL', quantity, 
                                     current_price, f'Prediction: {prediction}')
        
        return self.get_portfolio_summary()
    
    def get_portfolio_summary(self) -> Dict:
        """
        Generate portfolio summary
        """
        total_position_value = 0
        
        # Calculate current position values (would need current prices)
        for symbol, position in self.positions.items():
            # For demo purposes, assume current price = avg price
            position_value = position['quantity'] * position['avg_price']
            total_position_value += position_value
        
        total_portfolio_value = self.current_capital + total_position_value
        total_return = ((total_portfolio_value - self.initial_capital) / self.initial_capital) * 100
        
        summary = {
            'initial_capital': self.initial_capital,
            'current_cash': self.current_capital,
            'position_value': total_position_value,
            'total_portfolio_value': total_portfolio_value,
            'total_return_pct': total_return,
            'number_of_positions': len(self.positions),
            'number_of_transactions': len(self.transaction_history),
            'positions': self.positions.copy()
        }
        
        return summary
    
    def print_summary(self):
        """
        Print portfolio summary
        """
        summary = self.get_portfolio_summary()
        
        print("\n" + "="*50)
        print("PORTFOLIO SUMMARY")
        print("="*50)
        print(f"Initial Capital: ${summary['initial_capital']:,.2f}")
        print(f"Current Cash: ${summary['current_cash']:,.2f}")
        print(f"Position Value: ${summary['position_value']:,.2f}")
        print(f"Total Portfolio Value: ${summary['total_portfolio_value']:,.2f}")
        print(f"Total Return: {summary['total_return_pct']:.2f}%")
        print(f"Number of Positions: {summary['number_of_positions']}")
        print(f"Number of Transactions: {summary['number_of_transactions']}")
        
        if summary['positions']:
            print("\nCURRENT POSITIONS:")
            for symbol, position in summary['positions'].items():
                print(f"  {symbol}: {position['quantity']} shares @ ${position['avg_price']:.2f}")
    
    def export_transactions(self, filename: str = 'trading_history.csv'):
        """
        Export transaction history to CSV
        """
        if self.transaction_history:
            df = pd.DataFrame(self.transaction_history)
            df.to_csv(filename, index=False)
            print(f"Transaction history exported to {filename}")

# Example usage
def main():
    # Initialize algorithm
    algorithm = TradingAlgorithm(
        db_connection_string='your_database.db',  # Replace with your database
        initial_capital=100000
    )
    
    # Run the algorithm
    summary = algorithm.run_algorithm()
    
    # Print results
    algorithm.print_summary()
    
    # Export transaction history
    algorithm.export_transactions()

if __name__ == "__main__":
    main()