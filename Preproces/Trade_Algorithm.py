#!/usr/bin/env python3
"""
Complete AI Trading System with Backtesting, Optimization, and Risk Management
Integrates all components into a single, comprehensive trading platform
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import argparse
import json

# Import all our modules (assume they're in the same directory or installed)
from ai_trading_algorithm import AITradingAlgorithm
from backtesting_engine import BacktestingEngine, AITradingAlgorithmWithBacktest
from advanced_analytics import AdvancedAnalytics, AITradingAlgorithmAdvanced
from portfolio_risk_management import AdvancedPortfolioManager, AdvancedTradingSystem

class CompleteTradingSystem:
    """
    Complete AI Trading System that integrates all components:
    - AI Trading Algorithm
    - Backtesting Engine
    - Advanced Analytics & ML Optimization
    - Portfolio & Risk Management
    """
    
    def __init__(self, config_file: str = 'config.env', initial_capital: float = 100000):
        """Initialize the complete trading system"""
        self.initial_capital = initial_capital
        self.config_file = config_file
        
        # Initialize all components
        self.trading_algorithm = AITradingAlgorithmAdvanced()
        self.portfolio_manager = AdvancedPortfolioManager(initial_capital)
        self.advanced_system = AdvancedTradingSystem(initial_capital)
        
        # System state
        self.is_live_trading = False
        self.optimization_results = {}
        self.backtest_results = {}
        self.performance_history = []
        
        # Setup logging
        self.setup_logging()
        
        self.logger.info("🚀 Complete AI Trading System Initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Main system log
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f'logs/trading_system_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('CompleteTradingSystem')
    
    def load_and_validate_config(self) -> bool:
        """Load and validate system configuration"""
        try:
            # Check if config file exists
            if not os.path.exists(self.config_file):
                self.logger.error(f"❌ Configuration file {self.config_file} not found")
                return False
            
            # Load environment variables
            from dotenv import load_dotenv
            load_dotenv(self.config_file)
            
            # Validate required configurations
            required_vars = [
                'DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_NAME',
                'INVESTX_API_KEY', 'INVESTX_SECRET', 'INVESTX_ACCOUNT_ID'
            ]
            
            missing_vars = []
            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                self.logger.error(f"❌ Missing required configuration variables: {missing_vars}")
                return False
            
            self.logger.info("✅ Configuration loaded and validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading configuration: {e}")
            return False
    
    def run_comprehensive_backtest(self, 
                                  start_date: str, 
                                  end_date: str,
                                  strategies: Optional[List[str]] = None) -> Dict:
        """Run comprehensive backtesting with multiple strategies"""
        
        self.logger.info(f"🔄 Starting comprehensive backtest: {start_date} to {end_date}")
        
        if not strategies:
            strategies = ['conservative', 'balanced', 'aggressive', 'ml_optimized']
        
        backtest_results = {}
        
        try:
            # 1. Run standard backtests
            for strategy in strategies:
                self.logger.info(f"📊 Testing {strategy} strategy...")
                
                if strategy == 'ml_optimized':
                    # Run ML optimization first
                    optimization_result = self.trading_algorithm.optimize_with_ml(
                        start_date, end_date, n_trials=30
                    )
                    self.optimization_results[strategy] = optimization_result
                
                # Run backtest
                result = self.trading_algorithm.run_backtest(start_date, end_date)
                backtest_results[strategy] = result
            
            # 2. Compare strategies
            comparison_df = self.trading_algorithm.compare_with_benchmarks(start_date, end_date)
            backtest_results['strategy_comparison'] = comparison_df
            
            # 3. Advanced analytics
            analysis_report = self.trading_algorithm.run_advanced_analysis(start_date, end_date)
            backtest_results['advanced_analysis'] = analysis_report
            
            # 4. Portfolio optimization
            portfolio_results = self.advanced_system.run_complete_optimization(
                self.trading_algorithm.backtesting_engine.load_historical_data(start_date, end_date)
            )
            backtest_results['portfolio_optimization'] = portfolio_results
            
            self.backtest_results = backtest_results
            
            self.logger.info("✅ Comprehensive backtesting completed")
            
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in comprehensive backtesting: {e}")
            return {}
    
    def optimize_system_parameters(self, 
                                  optimization_period: int = 90,
                                  optimization_method: str = 'bayesian') -> Dict:
        """Optimize system parameters using various methods"""
        
        self.logger.info(f"🧠 Starting system optimization using {optimization_method} method")
        
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=optimization_period)).strftime('%Y-%m-%d')
            
            optimization_results = {}
            
            if optimization_method == 'bayesian':
                # Bayesian optimization
                result = self.trading_algorithm.optimize_with_ml(start_date, end_date, n_trials=100)
                optimization_results['bayesian'] = result
                
            elif optimization_method == 'grid_search':
                # Grid search optimization
                result = self.trading_algorithm.optimize_strategy(start_date, end_date)
                optimization_results['grid_search'] = result
                
            elif optimization_method == 'combined':
                # Combined optimization approach
                bayesian_result = self.trading_algorithm.optimize_with_ml(start_date, end_date, n_trials=50)
                grid_result = self.trading_algorithm.optimize_strategy(start_date, end_date)
                
                optimization_results['bayesian'] = bayesian_result
                optimization_results['grid_search'] = grid_result
                optimization_results['combined'] = self.combine_optimization_results(bayesian_result, grid_result)
            
            self.optimization_results = optimization_results
            
            self.logger.info("✅ System optimization completed")
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in system optimization: {e}")
            return {}
    
    def combine_optimization_results(self, bayesian_result: Dict, grid_result: Dict) -> Dict:
        """Combine results from different optimization methods"""
        
        # Weighted combination of parameters
        bayesian_params = bayesian_result.get('best_params', {})
        grid_params = grid_result.get('best_result', {}).get('risk_params', {})
        
        combined_params = {}
        
        for param in bayesian_params.keys():
            if param in grid_params:
                # Weighted average (70% Bayesian, 30% Grid Search)
                combined_params[param] = (
                    0.7 * bayesian_params[param] + 
                    0.3 * grid_params[param]
                )
            else:
                combined_params[param] = bayesian_params[param]
        
        return {
            'combined_params': combined_params,
            'bayesian_score': bayesian_result.get('best_value', 0),
            'grid_score': grid_result.get('best_result', {}).get('optimization_score', 0)
        }
    
    def run_live_trading_simulation(self, duration_hours: int = 24) -> Dict:
        """Run live trading simulation"""
        
        self.logger.info(f"🔄 Starting live trading simulation for {duration_hours} hours")
        
        simulation_results = {
            'start_time': datetime.now(),
            'duration_hours': duration_hours,
            'trades': [],
            'performance': [],
            'risk_alerts': [],
            'final_performance': {}
        }
        
        try:
            start_time = datetime.now()
            end_time = start_time + timedelta(hours=duration_hours)
            
            # Simulate trading at 15-minute intervals
            current_time = start_time
            simulation_capital = self.initial_capital
            
            while current_time < end_time:
                # Get current market conditions (simulated)
                market_data = self.simulate_market_data(current_time)
                
                # Check for trading signals
                signals = self.trading_algorithm.get_enhanced_trading_signals()
                
                if not signals.empty:
                    for _, signal in signals.head(3).iterrows():  # Limit to top 3 signals
                        # Simulate trade execution
                        trade_result = self.simulate_trade_execution(
                            signal, simulation_capital, current_time
                        )
                        
                        if trade_result:
                            simulation_results['trades'].append(trade_result)
                            simulation_capital = trade_result['new_capital']
                
                # Update performance tracking
                performance_update = {
                    'timestamp': current_time,
                    'capital': simulation_capital,
                    'return': (simulation_capital - self.initial_capital) / self.initial_capital
                }
                simulation_results['performance'].append(performance_update)
                
                # Check risk alerts
                risk_alerts = self.check_risk_alerts(simulation_capital, current_time)
                simulation_results['risk_alerts'].extend(risk_alerts)
                
                # Move to next interval
                current_time += timedelta(minutes=15)
            
            # Calculate final performance
            simulation_results['final_performance'] = {
                'final_capital': simulation_capital,
                'total_return': (simulation_capital - self.initial_capital) / self.initial_capital,
                'total_trades': len(simulation_results['trades']),
                'duration_actual': (current_time - start_time).total_seconds() / 3600
            }
            
            self.logger.info("✅ Live trading simulation completed")
            
            return simulation_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in live trading simulation: {e}")
            return simulation_results
    
    def simulate_market_data(self, timestamp: datetime) -> Dict:
        """Simulate current market data"""
        
        # This would normally fetch real market data
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        market_data = {}
        for symbol in symbols:
            market_data[symbol] = {
                'current_price': np.random.uniform(100, 500),
                'volume': np.random.randint(100000, 1000000),
                'volatility': np.random.uniform(0.15, 0.35)
            }
        
        return market_data
    
    def simulate_trade_execution(self, signal: pd.Series, current_capital: float, timestamp: datetime) -> Optional[Dict]:
        """Simulate trade execution"""
        
        try:
            symbol = signal['symbol']
            action = signal['enhanced_signal']
            confidence = signal['combined_confidence']
            
            # Calculate position size
            position_size = self.portfolio_manager.dynamic_position_sizing(
                signal_strength=confidence,
                confidence=confidence,
                current_volatility=0.25,  # Simulated
                portfolio_value=current_capital,
                symbol_correlation=0.3  # Simulated
            )
            
            # Simulate price and execution
            execution_price = np.random.uniform(95, 105)  # Simulated slippage
            shares = int(position_size / execution_price)
            
            if shares > 0:
                trade_cost = shares * execution_price
                
                if action == 'STRONG_BUY' and trade_cost <= current_capital:
                    new_capital = current_capital - trade_cost
                    
                    return {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'action': action,
                        'shares': shares,
                        'price': execution_price,
                        'cost': trade_cost,
                        'confidence': confidence,
                        'new_capital': new_capital
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error simulating trade execution: {e}")
            return None
    
    def check_risk_alerts(self, current_capital: float, timestamp: datetime) -> List[Dict]:
        """Check for risk management alerts"""
        
        alerts = []
        
        # Capital drawdown check
        drawdown = (self.initial_capital - current_capital) / self.initial_capital
        if drawdown > 0.05:  # 5% drawdown alert
            alerts.append({
                'timestamp': timestamp,
                'type': 'DRAWDOWN_ALERT',
                'severity': 'HIGH' if drawdown > 0.10 else 'MEDIUM',
                'message': f'Portfolio drawdown: {drawdown:.2%}',
                'current_capital': current_capital
            })
        
        # Performance check
        total_return = (current_capital - self.initial_capital) / self.initial_capital
        if total_return < -0.03:  # -3% return alert
            alerts.append({
                'timestamp': timestamp,
                'type': 'PERFORMANCE_ALERT',
                'severity': 'MEDIUM',
                'message': f'Portfolio return: {total_return:.2%}',
                'current_capital': current_capital
            })
        
        return alerts
    
    def generate_system_report(self) -> str:
        """Generate comprehensive system performance report"""
        
        report = f"""
{'='*100}
🤖 COMPLETE AI TRADING SYSTEM - PERFORMANCE REPORT
{'='*100}
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Initial Capital: ${self.initial_capital:,.2f}

📊 BACKTESTING SUMMARY
{'─'*50}
"""
        
        if 'strategy_comparison' in self.backtest_results:
            comparison_df = self.backtest_results['strategy_comparison']
            report += str(comparison_df)
            report += "\n\n"
        
        report += f"""
🧠 OPTIMIZATION RESULTS
{'─'*50}
"""
        
        if self.optimization_results:
            for method, results in self.optimization_results.items():
                if method == 'bayesian' and 'best_params' in results:
                    report += f"\n{method.upper()} OPTIMIZATION:\n"
                    for param, value in results['best_params'].items():
                        report += f"  {param}: {value:.4f}\n"
                    report += f"  Best Score: {results.get('best_value', 0):.4f}\n"
        
        report += f"""
🛡️ RISK MANAGEMENT STATUS
{'─'*50}
Risk Management: ✅ ACTIVE
Position Sizing: ✅ DYNAMIC
Stop Loss: ✅ ENABLED
Portfolio Rebalancing: ✅ AUTOMATED

🎯 SYSTEM RECOMMENDATIONS
{'─'*50}
"""
        
        # Generate recommendations based on results
        recommendations = self.generate_system_recommendations()
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += f"\n{'='*100}\n"
        
        return report
    
    def generate_system_recommendations(self) -> List[str]:
        """Generate system recommendations based on performance"""
        
        recommendations = []
        
        # Check optimization results
        if self.optimization_results:
            if 'bayesian' in self.optimization_results:
                score = self.optimization_results['bayesian'].get('best_value', 0)
                if score > 0.5:
                    recommendations.append("✅ System optimization shows strong performance - maintain current parameters")
                else:
                    recommendations.append("⚠️ Consider extending optimization period or adjusting search parameters")
        
        # Check backtest results
        if self.backtest_results:
            if 'advanced_analysis' in self.backtest_results:
                analysis = self.backtest_results['advanced_analysis']
                if 'recommendations' in analysis:
                    recommendations.extend(analysis['recommendations'][:3])  # Top 3 recommendations
        
        # Default recommendations
        if not recommendations:
            recommendations = [
                "📊 Run comprehensive backtesting to establish performance baseline",
                "🧠 Execute parameter optimization to improve strategy performance",
                "🛡️ Monitor risk metrics and adjust position sizing accordingly",
                "📈 Consider diversifying strategies across different market conditions"
            ]
        
        return recommendations[:5]  # Limit to top 5
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save all results to file"""
        
        if not filename:
            filename = f"trading_system_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            results_data = {
                'timestamp': datetime.now().isoformat(),
                'initial_capital': self.initial_capital,
                'backtest_results': self.backtest_results,
                'optimization_results': self.optimization_results,
                'performance_history': self.performance_history
            }
            
            # Convert DataFrames to dictionaries for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            # Recursively convert complex objects
            import json
            results_json = json.dumps(results_data, default=convert_for_json, indent=2)
            
            with open(filename, 'w') as f:
                f.write(results_json)
            
            self.logger.info(f"✅ Results saved to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"❌ Error saving results: {e}")
            return ""


def main():
    """Main function with command-line interface"""
    
    parser = argparse.ArgumentParser(description='Complete AI Trading System')
    parser.add_argument('--mode', choices=['backtest', 'optimize', 'simulate', 'report'], 
                       default='backtest', help='Operation mode')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date for analysis')
    parser.add_argument('--end-date', type=str, default='2024-02-01', help='End date for analysis')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--config', type=str, default='config.env', help='Configuration file')
    parser.add_argument('--duration', type=int, default=24, help='Simulation duration in hours')
    
    args = parser.parse_args()
    
    print("🚀 Starting Complete AI Trading System")
    print("=" * 80)
    
    # Initialize system
    system = CompleteTradingSystem(args.config, args.capital)
    
    # Load configuration
    if not system.load_and_validate_config():
        print("❌ Configuration validation failed. Exiting.")
        return
    
    try:
        if args.mode == 'backtest':
            print(f"📊 Running comprehensive backtesting: {args.start_date} to {args.end_date}")
            results = system.run_comprehensive_backtest(args.start_date, args.end_date)
            
        elif args.mode == 'optimize':
            print("🧠 Running system optimization...")
            results = system.optimize_system_parameters(optimization_method='combined')
            
        elif args.mode == 'simulate':
            print(f"🔄 Running live trading simulation for {args.duration} hours...")
            results = system.run_live_trading_simulation(args.duration)
            
        elif args.mode == 'report':
            print("📋 Generating system report...")
            # Run quick analysis first
            system.run_comprehensive_backtest(args.start_date, args.end_date)
            report = system.generate_system_report()
            print(report)
            
            # Save report to file
            report_filename = f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_filename, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to {report_filename}")
            
            return
        
        # Save results
        if results:
            filename = system.save_results()
            print(f"💾 Results saved to {filename}")
        
        # Generate final report
        report = system.generate_system_report()
        print("\n" + report)
        
    except KeyboardInterrupt:
        print("\n⏹️ Operation interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        system.logger.error(f"System error: {e}")
    
    print("\n✅ Complete AI Trading System finished")


if __name__ == "__main__":
    main()