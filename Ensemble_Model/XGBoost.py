import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
import logging
import warnings
import os
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedUnifiedTradingSystem:
    """
    Fixed Trading System - แก้ไขปัญหาพื้นฐาน
    """
    
    def __init__(self, price_threshold=0.5):  # เปลี่ยนจาก 0.0 เป็น 0.5%
        self.price_threshold = price_threshold
        self.model = None
        self.feature_names = None
        self.training_stats = {}
        self.data_stats = {}
        self.scaler = None
    
    def validate_input_data(self, df):
        """ตรวจสอบข้อมูลที่เข้ามา"""
        logger.info("🔍 Validating input data...")
        
        required_columns = ['Ticker', 'Date', 'Actual_Price', 'Predicted_Price_LSTM', 
                           'Predicted_Price_GRU', 'Predicted_Dir_LSTM', 'Predicted_Dir_GRU']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # ตรวจสอบข้อมูลที่ผิดปกติ
        issues = []
        
        # 1. ราคาที่เป็น 0 หรือติดลบ
        invalid_prices = (df['Actual_Price'] <= 0).sum()
        if invalid_prices > 0:
            issues.append(f"Invalid prices: {invalid_prices} rows")
        
        # 2. Missing values
        missing_data = df[required_columns].isnull().sum().sum()
        if missing_data > 0:
            issues.append(f"Missing values: {missing_data} cells")
        
        # 3. Direction ที่อยู่นอกช่วง 0-1
        invalid_dirs_lstm = ((df['Predicted_Dir_LSTM'] < 0) | (df['Predicted_Dir_LSTM'] > 1)).sum()
        invalid_dirs_gru = ((df['Predicted_Dir_GRU'] < 0) | (df['Predicted_Dir_GRU'] > 1)).sum()
        if invalid_dirs_lstm + invalid_dirs_gru > 0:
            issues.append(f"Invalid directions: LSTM={invalid_dirs_lstm}, GRU={invalid_dirs_gru}")
        
        # 4. ราคาที่แตกต่างกันมากเกินไป
        price_diff_lstm = abs(df['Predicted_Price_LSTM'] - df['Actual_Price']) / df['Actual_Price']
        price_diff_gru = abs(df['Predicted_Price_GRU'] - df['Actual_Price']) / df['Actual_Price']
        
        extreme_diff_lstm = (price_diff_lstm > 0.5).sum()  # มากกว่า 50%
        extreme_diff_gru = (price_diff_gru > 0.5).sum()
        
        if extreme_diff_lstm + extreme_diff_gru > 0:
            issues.append(f"Extreme price differences: LSTM={extreme_diff_lstm}, GRU={extreme_diff_gru}")
        
        if issues:
            logger.warning("Data issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✅ Data validation passed")
        
        return len(issues) == 0
    
    def create_simple_features(self, df, is_training=True):
        """สร้าง features แบบง่าย ๆ ที่เข้าใจได้"""
        logger.info("🔧 Creating simple features...")
        
        df_clean = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        if is_training:
            # สำหรับ training: ใช้ Actual_Price เป็น current price และ target
            df_clean['Current_Price'] = df_clean['Actual_Price']
            # สร้าง future price โดยการ shift กลับ (ปลอดภัยกว่า)
            df_clean['Future_Price'] = df_clean.groupby('Ticker')['Actual_Price'].shift(-1)
            
            # คำนวณ return และ direction
            df_clean['Target_Return_Pct'] = (
                (df_clean['Future_Price'] - df_clean['Current_Price']) / 
                df_clean['Current_Price'] * 100
            )
            df_clean['Target_Direction'] = (
                df_clean['Target_Return_Pct'] > self.price_threshold
            ).astype(int)
            
            # เอาแถวสุดท้ายของแต่ละ ticker ออก (ไม่มี future price)
            df_clean = df_clean.groupby('Ticker').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
            
        else:
            # สำหรับ prediction: ใช้ราคาปัจจุบันที่ส่งมา
            if 'Current_Price' not in df_clean.columns:
                df_clean['Current_Price'] = df_clean['Actual_Price']
        
        # 1. ความแตกต่างของการทำนายราคา
        df_clean['Price_Diff_Pct'] = abs(
            df_clean['Predicted_Price_LSTM'] - df_clean['Predicted_Price_GRU']
        ) / df_clean['Current_Price'] * 100
        
        # 2. ราคาเฉลี่ย
        df_clean['Price_Avg'] = (
            df_clean['Predicted_Price_LSTM'] + df_clean['Predicted_Price_GRU']
        ) / 2
        
        # 3. การเปลี่ยนแปลงราคาจากปัจจุบัน
        df_clean['LSTM_Price_Change_Pct'] = (
            (df_clean['Predicted_Price_LSTM'] - df_clean['Current_Price']) / 
            df_clean['Current_Price'] * 100
        )
        df_clean['GRU_Price_Change_Pct'] = (
            (df_clean['Predicted_Price_GRU'] - df_clean['Current_Price']) / 
            df_clean['Current_Price'] * 100
        )
        df_clean['Avg_Price_Change_Pct'] = (
            df_clean['LSTM_Price_Change_Pct'] + df_clean['GRU_Price_Change_Pct']
        ) / 2
        
        # 4. Direction features
        df_clean['LSTM_Dir_Binary'] = (df_clean['Predicted_Dir_LSTM'] > 0.5).astype(int)
        df_clean['GRU_Dir_Binary'] = (df_clean['Predicted_Dir_GRU'] > 0.5).astype(int)
        df_clean['Dir_Agreement'] = (
            df_clean['LSTM_Dir_Binary'] == df_clean['GRU_Dir_Binary']
        ).astype(int)
        
        # 5. Direction confidence
        df_clean['LSTM_Dir_Confidence'] = abs(df_clean['Predicted_Dir_LSTM'] - 0.5) * 2
        df_clean['GRU_Dir_Confidence'] = abs(df_clean['Predicted_Dir_GRU'] - 0.5) * 2
        df_clean['Avg_Dir_Confidence'] = (
            df_clean['LSTM_Dir_Confidence'] + df_clean['GRU_Dir_Confidence']
        ) / 2
        
        # 6. Consistency check (สำคัญมาก!)
        lstm_price_direction = (df_clean['LSTM_Price_Change_Pct'] > self.price_threshold).astype(int)
        gru_price_direction = (df_clean['GRU_Price_Change_Pct'] > self.price_threshold).astype(int)
        
        df_clean['LSTM_Consistency'] = (
            lstm_price_direction == df_clean['LSTM_Dir_Binary']
        ).astype(int)
        df_clean['GRU_Consistency'] = (
            gru_price_direction == df_clean['GRU_Dir_Binary']
        ).astype(int)
        df_clean['Overall_Consistency'] = (
            df_clean['LSTM_Consistency'] + df_clean['GRU_Consistency']
        ) / 2
        
        # 7. Market context
        thai_stocks = ['ADVANC', 'DIF', 'DITTO', 'HUMAN', 'INET', 'INSET', 'JAS', 'JMART', 'TRUE']
        df_clean['Is_Thai'] = df_clean['Ticker'].isin(thai_stocks).astype(int)
        
        # 8. Price level categories
        def categorize_price(row):
            price = row['Current_Price']
            if row['Is_Thai']:
                if price < 10: return 0      # Low
                elif price < 50: return 1   # Medium  
                else: return 2              # High
            else:  # US stocks
                if price < 100: return 0    # Low
                elif price < 300: return 1  # Medium
                else: return 2              # High
        
        df_clean['Price_Category'] = df_clean.apply(categorize_price, axis=1)
        
        # 9. Simple interaction features
        df_clean['Agreement_Confidence'] = df_clean['Dir_Agreement'] * df_clean['Avg_Dir_Confidence']
        df_clean['Consistency_Score'] = df_clean['Overall_Consistency'] * df_clean['Avg_Dir_Confidence']
        
        logger.info(f"   Created {len([c for c in df_clean.columns if c not in df.columns])} features")
        
        return df_clean
    
    def train_simple_model(self, df):
        """เทรนโมเดลแบบง่าย ๆ"""
        logger.info("🚀 Training Simple XGBoost Model...")
        
        # Validate input data
        if not self.validate_input_data(df):
            logger.warning("Data validation failed, but continuing...")
        
        # Create features
        df_processed = self.create_simple_features(df, is_training=True)
        
        # Remove rows with missing target
        df_processed = df_processed.dropna(subset=['Target_Return_Pct', 'Target_Direction'])
        
        if len(df_processed) == 0:
            raise ValueError("No valid training data after preprocessing")
        
        # Define simple feature set
        feature_cols = [
            'Price_Diff_Pct', 'LSTM_Price_Change_Pct', 'GRU_Price_Change_Pct', 'Avg_Price_Change_Pct',
            'LSTM_Dir_Confidence', 'GRU_Dir_Confidence', 'Avg_Dir_Confidence',
            'Dir_Agreement', 'LSTM_Consistency', 'GRU_Consistency', 'Overall_Consistency',
            'Is_Thai', 'Price_Category', 'Agreement_Confidence', 'Consistency_Score'
        ]
        
        # Select available features
        available_features = [f for f in feature_cols if f in df_processed.columns]
        logger.info(f"   Using {len(available_features)} features")
        
        # Prepare data
        X = df_processed[available_features].fillna(0)
        y_returns = df_processed['Target_Return_Pct']
        y_directions = df_processed['Target_Direction']
        
        # Remove extreme outliers
        return_q99 = y_returns.quantile(0.99)
        return_q01 = y_returns.quantile(0.01)
        outlier_mask = (y_returns >= return_q01) & (y_returns <= return_q99)
        
        X = X[outlier_mask].reset_index(drop=True)
        y_returns = y_returns[outlier_mask].reset_index(drop=True)
        y_directions = y_directions[outlier_mask].reset_index(drop=True)
        
        logger.info(f"   After outlier removal: {len(X)} samples")
        
        # Time-based split (80/20)
        split_date = pd.to_datetime(df_processed['Date']).quantile(0.8)
        train_mask = pd.to_datetime(df_processed['Date']) < split_date
        train_mask = train_mask[outlier_mask].reset_index(drop=True)
        test_mask = ~train_mask
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_returns_train, y_returns_test = y_returns[train_mask], y_returns[test_mask]
        y_directions_train, y_directions_test = y_directions[train_mask], y_directions[test_mask]
        
        logger.info(f"   Train: {len(X_train)} samples")
        logger.info(f"   Test: {len(X_test)} samples")
        logger.info(f"   Direction balance - Train: {y_directions_train.mean():.3f}, Test: {y_directions_test.mean():.3f}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Simple XGBoost parameters
        xgb_params = {
            'n_estimators': 100,        # ลดลงจาก 200
            'max_depth': 4,            # ลดลงจาก 5
            'learning_rate': 0.1,      # เพิ่มขึ้นจาก 0.08
            'subsample': 0.8,          # ลดลงจาก 0.85
            'colsample_bytree': 0.8,   # ลดลงจาก 0.85
            'reg_alpha': 0.1,          # ลดลงจาก 0.3
            'reg_lambda': 0.1,         # ลดลงจาก 0.3
            'random_state': 42,
            'min_child_weight': 5,     # เพิ่มขึ้นจาก 3
            'gamma': 0.05              # ลดลงจาก 0.1
        }
        
        # Train separate models
        self.price_model = xgb.XGBRegressor(objective='reg:squarederror', **xgb_params)
        self.direction_model = xgb.XGBClassifier(objective='binary:logistic', **xgb_params)
        
        # Fit models
        self.price_model.fit(X_train_scaled, y_returns_train)
        self.direction_model.fit(X_train_scaled, y_directions_train)
        
        self.feature_names = available_features
        
        # Evaluate
        self._evaluate_model_performance(X_train_scaled, X_test_scaled, 
                                       y_returns_train, y_returns_test,
                                       y_directions_train, y_directions_test)
        
        logger.info("✅ Simple model training completed!")
        return self.training_stats
    
    def _evaluate_model_performance(self, X_train, X_test, y_returns_train, y_returns_test,
                                   y_directions_train, y_directions_test):
        """ประเมินประสิทธิภาพโมเดล"""
        
        # Predictions
        pred_returns_train = self.price_model.predict(X_train)
        pred_returns_test = self.price_model.predict(X_test)
        
        pred_dir_probs_train = self.direction_model.predict_proba(X_train)[:, 1]
        pred_dir_probs_test = self.direction_model.predict_proba(X_test)[:, 1]
        
        pred_directions_train = (pred_dir_probs_train > 0.5).astype(int)
        pred_directions_test = (pred_dir_probs_test > 0.5).astype(int)
        
        # Metrics
        train_price_r2 = r2_score(y_returns_train, pred_returns_train)
        test_price_r2 = r2_score(y_returns_test, pred_returns_test)
        
        train_dir_acc = accuracy_score(y_directions_train, pred_directions_train)
        test_dir_acc = accuracy_score(y_directions_test, pred_directions_test)
        
        # Consistency check (สำคัญมาก!)
        train_price_directions = (pred_returns_train > self.price_threshold).astype(int)
        test_price_directions = (pred_returns_test > self.price_threshold).astype(int)
        
        train_consistency = (train_price_directions == pred_directions_train).mean()
        test_consistency = (test_price_directions == pred_directions_test).mean()
        
        # Store stats
        self.training_stats = {
            'train_price_r2': train_price_r2,
            'test_price_r2': test_price_r2,
            'train_direction_accuracy': train_dir_acc,
            'test_direction_accuracy': test_dir_acc,
            'train_consistency': train_consistency,
            'test_consistency': test_consistency,
            'price_threshold': self.price_threshold,
            'feature_count': len(self.feature_names)
        }
        
        # Logging
        logger.info("📊 Model Performance:")
        logger.info(f"   Price R² - Train: {train_price_r2:.4f}, Test: {test_price_r2:.4f}")
        logger.info(f"   Direction Accuracy - Train: {train_dir_acc:.4f}, Test: {test_dir_acc:.4f}")
        logger.info(f"   🎯 Price-Direction Consistency - Train: {train_consistency:.4f}, Test: {test_consistency:.4f}")
        
        # Quality assessment
        if test_consistency < 0.7:
            logger.warning("🚨 LOW CONSISTENCY: Model predictions are inconsistent!")
        elif test_consistency >= 0.8:
            logger.info("✅ HIGH CONSISTENCY: Model predictions are consistent!")
        else:
            logger.info("⚠️ MODERATE CONSISTENCY: Model predictions are acceptable")
    
    def predict_signals(self, input_data):
        """Generate trading signals - ไม่แก้ไขผลลัพธ์อัตโนมัติ"""
        if self.price_model is None or self.direction_model is None:
            raise ValueError("Model must be trained first")
        
        if isinstance(input_data, dict):
            df = pd.DataFrame(input_data)
        else:
            df = input_data.copy()
        
        # Create features
        df_processed = self.create_simple_features(df, is_training=False)
        
        # Select features
        X = df_processed[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict (ใช้ผลลัพธ์ตรงจากโมเดล ไม่แก้ไข)
        predicted_returns = self.price_model.predict(X_scaled)
        predicted_dir_probs = self.direction_model.predict_proba(X_scaled)[:, 1]
        predicted_directions = (predicted_dir_probs > 0.5).astype(int)
        
        # Calculate predicted prices
        current_prices = df_processed['Current_Price'].values
        predicted_prices = current_prices * (1 + predicted_returns / 100)
        
        # ✅ ตรวจสอบความสอดคล้อง แต่ไม่แก้ไข (เก็บเป็นข้อมูลเท่านั้น)
        price_implied_directions = (predicted_returns > self.price_threshold).astype(int)
        inconsistent_mask = (predicted_directions != price_implied_directions)
        
        # 📊 เก็บสถิติความไม่สอดคล้อง แต่ไม่แก้ไข
        if inconsistent_mask.sum() > 0:
            logger.info(f"🚨 Found {inconsistent_mask.sum()}/{len(inconsistent_mask)} inconsistent predictions (NOT FIXED)")
            logger.info(f"🎯 Raw Model Consistency: {(1 - inconsistent_mask.mean()):.1%}")
        else:
            logger.info("✅ All predictions are naturally consistent")
        
        # ❌ ลบส่วนการแก้ไขออกทั้งหมด - ไม่แก้ไข predicted_directions และ predicted_dir_probs
        # ❌ predicted_directions[inconsistent_mask] = price_implied_directions[inconsistent_mask]
        # ❌ predicted_dir_probs[inconsistent_mask] = np.where(...)
        
        # ✅ คำนวณ confidence แบบธรรมชาติ (ไม่มี consistency bonus)
        base_confidence = np.abs(predicted_dir_probs - 0.5) * 2
        # ❌ ลบ consistency bonus ออก
        # ❌ consistency_bonus = (~inconsistent_mask).astype(float) * 0.1
        final_confidence = np.clip(base_confidence, 0.1, 0.9)
        
        # Generate results (เพิ่มข้อมูลความไม่สอดคล้อง)
        results = []
        for idx in range(len(df_processed)):
            results.append({
                'Ticker': df_processed.iloc[idx]['Ticker'],
                'Current_Price': current_prices[idx],
                'Predicted_Price': predicted_prices[idx],
                'Predicted_Return_Pct': predicted_returns[idx],
                'Predicted_Direction': predicted_directions[idx],  # ✅ ผลจริงจากโมเดล
                'Direction_Probability': predicted_dir_probs[idx],  # ✅ ผลจริงจากโมเดล
                'Confidence': final_confidence[idx],  # ✅ ไม่มี artificial bonus
                'Is_Inconsistent': inconsistent_mask[idx],  # ✅ เพิ่มข้อมูลความไม่สอดคล้อง
                'Price_Implied_Direction': price_implied_directions[idx],  # ✅ เพิ่มเพื่อเปรียบเทียบ
                'Model_Consistency': self.training_stats.get('test_consistency', 0)  # ✅ ความสอดคล้องจริงจากการเทรน
            })
        
        return pd.DataFrame(results)
    
    def save_model(self, path='./fixed_unified_trading_model.pkl'):
        """Save model"""
        model_data = {
            'price_model': self.price_model,
            'direction_model': self.direction_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'price_threshold': self.price_threshold,
            'version': '1.0_fixed'
        }
        joblib.dump(model_data, path)
        logger.info(f"✅ Fixed model saved to {path}")
    
    def load_model(self, path='./fixed_unified_trading_model.pkl'):
        """Load model"""
        model_data = joblib.load(path)
        self.price_model = model_data['price_model']
        self.direction_model = model_data['direction_model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.training_stats = model_data['training_stats']
        self.price_threshold = model_data.get('price_threshold', 0.5)
        logger.info(f"✅ Fixed model loaded from {path}")

# ===== MAIN TRAINING FUNCTION =====

def train_fixed_system():
    """Train the fixed system"""
    try:
        logger.info("🚀 Starting Fixed Unified Training...")
        
        # Load data
        lstm_path = "../LSTM_model/all_predictions_per_day_multi_task.csv"
        gru_path = "../GRU_Model/all_predictions_per_day_multi_task.csv"
        
        if not os.path.exists(lstm_path) or not os.path.exists(gru_path):
            raise FileNotFoundError("Required data files not found")
        
        lstm_df = pd.read_csv(lstm_path)
        gru_df = pd.read_csv(gru_path)
        
        # Combine data
        df = pd.DataFrame({
            "Ticker": lstm_df["Ticker"],
            "Date": pd.to_datetime(lstm_df["Date"]),
            "Actual_Price": lstm_df["Actual_Price"],
            "Predicted_Price_LSTM": lstm_df["Predicted_Price"],
            "Predicted_Price_GRU": gru_df["Predicted_Price"],
            "Actual_Direction": lstm_df["Actual_Dir"],
            "Predicted_Dir_LSTM": lstm_df["Predicted_Dir"],
            "Predicted_Dir_GRU": gru_df["Predicted_Dir"]
        })
        
        logger.info(f"   Raw data: {len(df)} records")
        
        # Initialize and train fixed system
        trading_system = FixedUnifiedTradingSystem(price_threshold=0.5)
        training_stats = trading_system.train_simple_model(df)
        
        # Save model
        trading_system.save_model('./fixed_unified_trading_model.pkl')
        
        # Display results
        logger.info("\n🎉 Fixed Training Results:")
        logger.info(f"   Price R²: {training_stats['test_price_r2']:.4f}")
        logger.info(f"   Direction Accuracy: {training_stats['test_direction_accuracy']:.4f}")
        logger.info(f"   🎯 Model Consistency: {training_stats['test_consistency']:.4f}")
        
        # Quality assessment
        if training_stats['test_consistency'] >= 0.8:
            logger.info("✅ EXCELLENT: High model consistency!")
        elif training_stats['test_consistency'] >= 0.7:
            logger.info("✅ GOOD: Acceptable model consistency")
        else:
            logger.warning("⚠️ POOR: Low model consistency - needs improvement")
        
        return trading_system, training_stats
        
    except Exception as e:
        logger.error(f"❌ Fixed training failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# ===== USAGE EXAMPLE =====

if __name__ == "__main__":
    print("🚀 Starting Fixed XGBoost Training...")
    
    try:
        # Train fixed system
        trading_system, stats = train_fixed_system()
        
        print("\n📊 Final Results Summary:")
        print(f"   Price R²: {stats['test_price_r2']:.4f}")
        print(f"   Direction Accuracy: {stats['test_direction_accuracy']:.4f}")
        print(f"   🎯 Model Consistency: {stats['test_consistency']:.4f}")
        
        # Test prediction
        print("\n🧪 Testing Fixed Prediction...")
        sample_data = {
            'Ticker': ['AAPL'],
            'Date': ['2025-07-30'],
            'Actual_Price': [213.88],  # ใช้เป็น current price
            'Predicted_Price_LSTM': [215.50],
            'Predicted_Price_GRU': [214.80],
            'Predicted_Dir_LSTM': [0.65],  # ขึ้น
            'Predicted_Dir_GRU': [0.62]    # ขึ้น
        }
        
        signals = trading_system.predict_signals(sample_data)
        result = signals.iloc[0]
        
        print(f"\nFixed prediction for {result['Ticker']}:")
        print(f"   Current Price: ${result['Current_Price']:.2f}")
        print(f"   Predicted Price: ${result['Predicted_Price']:.2f}")
        print(f"   Expected Return: {result['Predicted_Return_Pct']:+.2f}%")
        print(f"   Direction: {'📈 UP' if result['Predicted_Direction'] == 1 else '📉 DOWN'}")
        print(f"   Confidence: {result['Confidence']:.3f}")
        print(f"   Is Inconsistent: {'Yes' if result['Is_Inconsistent'] else 'No'}")
        print(f"   Model Consistency: {result['Model_Consistency']:.3f}")
        
        print(f"\n✅ Fixed model saved as: ./fixed_unified_trading_model.pkl")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        print(traceback.format_exc())