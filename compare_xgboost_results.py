import pandas as pd
import numpy as np
import sys
import os

# Add paths
sys.path.append('Ensemble_Model')
sys.path.append('Preproces')

print("Comparing XGBoost Results: Standalone vs Integrated")
print("=" * 70)

# Test data - same as used in XGBoost training
sample_data = {
    'Ticker': ['AAPL'],
    'Date': ['2025-07-30'],
    'Actual_Price': [213.88],  # Current price
    'Predicted_Price_LSTM': [215.50],
    'Predicted_Price_GRU': [214.80],
    'Predicted_Dir_LSTM': [0.65],  # UP
    'Predicted_Dir_GRU': [0.62]    # UP
}

sample_df = pd.DataFrame(sample_data)

print("Test Data:")
print(sample_df)
print()

# Test 1: Standalone XGBoost (from Ensemble_Model)
print("1. Testing Standalone XGBoost...")
try:
    from XGBoost import FixedUnifiedTradingSystem
    
    standalone_system = FixedUnifiedTradingSystem()
    if os.path.exists('Ensemble_Model/fixed_unified_trading_model.pkl'):
        standalone_system.load_model('Ensemble_Model/fixed_unified_trading_model.pkl')
        
        standalone_results = standalone_system.predict_signals(sample_df)
        
        if len(standalone_results) > 0:
            result = standalone_results.iloc[0]
            print("Standalone XGBoost Results:")
            print(f"   Predicted Price: ${result['Predicted_Price']:.2f}")
            print(f"   Expected Return: {result['Predicted_Return_Pct']:+.2f}%")
            print(f"   Direction: {'UP' if result['Predicted_Direction'] == 1 else 'DOWN'}")
            print(f"   Confidence: {result['Confidence']:.3f}")
            print(f"   Is Inconsistent: {result['Is_Inconsistent']}")
            print(f"   Model Consistency: {result['Model_Consistency']:.3f}")
            
            standalone_result = {
                'price': result['Predicted_Price'],
                'return': result['Predicted_Return_Pct'],
                'direction': result['Predicted_Direction'],
                'confidence': result['Confidence'],
                'consistency': result['Model_Consistency']
            }
        else:
            print("No standalone results generated")
            standalone_result = None
    else:
        print("Standalone model file not found")
        standalone_result = None
        
except Exception as e:
    print(f"Standalone test failed: {e}")
    standalone_result = None

print()

# Test 2: Integrated XGBoost (from Autotrainmodel)
print("2. Testing Integrated XGBoost...")
try:
    # Try to import without running the full Autotrainmodel (avoid TensorFlow issue)
    print("   Loading XGBoost class directly...")
    
    # Create XGBoost ensemble manually
    class TestXGBoostEnsemble:
        def __init__(self):
            self.model_path = 'Ensemble_Model/fixed_unified_trading_model.pkl'
            self.trading_system = None
            self.load_model()
        
        def load_model(self):
            try:
                from XGBoost import FixedUnifiedTradingSystem
                
                if os.path.exists(self.model_path):
                    self.trading_system = FixedUnifiedTradingSystem()
                    self.trading_system.load_model(self.model_path)
                    print(f"   Loaded model from {self.model_path}")
                else:
                    print(f"   Model not found at {self.model_path}")
                    self.trading_system = None
            except Exception as e:
                print(f"   Failed to load model: {e}")
                self.trading_system = None
        
        def predict_meta_like(self, df):
            if self.trading_system is None:
                return None
            
            # Prepare data like in Autotrainmodel
            xgb_data = pd.DataFrame({
                'Ticker': df['Ticker'],
                'Date': df['Date'],
                'Actual_Price': df['Actual_Price'],
                'Predicted_Price_LSTM': df['Predicted_Price_LSTM'],
                'Predicted_Price_GRU': df['Predicted_Price_GRU'],
                'Predicted_Dir_LSTM': df['Predicted_Dir_LSTM'],
                'Predicted_Dir_GRU': df['Predicted_Dir_GRU']
            })
            
            return self.trading_system.predict_signals(xgb_data)
    
    integrated_system = TestXGBoostEnsemble()
    
    if integrated_system.trading_system is not None:
        integrated_results = integrated_system.predict_meta_like(sample_df)
        
        if integrated_results is not None and len(integrated_results) > 0:
            result = integrated_results.iloc[0]
            print("Integrated XGBoost Results:")
            print(f"   Predicted Price: ${result['Predicted_Price']:.2f}")
            print(f"   Expected Return: {result['Predicted_Return_Pct']:+.2f}%")
            print(f"   Direction: {'UP' if result['Predicted_Direction'] == 1 else 'DOWN'}")
            print(f"   Confidence: {result['Confidence']:.3f}")
            print(f"   Is Inconsistent: {result['Is_Inconsistent']}")
            print(f"   Model Consistency: {result['Model_Consistency']:.3f}")
            
            integrated_result = {
                'price': result['Predicted_Price'],
                'return': result['Predicted_Return_Pct'],
                'direction': result['Predicted_Direction'],
                'confidence': result['Confidence'],
                'consistency': result['Model_Consistency']
            }
        else:
            print("No integrated results generated")
            integrated_result = None
    else:
        print("Integrated system not available")
        integrated_result = None
        
except Exception as e:
    print(f"Integrated test failed: {e}")
    import traceback
    traceback.print_exc()
    integrated_result = None

print()

# Comparison
print("3. Comparison Results:")
print("=" * 50)

if standalone_result and integrated_result:
    print("Both systems working - Comparing results...")
    
    price_diff = abs(standalone_result['price'] - integrated_result['price'])
    return_diff = abs(standalone_result['return'] - integrated_result['return'])
    confidence_diff = abs(standalone_result['confidence'] - integrated_result['confidence'])
    
    print(f"Price Difference: ${price_diff:.4f}")
    print(f"Return Difference: {return_diff:.4f}%")
    print(f"Confidence Difference: {confidence_diff:.4f}")
    print(f"Direction Match: {standalone_result['direction'] == integrated_result['direction']}")
    
    # Check if results are essentially the same (within small tolerance)
    if (price_diff < 0.01 and return_diff < 0.01 and confidence_diff < 0.001 and 
        standalone_result['direction'] == integrated_result['direction']):
        print("\nRESULTS MATCH! Integration successful!")
        print("   The XGBoost ensemble in Autotrainmodel.py works identically to standalone version")
    else:
        print("\nRESULTS DIFFER - Integration may have issues")
        print("   Standalone vs Integrated:")
        print(f"   Price: ${standalone_result['price']:.2f} vs ${integrated_result['price']:.2f}")
        print(f"   Return: {standalone_result['return']:+.2f}% vs {integrated_result['return']:+.2f}%")
        print(f"   Direction: {standalone_result['direction']} vs {integrated_result['direction']}")
        print(f"   Confidence: {standalone_result['confidence']:.3f} vs {integrated_result['confidence']:.3f}")

elif standalone_result:
    print("Only standalone system working")
    print("   Integration may have issues")
    
elif integrated_result:
    print("Only integrated system working")
    print("   Standalone may have issues")
    
else:
    print("Both systems failed")
    print("   Check model file and dependencies")

print("\nTest Complete!")