import pandas as pd
import numpy as np
from Autotrainmodel import XGBoostEnsembleMetaLearner

# Create test data similar to what we used during XGBoost training
test_data = pd.DataFrame({
    'Ticker': ['AAPL', 'ADVANC', 'AMD'],
    'Date': ['2025-08-04', '2025-08-04', '2025-08-04'],
    'Current_Price': [213.88, 220.50, 145.30],
    'Predicted_Price_LSTM': [215.50, 222.10, 147.20],
    'Predicted_Price_GRU': [214.80, 221.40, 146.80],
    'Predicted_Dir_LSTM': [0.65, 0.72, 0.58],
    'Predicted_Dir_GRU': [0.62, 0.68, 0.55]
})

print("🧪 Testing XGBoost Integration in Autotrainmodel.py")
print("=" * 60)

# Initialize XGBoost ensemble
print("1. Initializing XGBoost Ensemble...")
try:
    ensemble = XGBoostEnsembleMetaLearner()
    print("✅ XGBoost Ensemble initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    exit(1)

# Test prediction
print("\n2. Testing prediction with sample data...")
print("Sample data:")
print(test_data[['Ticker', 'Current_Price', 'Predicted_Price_LSTM', 'Predicted_Price_GRU']])

try:
    # Use predict_meta method
    result_df = ensemble.predict_meta(test_data.copy())
    
    print("\n3. XGBoost Results:")
    print("=" * 60)
    
    # Check if XGBoost predictions were added
    xgb_columns = [col for col in result_df.columns if 'XGB' in col]
    if xgb_columns:
        print("✅ XGBoost predictions generated successfully!")
        print(f"XGBoost columns: {xgb_columns}")
        
        # Display results for each stock
        for idx, row in result_df.iterrows():
            ticker = row['Ticker']
            current_price = row['Current_Price']
            
            if 'XGB_Predicted_Price' in row:
                xgb_price = row['XGB_Predicted_Price']
                xgb_direction = row['XGB_Predicted_Direction']
                xgb_confidence = row['XGB_Confidence']
                ensemble_method = row.get('Ensemble_Method', 'Unknown')
                
                print(f"\n📊 {ticker}:")
                print(f"   Current Price: ${current_price:.2f}")
                print(f"   XGB Predicted Price: ${xgb_price:.2f}")
                print(f"   XGB Direction: {'UP' if xgb_direction == 1 else 'DOWN'}")
                print(f"   XGB Confidence: {xgb_confidence:.3f}")
                print(f"   Method: {ensemble_method}")
                
                # Compare with individual models
                lstm_price = row['Predicted_Price_LSTM']
                gru_price = row['Predicted_Price_GRU']
                print(f"   LSTM Price: ${lstm_price:.2f}")
                print(f"   GRU Price: ${gru_price:.2f}")
                print(f"   Simple Average: ${(lstm_price + gru_price)/2:.2f}")
                
    else:
        print("❌ No XGBoost predictions found in results")
        print("Available columns:", list(result_df.columns))
        
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n4. Model Status Check:")
print(f"   Model Available: {ensemble.is_model_available()}")
print(f"   Model Status: {ensemble.get_model_status()}")

print("\n🎯 Test Complete!")