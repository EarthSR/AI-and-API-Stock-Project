import numpy as np
import xgboost as xgb
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# ✅ โหลดโมเดลที่ฝึกไว้แล้ว
model_1 = load_model('model_1.keras')
model_2 = load_model('model_2.keras')
model_3 = load_model('model_3.keras')

df = pd.read_csv('../merged_stock_sentiment_financial.csv')
# ✅ โหลด Scaler ที่เคยใช้สเกลข้อมูล
scaler_features = joblib.load('./LSTM_model/scaler_features.pkl')
scaler_target = joblib.load('./LSTM_model/scaler_target.pkl')

feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment','Total Revenue','QoQ Growth (%)', 
                   'YoY Growth (%)', 'Net Profit', 'Earnings Per Share (EPS)', 'ROA (%)', 'ROE (%)', 
                   'Gross Margin (%)', 'Net Profit Margin (%)', 'Debt to Equity ', 'P/E Ratio ',
                   'P/BV Ratio ', 'Dividend Yield (%)','RSI', 'EMA_10', 'EMA_20', 'MACD', 'MACD_Signal',
                   'Bollinger_High', 'Bollinger_Low']


sorted_dates = df['Date'].unique()
train_cutoff = sorted_dates[int(len(sorted_dates) * 6 / 7)]  # ขอบเขตที่ 6 ปี

# ข้อมูล train, test
train_df = df[df['Date'] <= train_cutoff].copy()
test_df = df[df['Date'] > train_cutoff].copy()

train_df.to_csv('train_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)
print("Train cutoff:", train_cutoff)
print("First date in train set:", train_df['Date'].min())
print("Last date in train set:", train_df['Date'].max())

# ✅ ทำนายผลลัพธ์จากแต่ละโมเดล
y_pred_1 = model_1.predict(test_features_scaled).flatten()
y_pred_2 = model_2.predict(test_features_scaled).flatten()
y_pred_3 = model_3.predict(test_features_scaled).flatten()

# ✅ รวมผลลัพธ์เป็นฟีเจอร์ของ XGBoost
X_test_xgb = np.column_stack((y_pred_1, y_pred_2, y_pred_3))

# ✅ โหลดโมเดล XGBoost ที่เคยฝึกไว้
xgb_model = joblib.load('xgb_meta_learner.pkl')

# ✅ ทำนายผลลัพธ์สุดท้ายด้วย XGBoost
y_pred_xgb = xgb_model.predict(X_test_xgb)

# ✅ ย้อนกลับค่าที่ทำนายกลับไปสเกลเดิม
y_pred_final = scaler_target.inverse_transform(y_pred_xgb.reshape(-1, 1))
y_test_final = scaler_target.inverse_transform(test_targets.reshape(-1, 1))

# ✅ ประเมินผลลัพธ์
mae = mean_absolute_error(y_test_final, y_pred_final)
mse = mean_squared_error(y_test_final, y_pred_final)
r2 = r2_score(y_test_final, y_pred_final)

print(f"✅ MAE: {mae:.4f}")
print(f"✅ MSE: {mse:.4f}")
print(f"✅ R2 Score: {r2:.4f}")
