import pandas as pd
import numpy as np

# โหลดข้อมูลการพยากรณ์จากแต่ละโมเดล
model_LSTM = pd.read_csv('./predictions_per_ticker_LSTM.csv')
model_MLP = pd.read_csv('./predictions_per_ticker_MLP.csv')
model_RNN = pd.read_csv('./predictions_per_ticker_RNN.csv')
model_CNN = pd.read_csv('./predictions_per_ticker_CNN.csv')
model_GRU = pd.read_csv('./predictions_per_ticker_GRU.csv')

# โหลดข้อมูลความถูกต้อง (accuracy) ของแต่ละโมเดลรายหุ้น
accuracy_LSTM = pd.read_csv('./metrics_per_ticker_LSTM.csv')
accuracy_GRU = pd.read_csv('./metrics_per_ticker_GRU.csv')
accuracy_RNN = pd.read_csv('./metrics_per_ticker_RNN.csv')
accuracy_CNN = pd.read_csv('./metrics_per_ticker_CNN.csv')
accuracy_MLP = pd.read_csv('./metrics_per_ticker_MLP.csv')

# รวม accuracy ของทุกโมเดลใน DataFrame เดียวกัน
accuracy = pd.DataFrame({
    'Ticker': accuracy_LSTM['Ticker'],  # สมมติว่าทุกไฟล์มีคอลัมน์ Ticker เหมือนกัน
    'LSTM_Accuracy': accuracy_LSTM['R2'],
    'MLP_Accuracy': accuracy_MLP['R2'],
    'RNN_Accuracy': accuracy_RNN['R2'],
    'CNN_Accuracy': accuracy_CNN['R2'],
    'GRU_Accuracy': accuracy_GRU['R2']
})

# รวมข้อมูลการพยากรณ์ทั้งหมดใน DataFrame เดียวกัน
predictions = pd.DataFrame({
    'Ticker': model_LSTM['Ticker'],  # สมมติว่าไฟล์พยากรณ์แต่ละไฟล์มีคอลัมน์ Ticker
    'LSTM': model_LSTM['prediction'],
    'MLP': model_MLP['prediction'],
    'RNN': model_RNN['prediction'],
    'CNN': model_CNN['prediction'],
    'GRU': model_GRU['prediction']
})

# 1. Ensemble โดยใช้ Mean
predictions['Mean'] = predictions[['LSTM', 'MLP', 'RNN', 'CNN', 'GRU']].mean(axis=1)

# 2. Ensemble โดยใช้ Median
predictions['Median'] = predictions[['LSTM', 'MLP', 'RNN', 'CNN', 'GRU']].median(axis=1)

# รวมข้อมูลความถูกต้องของโมเดลเข้ากับ DataFrame
predictions = predictions.merge(accuracy, on='Ticker')

# 3. Ensemble โดยใช้ Weighted Sum
predictions['Weighted'] = (
    predictions['LSTM'] * predictions['LSTM_Accuracy'] +
    predictions['MLP'] * predictions['MLP_Accuracy'] +
    predictions['RNN'] * predictions['RNN_Accuracy'] +
    predictions['CNN'] * predictions['CNN_Accuracy'] +
    predictions['GRU'] * predictions['GRU_Accuracy']
) / (
    predictions['LSTM_Accuracy'] +
    predictions['MLP_Accuracy'] +
    predictions['RNN_Accuracy'] +
    predictions['CNN_Accuracy'] +
    predictions['GRU_Accuracy']
)

# เลือกเฉพาะคอลัมน์ที่ต้องการ
final_predictions = predictions[['Ticker', 'Mean', 'Median', 'Weighted']]

# บันทึกผลลัพธ์การ Ensemble ลงไฟล์
final_predictions.to_csv('./ensemble_predictions_summary.csv', index=False)

# แสดงตัวอย่างผลลัพธ์
print(final_predictions.head())
