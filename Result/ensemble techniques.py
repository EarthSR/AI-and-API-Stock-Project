import pandas as pd
import numpy as np

# โหลดข้อมูลการพยากรณ์จากแต่ละโมเดล
model_LSTM = pd.read_csv('./predictions_per_ticker_LSTM.csv')
model_MLP = pd.read_csv('./predictions_per_ticker_MLP.csv')
model_RNN = pd.read_csv('./predictions_per_ticker_RNN.csv')
model_CNN = pd.read_csv('./predictions_per_ticker_CNN.csv')
model_GRU = pd.read_csv('./predictions_per_ticker_GRU.csv')

# โหลดข้อมูล MAPE ของแต่ละโมเดลรายหุ้น
accuracy_LSTM = pd.read_csv('./metrics_per_ticker_LSTM.csv')
accuracy_MLP = pd.read_csv('./metrics_per_ticker_MLP.csv')
accuracy_RNN = pd.read_csv('./metrics_per_ticker_RNN.csv')
accuracy_CNN = pd.read_csv('./metrics_per_ticker_CNN.csv')
accuracy_GRU = pd.read_csv('./metrics_per_ticker_GRU.csv')

# รวมข้อมูล MAPE ใน DataFrame เดียว
accuracy = pd.DataFrame({
    'Ticker': accuracy_LSTM['Ticker'],
    'LSTM_MAPE': accuracy_LSTM['MAPE'],
    'MLP_MAPE': accuracy_MLP['MAPE'],
    'RNN_MAPE': accuracy_RNN['MAPE'],
    'CNN_MAPE': accuracy_CNN['MAPE'],
    'GRU_MAPE': accuracy_GRU['MAPE']
})

# รวมข้อมูลการพยากรณ์ใน DataFrame เดียว
predictions = pd.DataFrame({
    'Ticker': model_LSTM['Ticker'],
    'LSTM': model_LSTM['Predicted'],
    'MLP': model_MLP['Predicted'],
    'RNN': model_RNN['Predicted'],
    'CNN': model_CNN['Predicted'],
    'GRU': model_GRU['Predicted']
})

# 1. คำนวณ Ensemble โดยใช้ Average
predictions['Average'] = predictions[['LSTM', 'MLP', 'RNN', 'CNN', 'GRU']].mean(axis=1)

# 2. คำนวณ Ensemble โดยใช้ Median
predictions['Median'] = predictions[['LSTM', 'MLP', 'RNN', 'CNN', 'GRU']].median(axis=1)

# รวมข้อมูล MAPE ของโมเดลเข้ากับ DataFrame การพยากรณ์
predictions = predictions.merge(accuracy, on='Ticker')

# กำหนด epsilon เพื่อป้องกันการหารด้วยศูนย์
epsilon = 1e-6

# คำนวณน้ำหนักจากส่วนกลับของ MAPE
for model in ['LSTM', 'MLP', 'RNN', 'CNN', 'GRU']:
    predictions[f'{model}_weight'] = 1 / (predictions[f'{model}_MAPE'] + epsilon)

# 3. คำนวณ Ensemble โดยใช้ Weight Sum
predictions['WeightSum'] = (
    predictions['LSTM'] * predictions['LSTM_weight'] +
    predictions['MLP']  * predictions['MLP_weight'] +
    predictions['RNN']  * predictions['RNN_weight'] +
    predictions['CNN']  * predictions['CNN_weight'] +
    predictions['GRU']  * predictions['GRU_weight']
) / (
    predictions['LSTM_weight'] +
    predictions['MLP_weight'] +
    predictions['RNN_weight'] +
    predictions['CNN_weight'] +
    predictions['GRU_weight']
)

# เลือกเฉพาะคอลัมน์ที่ต้องการ
final_predictions = predictions[['Ticker', 'Average', 'Median', 'WeightSum']]

# บันทึกผลลัพธ์
final_predictions.to_csv('./ensemble_predictions_summary.csv', index=False)

# คำนวณค่าเฉลี่ยต่อ Ticker
final_summary = final_predictions.groupby('Ticker', as_index=False)[['Average', 'Median', 'WeightSum']].mean()
final_summary.to_csv('./ensemble_predictions_summary_per_ticker.csv', index=False)

# แสดงตัวอย่างผลลัพธ์
print(final_predictions.head())
print(final_summary.head())
