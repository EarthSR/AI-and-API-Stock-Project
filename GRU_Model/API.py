import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

# สร้างแอปพลิเคชัน Flask
app = Flask(__name__)

# ฟังก์ชัน Preprocess ข้อมูล
def preprocess_input(input_data):
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)', 'Sentiment', 'Confidence',
                       'RSI', 'SMA_10', 'SMA_200', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low']
    
    # เปลี่ยนข้อมูลจาก list of dict เป็น DataFrame
    df = pd.DataFrame(input_data)

    # ตรวจสอบข้อมูลก่อนการ preprocess
    print(f"Original DataFrame:\n{df}")

    # การคำนวณและแทนที่ NaN หรือค่าผิดพลาด
    df.fillna(method='ffill', inplace=True)
    df.replace(0, np.nan, inplace=True)  # แทนที่ 0 ด้วย NaN
    df.fillna(df.mean(), inplace=True)   # เติมค่าที่ขาดหายด้วยค่าเฉลี่ยของแต่ละคอลัมน์

    # คำนวณค่า Change และ Change (%) ให้ถูกต้อง
    df['Change'] = df['Close'] - df['Open']
    df['Change (%)'] = (df['Change'] / df['Open']) * 100  # เปลี่ยนจากส่วนต่างเป็นเปอร์เซ็นต์

    # ฟิลด์ต่างๆที่ต้องการ
    df['RSI'] = df['RSI']
    df['RSI'].fillna(method='ffill', inplace=True)
    df['RSI'].fillna(0, inplace=True)

    df['SMA_10'] = df['SMA_10']
    df['SMA_200'] = df['SMA_200']
    df['MACD'] = df['MACD']
    df['MACD_Signal'] = df['MACD_Signal']
    df['Bollinger_High'] = df['Bollinger_High']
    df['Bollinger_Low'] = df['Bollinger_Low']

    # เติม NaN ที่เหลือ
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)

    # ตรวจสอบข้อมูลหลังการ preprocess
    print(f"Processed DataFrame:\n{df}")

    # ตรวจสอบค่าต่ำสุดและค่าสูงสุดของข้อมูล
    print(f"Min values:\n{df[feature_columns].min()}")
    print(f"Max values:\n{df[feature_columns].max()}")

    # ใช้ MinMaxScaler แทน StandardScaler เพื่อให้ข้อมูลอยู่ในช่วงที่เหมาะสม (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(df[feature_columns])

    # ตรวจสอบค่าที่ถูกสเกลแล้ว
    print(f"Scaled Features:\n{features_scaled}")

    return features_scaled


# ฟังก์ชันสำหรับการทำนาย
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # ข้อมูล ticker_ids
        ticker_ids = data.get('ticker_ids')
        if not ticker_ids:
            return jsonify({'error': 'ticker_ids is required'}), 400

        # ข้อมูล features
        features_data = data.get('features')
        if not features_data:
            return jsonify({'error': 'features data is required'}), 400

        # ประมวลผลข้อมูล
        features = preprocess_input(features_data)

        # โหลดโมเดลที่ฝึกเสร็จแล้ว
        model = load_model('./best_price_model_full.keras')

        # ทำนายผลลัพธ์จากโมเดล
        predictions = model.predict(features)

        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
