# AI Stock Prediction System 📈

ระบบทำนายราคาหุ้นด้วย AI ที่ใช้โมเดล Deep Learning หลายตัวร่วมกับ Ensemble Learning สำหรับทำนายราคาหุ้นอเมริกาและไทย

## ✨ คุณสมบัติหลัก

- **🧠 Multi-Model Architecture**: ใช้ LSTM และ GRU ร่วมกับ XGBoost Ensemble
- **📊 Multi-Task Learning**: ทำนายทั้งราคาและทิศทางการเคลื่อนไหวพร้อมกัน
- **🔄 Walk-Forward Validation**: ระบบ Mini-Retrain ทุก 3 วันสำหรับความแม่นยำสูงสุด
- **📰 Sentiment Analysis**: วิเคราะห์ข่าวสารด้วย FinBERT และ NLP
- **🌐 Multi-Market Support**: รองรับหุ้นอเมริกา (10 ตัว) และหุ้นไทย (9 ตัว)
- **📱 API Integration**: API สำหรับ Paper Trading และ Mobile App

## 🏗️ สถาปัตยกรรมระบบ

### 1. โมเดลหลัก
```
📊 Raw Data (Stock + News + Financial)
    ↓
🔧 Data Processing & Technical Indicators  
    ↓
🧠 LSTM Model ──┐
                ├── 🎯 XGBoost Ensemble → 📈 Final Predictions
🧠 GRU Model  ──┘
```

### 2. การไหลของข้อมูล
- **ข้อมูลหุ้น**: ราคา, ปริมาณ, ตัวชี้วัดทางเทคนิค (19 features)
- **ข้อมูลข่าวสาร**: Sentiment Analysis จากข่าวเศรษฐกิจ
- **ข้อมูลการเงิน**: งบการเงินรายไตรมาส, อัตราส่วนทางการเงิน

## 🚀 การติดตั้ง

### ข้อกำหนดระบบ
- Python 3.11+
- CUDA-compatible GPU (แนะนำ)
- MySQL Database
- 8GB+ RAM

### 1. Clone Repository
```bash
git clone <repository-url>
cd "AI and API Stock Project"
```

### 2. สร้าง Environment
```bash
# สร้าง conda environment
conda env create -f environment.yml
conda activate pytorch

# ติดตั้ง TA-Lib (สำหรับ Technical Indicators)
pip install TA_Lib-0.4.28-cp312-cp312-win_amd64.whl
```

### 3. ตั้งค่า Database
```bash
# สร้างไฟล์ config.env ใน Preproces/
DB_HOST=127.0.0.1
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=TradeMine
DB_PORT=3306
FMP_API_KEY=your_fmp_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

## 🎯 การใช้งาน

### การเทรนโมเดล
```bash
# เทรน LSTM model
cd LSTM_model
python LSTM_model.py

# เทรน GRU model
cd GRU_Model
python GRU_model.py
```

### การทำนายราคา
```bash
# รันระบบทำนาย (จะตรวจสอบการ retrain อัตโนมัติ)
cd Preproces
python Autotrainmodel.py
```

### การรัน API Server
```bash
# Paper Trading API
cd API_Mobile_Web
python API_Papertrade.py

# Node.js Server (สำหรับ Mobile App)
npm install
node server.js
```

## 📊 หุ้นที่รองรับ

### 🇺🇸 หุ้นอเมริกา (10 ตัว)
- **Tech Giants**: AAPL, NVDA, MSFT, AMZN, GOOGL, META
- **Growth Stocks**: TSLA, AVGO, AMD
- **Semiconductor**: TSM

### 🇹🇭 หุ้นไทย (9 ตัว)
- **Telecom**: ADVANC, TRUE
- **Tech**: DITTO, INET, INSET, HUMAN
- **Consumer**: JMART
- **Services**: DIF, JAS

## 🔧 การทำงานของระบบ

### 1. โหมดการทำงาน
- **Prediction-Only**: ใช้โมเดลที่เทรนแล้ว (ถ้า retrain ล่าสุด < 3 วัน)
- **Retrain Mode**: เทรนโมเดลใหม่ด้วยข้อมูลล่าสุด (ถ้า retrain ล่าสุด ≥ 3 วัน)

### 2. Walk-Forward Validation
- **Mini-Retrain Frequency**: ทุก 3 วันในข้อมูล
- **Optimal Parameters**: 
  - LSTM: chunk_size=100, units=48-24, lr=1.70e-04
  - GRU: chunk_size=100, units=48-24, lr=1.20e-04

### 3. Ensemble Learning
- **LSTM + GRU**: ทำนายราคาและทิศทางแยกกัน
- **XGBoost**: รวมผลทำนายและปรับปรุงคุณภาพ
- **Consistency Rate**: ติดตามความสอดคล้องของโมเดล (~89.5%)

## 📁 โครงสร้างโปรเจค

```
AI and API Stock Project/
├── 🧠 LSTM_model/                  # LSTM Neural Network
├── 🧠 GRU_Model/                   # GRU Neural Network  
├── 🎯 Ensemble_Model/              # XGBoost Ensemble
├── 🔧 Preproces/                   # Data Processing & Main System
│   ├── usa/                        # US Stock Data Pipeline
│   ├── thai/                       # Thai Stock Data Pipeline
│   └── Autotrainmodel.py          # Main Prediction System
├── 📱 API_Mobile_Web/              # API Server & Mobile App
├── 📊 All_Auto/                    # Data Collection Automation
└── 🌐 stock_react/                 # React Frontend (Optional)
```

## 🚨 ข้อควรระวัง

### 1. Scaler Consistency
- **สำคัญมาก**: การใช้ scaler ที่ไม่ตรงกันระหว่าง training และ prediction จะทำให้ราคาทำนายผิดมาก
- Scaler ถูกเทรนด้วย `TargetPrice = Close.shift(-1)` (ราคาวันถัดไป)
- ต้องใช้ scaler เดียวกันจาก `ticker_scalers.pkl`

### 2. การ Retrain
- ระบบจะ retrain อัตโนมัติเมื่อถึงเวลา
- ไฟล์ `last_retrain_model.txt` เก็บวันที่ retrain ล่าสุด
- การ retrain ใช้เวลา ~2-4 ชั่วโมง (ขึ้นอยู่กับ GPU)

### 3. ข้อมูลและ API
- ต้อง API keys สำหรับดึงข้อมูลหุ้นและข่าวสาร
- Database ต้องมีข้อมูลอย่างน้อย 1 ปี สำหรับการเทรน
- การอัพเดทข้อมูลควรทำทุกวัน

## 📈 ผลการทำงาน

### ประสิทธิภาพโมเดล
- **Direction Accuracy**: ~72% (ทิศทางการเคลื่อนไหว)
- **Model Consistency**: ~89.5% (ความสอดคล้องระหว่างโมเดล)
- **Coverage**: 100% (ครอบคลุมหุ้นทุกตัว)

### ตัวอย่างการทำนาย
```
📈 XGBoost Final Predictions:
   AAPL: $205.94 (UP 100.0%)
   NVDA: $189.42 (UP 100.0%) 
   MSFT: $242.02 (UP 100.0%)
   AMZN: $218.53 (UP 100.0%)
   GOOGL: $192.46 (UP 100.0%)
```

## 🛠️ การแก้ไขปัญหา

### ปัญหาที่พบบ่อย
1. **ราคาทำนายผิดปกติ**: ปกติเกิดจาก scaler mismatch → ต้อง retrain โมเดล
2. **TensorFlow Warnings**: เป็นเรื่องปกติ ไม่กระทบการทำงาน
3. **Database Connection Error**: ตรวจสอบ MySQL และ `config.env`
4. **Missing Dependencies**: ติดตั้งตาม `environment.yml`

### คำสั่งตรวจสอบ
```bash
# ตรวจสอบไฟล์โมเดล
python -c "import os; print('LSTM Model:', 'Found' if os.path.exists('LSTM_model/best_hypertuned_model.keras') else 'Missing')"

# ตรวจสอบ scalers
python -c "import joblib; scalers = joblib.load('LSTM_model/ticker_scalers.pkl'); print('Scalers:', len(scalers))"
```

## 📞 การสนับสนุน

- **Documentation**: ดู `CLAUDE.md` สำหรับรายละเอียดเทคนิค
- **Logs**: ตรวจสอบไฟล์ log ใน `retrain_logs/` และ `training.log`
- **Issues**: หากพบปัญหาให้ตรวจสอบ log files ก่อน

## 📄 License

โปรเจคนี้สำหรับการศึกษาและการใช้งานส่วนตัว

---

*🤖 Built with TensorFlow, PyTorch, XGBoost, and ❤️*