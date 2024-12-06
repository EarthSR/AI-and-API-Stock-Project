import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datasets import Dataset
from sklearn.model_selection import train_test_split

# อ่านไฟล์ JSON สำหรับข้อมูลเทรน
print("Loading training data...")
train = pd.read_json("./News_Category_Dataset_v3.json", lines=True)

# อ่านไฟล์ CSV สำหรับข้อมูลทดสอบ
print("Loading test data...")
test = pd.read_csv("./American_stocks/imf_news_data/imf_news_full.csv")

# ตรวจสอบคอลัมน์ที่มีข้อมูลที่ต้องการ
print("Checking columns in datasets...")
print(f"Columns in train dataset: {train.columns}")
print(f"Columns in test dataset: {test.columns}")

# รวม 'short_description', 'headline', และ 'category' สำหรับข้อมูลเทรน
print("Preparing training text data...")
text_data_train = train['short_description'] + ' ' + train['category']

# แปลงข้อความจากข้อมูลเทรนเป็นเวกเตอร์ TF-IDF
print("Transforming training text data into TF-IDF vectors...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train = vectorizer.fit_transform(text_data_train)

# กำหนดจำนวนกลุ่ม (clusters) ที่ต้องการ
num_clusters = 5
print(f"Clustering the training data into {num_clusters} categories...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# จัดกลุ่มข้อความในข้อมูลเทรน
train['predicted_category'] = kmeans.fit_predict(X_train)

# คำนวณค่า Silhouette Score เพื่อประเมินผล
silhouette_avg = silhouette_score(X_train, train['predicted_category'])
print(f"Silhouette Score for training data: {silhouette_avg}")

# แสดงผลข้อมูลที่จัดกลุ่มจากข้อมูลเทรน
print("Sample of clustered training data:")
print(train[['headline', 'predicted_category']].head())

# ตรวจสอบคอลัมน์ที่ใช้ในข้อมูลทดสอบ
print("Preparing test data for clustering...")
text_data_test = test['title']  # หรือ test['content']

# แปลงข้อความจากข้อมูลทดสอบเป็นเวกเตอร์ TF-IDF
print("Transforming test data into TF-IDF vectors...")
X_test = vectorizer.transform(text_data_test)  # ใช้ transform แทน fit_transform สำหรับข้อมูลทดสอบ

# จัดกลุ่มข้อความในข้อมูลทดสอบ
print("Predicting clusters for test data...")
test['predicted_category'] = kmeans.predict(X_test)

# แสดงผลข้อมูลที่จัดกลุ่มจากข้อมูลทดสอบ
print("Sample of clustered test data:")
print(test[['title', 'predicted_category']].head())

# Export ข้อมูลที่มีคอลัมน์ 'predicted_category' ลงในไฟล์ CSV
print("Exporting training data to CSV...")
train.to_csv('train_grouped_news_data.csv', index=False)

print("Exporting test data to CSV...")
test.to_csv('test_grouped_news_data.csv', index=False)

# แสดงข้อความเมื่อทำการบันทึกเสร็จ
print("CSV files have been exported successfully.")
