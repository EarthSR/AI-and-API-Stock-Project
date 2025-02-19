import requests
import csv

params = {
    "q": "stock technology",
    "engine": "google_news",
    "gl": "us",
    "hl": "en",
    "from": "2018-01-01",
    "api_key": "046bbebf837acffb1f766fb48a57a00bf416c11a169b61b84293266eaf1ef110"
}

url = "https://serpapi.com/search.json"

response = requests.get(url, params=params)
results = response.json()

# ตรวจสอบว่ามีข่าวใน "news_results" หรือไม่
if "news_results" in results:
    news_results = results["news_results"]
    
    # สร้างไฟล์ CSV
    with open('google_news_results.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["source","title", "link", "date", "description"])
        writer.writeheader()
        
        # เขียนข้อมูลข่าวลงใน CSV
        for article in news_results:
            writer.writerow({
                "source": article.get("source", ""),
                "title": article.get("title", ""),
                "link": article.get("link", ""),
                "date": article.get("date", ""),
                "description": article.get("description", "")
            })

    print("Data has been written to google_news_results.csv")
else:
    print("No news results found.")
