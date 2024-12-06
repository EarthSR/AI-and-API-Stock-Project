from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
from bs4 import BeautifulSoup
import pandas as pd
import urllib.parse

def scrape_bangkok_post_selenium(query):
    # Setup Chrome options
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--ignore-certificate-errors')  # Ignore SSL certificate errors
    #options.add_argument('--headless')  # Optional: Use headless mode for faster execution
    options.add_argument('--disable-software-rasterizer')  # Optional: GPU issue fix

    # Initialize Chrome driver with options
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # Initial search URL
    base_url = f'https://search.bangkokpost.com/search/result?q={query}&category=news'
    driver.get(base_url)

    news_data = []

    try:
        # Loop to handle pagination
        while True:
            # Wait for the search results to load and the relevant element to appear
            WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'mk-listnew--title'))  # The class showing search results
            )

            # Scroll down the page to load more articles
            for _ in range(5):
                driver.execute_script("window.scrollBy(0, 1000);")
                time.sleep(2)

            # Get the HTML of the page after it has loaded
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Find articles based on the class mk-listnew--title
            articles = soup.find_all('div', class_='mk-listnew--title')

            if not articles:
                print("No articles found.")
                break

            for article in articles:
                # Extract the title and link from the <h3><a> tag
                title_tag = article.find('h3').find('a')
                title = title_tag.get_text(strip=True)
                link = title_tag['href']
                
                # If the link is a redirect, extract the actual link from the 'href' parameter
                if 'track/visitAndRedirect' in link:
                    url_params = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
                    real_link = url_params['href'][0]
                else:
                    real_link = link

                # Visit the actual article link to scrape detailed content
                driver.get(real_link)
                time.sleep(3)  # Wait for the page to load

                # Use BeautifulSoup to parse the article page
                page_soup = BeautifulSoup(driver.page_source, 'html.parser')

                # Extract the publication date of the article
                date = page_soup.find('div', class_='article-info--col')
                if date:
                    date = date.find_next('p').get_text(strip=True)
                else:
                    date = 'Date not found'

                # **Extract the main article content** from the appropriate section
                content = page_soup.find('div', class_='article-content')  # Adjust the class name if necessary
                if content:
                    # Extract and join all <p> tags (paragraphs)
                    paragraphs = content.find_all('p')
                    full_content = '\n'.join([p.get_text(strip=True) for p in paragraphs])
                else:
                    full_content = 'Content not found'

                # Store the extracted data in a list
                news_data.append({
                    "Title": title,
                    "Link": real_link,
                    "Published Date": date,
                    "Content": full_content
                })

            # Get the current URL after page loading to find "Next"
            current_url = driver.current_url
            print(f"Current URL: {current_url}")  # Debugging the current URL
            
            # Try to find the "Next" button by checking the pagination section
            try:
                # Locate the "Next" button link using a more general XPath or partial link text
                next_button = WebDriverWait(driver, 20).until(
                    EC.element_to_be_clickable((By.XPATH, "//li/a[text()='Next']"))
                )

                # Click the "Next" button
                next_button.click()
                time.sleep(3)  # Wait for the next page to load

            except Exception as e:
                print("No more pages or failed to click 'Next':", e)
                break

        # Close the browser
        driver.quit()

        # Return the collected data as a DataFrame
        return pd.DataFrame(news_data)

    except Exception as e:
        print(f"Error occurred: {e}")
        driver.quit()
        return pd.DataFrame()

# Example usage of the function
query = 'Stock'
news_df = scrape_bangkok_post_selenium(query)

if not news_df.empty:
    print(news_df)
    news_df.to_csv('bangkok_post_news_with_details.csv', index=False)
else:
    print("No news data found.")
