import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def get_articles_from_source(source):
    all_articles = []
    page = 1 
    limit = 50 
    while True:
        url = f'https://api.finlight.me/v1/articles/extended?query=apple&source={source}&from=2024-01-01&to=2024-12-31&limit={limit}&page={page}'
        headers = {
            'accept': 'application/json',
            'X-API-KEY': 'sk_f78dc1f2565f181ef1122b9a98d68e25d1de9155988d2001a2039880bc4a64ce',
        }
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error fetching articles from {source}: {response.status_code}")
            break
        data = response.json()
        if isinstance(data, dict) and 'articles' in data:
            articles = data['articles']
            if not articles:
                break 
            all_articles.extend(articles)
        else:
            break
        page += 1 
    return all_articles

def get_financial_articles():
    sources = ["www.reuters.com", "www.yahoo.com", "finance.yahoo.com", "www.cnbc.com", "www.bbc.com"]
    all_articles = []
    folder_path = os.path.join(os.getcwd(), "articles_txt")
    os.makedirs(folder_path, exist_ok=True)
    for source in sources:
        print(f"Fetching articles from {source}...")
        articles = get_articles_from_source(source)
        for article in articles:
            title = article.get('title', '')
            if "apple" not in title.lower():
                continue  # Skip articles without "Apple" in the title (case-insensitive)
            all_articles.append(article)
            # Clean title for filename
            valid_title = title.replace('/', '_').replace('\\', '_').strip()
            file_path = os.path.join(folder_path, f"{valid_title}.txt")
            with open(file_path, "w", encoding="utf-8") as file:
                for key, value in article.items():
                    file.write(f"{key}: {value}\n\n")
            print(f"Saved: {file_path}")
    # Save CSV in the same folder as the Python notebook
    df = pd.DataFrame(all_articles)
    output_csv_path = os.path.join(os.getcwd(), "articles.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"\nSaved {len(df)} Apple-related articles to {output_csv_path}")
    return df

print(get_financial_articles())