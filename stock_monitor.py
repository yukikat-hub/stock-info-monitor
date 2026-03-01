import requests
import feedparser
import json
import os
import time
import urllib.parse
from datetime import datetime
from bs4 import BeautifulSoup

CONFIG_FILE = 'stocks_config.json'
STATE_FILE = 'last_notified.json'

def load_config():
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def fetch_google_news(ticker, name):
    # Google News RSS for the specific stock
    query = f"{ticker} {name}"
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ja&gl=JP&ceid=JP:ja"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries:
        articles.append({
            'id': entry.id,
            'title': entry.title,
            'link': entry.link,
            'published': entry.published,
            'source': 'Google News'
        })
    return articles

def fetch_tdnet(ticker):
    # TDnet (適時開示) - Searching for recent disclosures for the ticker
    url = f"https://www.release.tdnet.info/inbs/I_main_00.html" # Recent disclosures page
    articles = []
    try:
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # TDnet listing items are usually in a table
        rows = soup.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 5:
                ticker_col = cols[1].text.strip()
                if ticker in ticker_col:
                    time_str = cols[0].text.strip()
                    name_col = cols[2].text.strip()
                    title_col = cols[3].text.strip()
                    link_tag = cols[3].find('a')
                    
                    if link_tag:
                        pdf_link = "https://www.release.tdnet.info/inbs/" + link_tag.get('href')
                        articles.append({
                            'id': f"tdnet_{ticker}_{time_str}_{title_col}",
                            'title': f"【適時開示】{title_col}",
                            'link': pdf_link,
                            'published': f"{datetime.now().strftime('%Y-%m-%d')} {time_str}",
                            'source': 'TDnet'
                        })
    except Exception as e:
        print(f"Error fetching TDnet for {ticker}: {e}")
    
    return articles

def send_discord_notification(webhook_url, article):
    if not webhook_url or "YOUR_DISCORD" in webhook_url:
        print(f"Skipping notification (No Webhook): {article['title']}")
        return

    data = {
        "embeds": [{
            "title": article['title'],
            "url": article['link'],
            "description": f"Source: {article['source']}\nPublished: {article['published']}",
            "color": 3447003
        }]
    }
    response = requests.post(webhook_url, json=data)
    if response.status_code != 204:
        print(f"Failed to send notification: {response.status_code}")

def main():
    config = load_config()
    state = load_state()
    # Use environment variable for webhook if available
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL') or config.get('discord_webhook_url')
    
    new_state = state.copy()
    
    for stock in config['stocks']:
        ticker = stock['ticker']
        name = stock['name']
        print(f"Checking news for {ticker} {name}...")
        
        # Fetch Google News
        news_articles = fetch_google_news(ticker, name)
        # Fetch TDnet
        tdnet_articles = fetch_tdnet(ticker)
        
        all_articles = news_articles + tdnet_articles
        
        # Filter and notify
        for article in all_articles:
            article_id = article['id']
            title = article['title']
            
            # Check if already notified
            if article_id not in state.get(ticker, []):
                # Filter by keywords for general news (TDnet is always notified)
                is_relevant = article['source'] == 'TDnet'
                if not is_relevant:
                    for kw in config.get('keywords', []):
                        if kw in title:
                            is_relevant = True
                            break
                
                if is_relevant:
                    send_discord_notification(webhook_url, article)
                    if ticker not in new_state:
                        new_state[ticker] = []
                    new_state[ticker].append(article_id)
                
        # Keep only last 50 IDs to avoid bloat (inside the ticker loop)
        if ticker in new_state and len(new_state[ticker]) > 50:
            new_state[ticker] = new_state[ticker][-50:]

    save_state(new_state)

if __name__ == "__main__":
    main()
