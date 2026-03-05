import requests
import feedparser
import json
import os
import time
import urllib.parse
import argparse
import google.generativeai as genai
from datetime import datetime, timedelta
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

def fetch_google_news(ticker, name, days=None):
    # Google News RSS for the specific stock
    query = f"{ticker} {name}"
    if days:
        query += f" when:{days}d"
    
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

def summarize_with_gemini(api_key, ticker, name, articles):
    if not api_key:
        return "Gemini APIキーが設定されていないため、要約をスキップしました。"
    
    if not articles:
        return "今週の新しいニュースはありませんでした。"

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    titles = [a['title'] for a in articles]
    prompt = f"""
以下の銘柄に関する直近1週間のニュースタイトルのリストを読み取り、重要なトピックを最大3行で要約してください。
今後の投資判断に役立つ「注目ポイント」や「懸念点」があればそれも含めてください。

銘柄: {ticker} {name}
ニュースリスト:
{chr(10).join(titles)}

要約ルール:
- 箇条書きで3行以内。
- 具体的かつ冷静なトーンで。
- まったくニュースがない場合は「特筆すべきニュースはありませんでした」と回答。
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"要約の生成中にエラーが発生しました: {e}"

def send_discord_notification(webhook_url, content, is_embed=True):
    if not webhook_url or "YOUR_DISCORD" in webhook_url:
        print(f"Skipping notification (No Webhook): {content[:50]}...")
        return

    if is_embed:
        data = {"embeds": [content] if isinstance(content, dict) else [content]}
    else:
        data = {"content": content}

    response = requests.post(webhook_url, json=data)
    if response.status_code != 204:
        print(f"Failed to send notification: {response.status_code}")

def run_normal_mode(config, state, webhook_url):
    new_state = state.copy()
    for stock in config['stocks']:
        ticker = stock['ticker']
        name = stock['name']
        print(f"Checking news for {ticker} {name}...")
        
        all_articles = fetch_google_news(ticker, name) + fetch_tdnet(ticker)
        
        for article in all_articles:
            if article['id'] not in state.get(ticker, []):
                is_relevant = article['source'] == 'TDnet'
                if not is_relevant:
                    for kw in config.get('keywords', []):
                        if kw in article['title']:
                            is_relevant = True
                            break
                
                if is_relevant:
                    embed = {
                        "title": article['title'],
                        "url": article['link'],
                        "description": f"Source: {article['source']}\nPublished: {article['published']}",
                        "color": 3447003
                    }
                    send_discord_notification(webhook_url, embed)
                    if ticker not in new_state:
                        new_state[ticker] = []
                    new_state[ticker].append(article['id'])
        
        if ticker in new_state and len(new_state[ticker]) > 50:
            new_state[ticker] = new_state[ticker][-50:]
    save_state(new_state)

def run_summary_mode(config, webhook_url, gemini_api_key):
    today = datetime.now()
    is_monday = today.weekday() == 0
    
    summary_msg = f"## 📊 AI マーケット・ダイジェスト ({today.strftime('%Y/%m/%d')})\n"
    has_any_content = False
    
    for stock in config['stocks']:
        ticker = stock['ticker']
        name = stock['name']
        freq = stock.get('summary_frequency', 'weekly')
        
        days_to_fetch = 0
        label = ""
        
        if freq == 'daily':
            days_to_fetch = 1
            label = "【本日分】"
        elif freq == 'weekly' and is_monday:
            days_to_fetch = 7
            label = "【今週分】"
        
        if days_to_fetch > 0:
            print(f"Summarizing {freq} news for {ticker} {name}...")
            articles = fetch_google_news(ticker, name, days=days_to_fetch)
            if articles:
                summary = summarize_with_gemini(gemini_api_key, ticker, name, articles)
                summary_msg += f"**{label} {ticker} {name}**\n{summary}\n\n"
                has_any_content = True
            else:
                # Optional: mention no news for daily if you want
                pass
    
    if has_any_content:
        send_discord_notification(webhook_url, summary_msg, is_embed=False)
    else:
        print("No summaries to send today.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', action='store_true', help='Run in weekly summary mode')
    args = parser.parse_args()

    config = load_config()
    state = load_state()
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL') or config.get('discord_webhook_url')
    gemini_api_key = os.environ.get('GEMINI_API_KEY')

    if args.summary:
        run_summary_mode(config, webhook_url, gemini_api_key)
    else:
        run_normal_mode(config, state, webhook_url)

if __name__ == "__main__":
    main()
