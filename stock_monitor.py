import requests
import feedparser
import json
import os
import time
import urllib.parse
import argparse
from google import genai
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

def fetch_google_news(ticker, name, days=None):
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
    url = "https://www.release.tdnet.info/inbs/I_main_00.html"
    articles = []
    try:
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
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

def summarize_batch(api_key, stock_data_list):
    """Summarize multiple stocks in one prompt to save API quota."""
    if not api_key:
        return "Gemini APIキーが設定されていません。"

    client = genai.Client(api_key=api_key)
    
    combined_input = ""
    for item in stock_data_list:
        titles = [a['title'] for a in item['articles']]
        if not titles:
            continue
        combined_input += f"■ {item['label']} {item['ticker']} {item['name']}:\n"
        combined_input += "\n".join(titles) + "\n\n"

    if not combined_input:
        return None

    prompt = f"""
以下の銘柄リストについて、各銘柄ごとに直近の重要トピックを最大3行ずつ要約してください。
投資判断に役立つ要素（業績予想、提携、新技術、市場動向など）を優先してください。

入力データ:
{combined_input}

出力形式ルール:
- 各銘柄の冒頭に「**【銘柄名】**」を付けてください。
- 各銘柄3行以内の箇条書き。
- 冷静かつ客観的なトーン。
"""

    try:
        # Using Gemini 1.5 Flash as it is the most widely available free tier model
        print(f"Calling Gemini API with model: gemini-1.5-flash...")
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        if not response or not response.text:
            return "AIからの回答が空でした。内容が制限（セーフティフィルター）に抵触した可能性があります。"
        return response.text.strip()
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            return "Gemini APIの無料枠制限（429 Error）に達しました。しばらく時間をおいてから実行するか、APIキーの設定（AI Studio）を確認してください。"
        return f"AI要約の生成中にエラーが発生しました: {error_msg}"

def send_discord_notification(webhook_url, content, is_embed=True):
    if not webhook_url or "YOUR_DISCORD" in webhook_url:
        print(f"Skipping notification: {content[:100]}...")
        return

    if is_embed:
        data = {"embeds": [content] if isinstance(content, dict) else [content]}
    else:
        # Split message if it's too long for Discord (2000 chars)
        if len(content) > 1900:
            for i in range(0, len(content), 1900):
                send_discord_notification(webhook_url, content[i:i+1900], is_embed=False)
            return
        data = {"content": content}

    response = requests.post(webhook_url, json=data)
    if response.status_code != 204:
        print(f"Failed to send notification: {response.status_code}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()

    config = load_config()
    state = load_state()
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL') or config.get('discord_webhook_url')
    gemini_api_key = os.environ.get('GEMINI_API_KEY')

    if args.summary:
        today = datetime.now()
        is_monday = today.weekday() == 0
        stock_data_to_summarize = []
        
        for stock in config['stocks']:
            ticker = stock['ticker']
            name = stock['name']
            freq = stock.get('summary_frequency', 'weekly')
            
            days = 0
            label = ""
            if freq == 'daily':
                days, label = 1, "本日分"
            elif freq == 'weekly' and is_monday:
                days, label = 7, "今週分"
            
            if days > 0:
                articles = fetch_google_news(ticker, name, days=days)
                if articles:
                    stock_data_to_summarize.append({
                        'ticker': ticker, 'name': name, 'articles': articles, 'label': label
                    })

        if stock_data_to_summarize:
            print(f"Generating summary for {len(stock_data_to_summarize)} stocks...")
            full_summary = summarize_batch(gemini_api_key, stock_data_to_summarize)
            if full_summary:
                header = f"## 📊 AI マーケット・ダイジェスト ({today.strftime('%Y/%m/%d')})\n\n"
                send_discord_notification(webhook_url, header + full_summary, is_embed=False)
        else:
            print("No news to summarize today.")

    else:
        # Normal Alert Mode
        new_state = state.copy()
        for stock in config['stocks']:
            ticker, name = stock['ticker'], stock['name']
            print(f"Checking alerts for {ticker} {name}...")
            all_articles = fetch_google_news(ticker, name) + fetch_tdnet(ticker)
            for article in all_articles:
                if article['id'] not in state.get(ticker, []):
                    is_relevant = (article['source'] == 'TDnet') or any(kw in article['title'] for kw in config.get('keywords', []))
                    if is_relevant:
                        embed = {
                            "title": article['title'], "url": article['link'],
                            "description": f"Source: {article['source']}\nPublished: {article['published']}",
                            "color": 3447003
                        }
                        send_discord_notification(webhook_url, embed)
                        new_state.setdefault(ticker, []).append(article['id'])
            if ticker in new_state:
                new_state[ticker] = new_state[ticker][-50:]
        save_state(new_state)

if __name__ == "__main__":
    main()
