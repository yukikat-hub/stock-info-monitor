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
PORTFOLIO_FILE = 'portfolio.json'

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def fetch_google_news(ticker, name, days=None):
    query = f"{ticker} {name}"
    if days:
        query += f" when:{days}d"
    
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ja&gl=JP&ceid=JP:ja"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries:
        # Include snippet if available for more context
        summary = entry.get('summary', '') or entry.get('description', '')
        articles.append({
            'id': entry.id,
            'title': entry.title,
            'link': entry.link,
            'published': entry.published,
            'snippet': summary[:300], # Keep snippet to save tokens but provide context
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
                            'source': 'TDnet',
                            'snippet': f"TDnet適時開示情報: {title_col}"
                        })
    except Exception as e:
        print(f"Error fetching TDnet for {ticker}: {e}")
    return articles

def summarize_batch(api_key, stock_data_list):
    """Summarize multiple stocks with deep investment insights."""
    if not api_key:
        return "Gemini APIキーが設定されていません。"

    # Use stable v1 to avoid 404
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
    
    combined_input = ""
    for item in stock_data_list:
        combined_input += f"■ 【{item['ticker']}】{item['name']}\n"
        if item.get('portfolio'):
            p = item['portfolio']
            combined_input += f"（保有状況: {p.get('quantity', 0)}株, 平均取得単価: {p.get('average_acquisition_price', 0)}, 前日比損益: {p.get('unrealized_gain_loss', 0)}）\n"
        
        for a in item['articles']:
            combined_input += f"- {a['title']} ({a['source']})\n"
            if a.get('snippet') and len(a['snippet']) > 20:
                combined_input += f"  内容概要: {a['snippet'][:150]}...\n"
        combined_input += "\n"

    if not combined_input:
        return None

    prompt = f"""
あなたは百戦錬磨のプロ投資アナリストです。
投資家の保有ポートフォリオの状態（損益状況）も踏まえ、提供されたニュース情報から「本当に投資判断に役立つ示唆」を抽出してください。

【タスク】
各銘柄について、投資判断（Hold / Buy / Sell / Watch）に資する分析を300文字程度で出力してください。

【分析の視点】
1. **カタリスト（株価を動かす材料）**: 今回のニュースは、中長期的な業績（売上・利益）に直結するか？
2. **市場の期待値と反応**: すでに織り込み済みか、それともサプライズか？機関投資家の動きはどう予想されるか？
3. **リスク要因**: ニュースの裏に隠れた懸念点や、マクロ環境（金利・為替）との関係性は？
4. **ポートフォリオに基づいた助言**: 保有者の損益状況（含み損・含み益）を考慮し、ナンピン、利益確定、損切り、静観などのアクションにつながる示唆。

【入力データ】
{combined_input}

【出力形式】
- 銘柄ごとに「**【銘柄名 (ティッカー)】**」を見出しにする。
- 単なる「ニュースの要約」ではなく、「なぜ注目すべきか」「投資家はどう動くべきか」という一歩踏み込んだ分析。
- 冷静、論理的、かつ投資家に寄り添った口調（日本語）。
"""

    # List of models with 'models/' prefix for better SDK compatibility
    models_to_try = ['models/gemini-2.0-flash', 'models/gemini-1.5-flash', 'models/gemini-1.5-pro']
    
    # Import safety settings from the SDK
    from google.genai import types
    safety_settings = [
        types.SafetySetting(category='HATE_SPEECH', threshold='BLOCK_NONE'),
        types.SafetySetting(category='HARASSMENT', threshold='BLOCK_NONE'),
        types.SafetySetting(category='SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
        types.SafetySetting(category='DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
    ]

    for model_id in models_to_try:
        try:
            print(f"Trying Gemini model: {model_id}...")
            response = client.models.generate_content(
                model=model_id, 
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=safety_settings,
                    temperature=0.7
                )
            )
            if response and response.text:
                return response.text.strip()
            else:
                # If text is empty, it might be blocked
                print(f"Model {model_id} returned empty text. Finish reason: {getattr(response, 'candidates', [{}])[0].get('finish_reason', 'Unknown')}")
        except Exception as e:
            print(f"Model {model_id} failed: {str(e)[:150]}...")
            continue
    return "全てのAIモデルで生成に失敗しました。一時的な制限か、内容が制限に抵触した可能性があります。時間をおいて再度お試しください。"

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

    try:
        response = requests.post(webhook_url, json=data)
        if response.status_code != 204:
            print(f"Failed to send notification: {response.status_code}")
    except Exception as e:
        print(f"Error sending to Discord: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()

    config = load_json(CONFIG_FILE)
    state = load_json(STATE_FILE)
    portfolio = load_json(PORTFOLIO_FILE) # Load portfolio data
    
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL') or config.get('discord_webhook_url')
    gemini_api_key = os.environ.get('GEMINI_API_KEY')

    # Create a lookup for portfolio items
    portfolio_map = {item['ticker_code']: item for item in portfolio if 'ticker_code' in item}

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
                days, label = 1, "【デイリー分析】"
            elif freq == 'weekly' and is_monday:
                days, label = 7, "【週刊レポート】"
            
            if days > 0:
                articles = fetch_google_news(ticker, name, days=days)
                # Fallback: if no recent news, get 1-2 generic entries to give the AI something to talk about
                if not articles:
                    articles = fetch_google_news(ticker, name, days=14)[:2]
                
                stock_data_to_summarize.append({
                    'ticker': ticker, 
                    'name': name, 
                    'articles': articles, 
                    'label': label,
                    'portfolio': portfolio_map.get(ticker)
                })

        if stock_data_to_summarize:
            print(f"Generating high-quality analysis for {len(stock_data_to_summarize)} stocks...")
            full_summary = summarize_batch(gemini_api_key, stock_data_to_summarize)
            if full_summary:
                header = f"## 💎 AI 投資戦略サマリー ({today.strftime('%Y/%m/%d')})\n\n"
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
                    # For normal alerts, we filter by keywords. TDnet is always notified.
                    is_relevant = (article['source'] == 'TDnet') or any(kw in article['title'] for kw in config.get('keywords', []))
                    if is_relevant:
                        embed = {
                            "title": article['title'], "url": article['link'],
                            "description": f"Source: {article['source']}\nPublished: {article['published']}",
                            "color": 3447003
                        }
                        send_discord_notification(webhook_url, embed)
                        if ticker not in new_state: new_state[ticker] = []
                        new_state[ticker].append(article['id'])
            if ticker in new_state:
                new_state[ticker] = new_state[ticker][-50:]
        save_json(STATE_FILE, new_state)

if __name__ == "__main__":
    main()
