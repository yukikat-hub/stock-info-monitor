import requests
import feedparser
import json
import os
import time
import urllib.parse
import argparse
from google import genai
from google.genai import types
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
        summary = entry.get('summary', '') or entry.get('description', '')
        articles.append({
            'id': entry.id,
            'title': entry.title,
            'link': entry.link,
            'published': entry.published,
            'snippet': summary[:300],
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

def summarize_batch_with_retry(api_key, stock_data_list):
    """Summarize a small batch of stocks with retry and safety handling."""
    if not api_key:
        return "Gemini APIキーが設定されていません。"

    # Use stable v1
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

    # 証券アドバイスとしての判定を避けるため、役割を「高度な情報整理アシスタント」に修正
    prompt = f"""
あなたは高度なビジネス情報整理アシスタントです。
以下の銘柄ニュースとステータスに基づき、公開されている事実から「特に注目すべき変化」と「市場の一般的な反応」を客観的に整理してください。

【整理のポイント】
1. **事業上の重要変化**: 業績、提携、新技術など、企業の将来価値に影響しそうな事実。
2. **市場のコンセンサス**: 報道やデータから読み取れる、現在の市場参加者の一般的な見方や反応。
3. **客観的な事実関係**: 関連するマクロ環境（市場全体やセクターの動向）との紐付け。
4. **保有状況の整理**: 保有者の現在の損益状況を、事実として現在の市場価格と比較して整理（あくまでデータ整理として）。

※注意: 個別の売買の推奨（Buy/Sellなど）は行わないでください。事実に基づいた情報の「示唆」に留めてください。

【対象銘柄データ】
{combined_input}

【出力形式】
- 各銘柄「**【銘柄名 (ティッカー)】**」を見出しにする。
- 各銘柄、事実に基づいた深い洞察を200〜300文字程度で。
"""

    models_to_try = [
        'models/gemini-1.5-flash', 
        'models/gemini-2.0-flash', 
        'models/gemini-2.0-flash-lite',
        'models/gemini-1.5-pro'
    ]
    
    # 制限をさらに緩和
    safety_settings = [
        types.SafetySetting(category='HATE_SPEECH', threshold='BLOCK_NONE'),
        types.SafetySetting(category='HARASSMENT', threshold='BLOCK_NONE'),
        types.SafetySetting(category='SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
        types.SafetySetting(category='DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
        types.SafetySetting(category='CIVIC_INTEGRITY', threshold='BLOCK_NONE'),
    ]

    for model_id in models_to_try:
        try:
            print(f"Trying Gemini model: {model_id} for batch...")
            response = client.models.generate_content(
                model=model_id, 
                contents=prompt,
                config=types.GenerateContentConfig(safety_settings=safety_settings, temperature=0.3)
            )
            if response and response.text:
                return response.text.strip()
            else:
                # ブロック理由の詳細を取得
                candidates = getattr(response, 'candidates', [])
                reason = "不明"
                if candidates:
                    reason = getattr(candidates[0], 'finish_reason', 'Unknown')
                print(f"Model {model_id} blocked or returned empty. Reason: {reason}")
        except Exception as e:
            print(f"Model {model_id} error: {str(e)[:150]}")
            continue
    return f"要約生成に失敗しました（AI側の制限に抵触した可能性があります）。"

def send_discord_notification(webhook_url, content, is_embed=True):
    if not webhook_url or "YOUR_DISCORD" in webhook_url:
        print(f"Skipping notification: {content[:100]}...")
        return

    if is_embed:
        data = {"embeds": [content] if isinstance(content, dict) else [content]}
    else:
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
    portfolio = load_json(PORTFOLIO_FILE)
    
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL') or config.get('discord_webhook_url')
    gemini_api_key = os.environ.get('GEMINI_API_KEY')

    portfolio_map = {item['ticker_code']: item for item in portfolio if 'ticker_code' in item}

    if args.summary:
        today = datetime.now()
        is_monday = today.weekday() == 0
        stock_data_list = []
        
        print("Gathering news for summary...")
        for stock in config['stocks']:
            ticker = stock['ticker']
            name = stock['name']
            
            # If manually triggered (or periodic), we summarize anyway
            articles = fetch_google_news(ticker, name, days=7) # Past week news
            if not articles:
                articles = fetch_google_news(ticker, name, days=14)[:2] # Older fallback
            
            stock_data_list.append({
                'ticker': ticker, 'name': name, 'articles': articles, 
                'label': "【マーケット分析】", 'portfolio': portfolio_map.get(ticker)
            })

        if stock_data_list:
            # Batch processing: 3 stocks per request to avoid safety/token issues
            batch_size = 3
            summaries = []
            for i in range(0, len(stock_data_list), batch_size):
                batch = stock_data_list[i : i + batch_size]
                print(f"Processing batch {i//batch_size + 1} ({len(batch)} stocks)...")
                batch_summary = summarize_batch_with_retry(gemini_api_key, batch)
                summaries.append(batch_summary)
                time.sleep(1) # Tiny sleep to avoid RPM spikes

            full_summary = "\n\n".join(summaries)
            header = f"## 💎 AI 投資戦略サマリー ({today.strftime('%Y/%m/%d')})\n\n"
            send_discord_notification(webhook_url, header + full_summary, is_embed=False)
        else:
            print("No news to summarize.")

    else:
        # Alert Mode
        new_state = state.copy()
        for stock in config['stocks']:
            ticker, name = stock['ticker'], stock['name']
            print(f"Checking {ticker}...")
            all_articles = fetch_google_news(ticker, name) + fetch_tdnet(ticker)
            for article in all_articles:
                if article['id'] not in state.get(ticker, []):
                    is_relevant = (article['source'] == 'TDnet') or any(kw in article['title'] for kw in config.get('keywords', []))
                    if is_relevant:
                        send_discord_notification(webhook_url, {
                            "title": article['title'], "url": article['link'],
                            "description": f"Source: {article['source']}\nPublished: {article['published']}",
                            "color": 3447003
                        })
                        if ticker not in new_state: new_state[ticker] = []
                        new_state[ticker].append(article['id'])
            if ticker in new_state: new_state[ticker] = new_state[ticker][-50:]
        save_json(STATE_FILE, new_state)

if __name__ == "__main__":
    main()
