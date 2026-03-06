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
    return [] if filepath == PORTFOLIO_FILE else {}

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

    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
    
    combined_input = ""
    for item in stock_data_list:
        combined_input += f"■ 【{item['ticker']}】{item['name']}\n"
        if item.get('portfolio'):
            p = item['portfolio']
            combined_input += f"（保有状況: {p.get('quantity', 0)}株, 平均取得単価: {p.get('average_acquisition_price', 0)}, 現在価値: {p.get('market_value', 0)}, 前日比損益: {p.get('unrealized_gain_loss', 0)}）\n"
        
        for a in item['articles']:
            combined_input += f"- {a['title']} ({a['source']})\n"
            if a.get('snippet') and len(a['snippet']) > 20:
                combined_input += f"  内容概要: {a['snippet'][:150]}...\n"
        combined_input += "\n"

    if not combined_input:
        return None

    # 投資アドバイスと判定されないよう、役割を「高度な情報収集・整理アシスタント」に調整
    prompt = f"""
あなたは高度なビジネス情報整理アシスタントです。
以下のニュースとデータに基づき、市場の事実関係を客観的に要約・整理してください。

【整理の要件】
1. **事実の抽出**: ニュースが企業の事業や収益性に与える具体的な事実関係。
2. **市場のコンセンサス**: 報道等から読み取れる一般的な市場の反応や期待値。
3. **データ整理**: 保有銘柄の状況（規模・損益）に応じた情報の優先順位付け。

※注意: 本内容は売買を推奨するものではなく、公開情報の整理を目的としています。

【対象データ】
{combined_input}
"""

    # 以前成功したロジックに基づき、動的な発見とprefix試行を復活
    discovered_models = []
    try:
        for m in client.models.list():
            discovered_models.append(m.name)
    except: pass

    candidates = ['models/gemini-1.5-flash', 'gemini-1.5-flash', 'models/gemini-2.0-flash', 'gemini-2.0-flash']
    for m in discovered_models:
        if m not in candidates: candidates.append(m)

    safety_settings = [
        types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
        types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
        types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
        types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
    ]

    errors = []
    for model_id in candidates:
        try:
            print(f"Trying {model_id}...")
            response = client.models.generate_content(
                model=model_id, 
                contents=prompt,
                config=types.GenerateContentConfig(safety_settings=safety_settings, temperature=0.3)
            )
            if response and response.text:
                return response.text.strip()
            else:
                reason = getattr(response, 'candidates', [{}])[0].get('finish_reason', 'Unknown/Blocked')
                errors.append(f"{model_id}(Block:{reason})")
        except Exception as e:
            errors.append(f"{model_id}({str(e)[:40]})")
            continue

    # v1beta Fallback
    try:
        beta_client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
        response = beta_client.models.generate_content(model='gemini-1.5-flash', contents=prompt)
        if response and response.text: return response.text.strip()
    except: pass

    return f"⚠️ 要約生成に失敗しました。\n詳細ログ: {', '.join(errors[:3])}"

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
    except Exception as e:
        print(f"Error sending to Discord: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()

    config = load_json(CONFIG_FILE)
    if isinstance(config, list): config = {"stocks": [], "keywords": []} # Failsafe
    state = load_json(STATE_FILE)
    portfolio = load_json(PORTFOLIO_FILE)
    
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL') or config.get('discord_webhook_url')
    gemini_api_key = os.environ.get('GEMINI_API_KEY')

    portfolio_map = {item['ticker_code']: item for item in portfolio if 'ticker_code' in item}
    config_map = {s['ticker']: s for s in config.get('stocks', [])}

    if args.summary:
        today = datetime.now()
        
        # Combine all tickers (from config and portfolio)
        all_tickers = set(portfolio_map.keys()) | set(config_map.keys())
        
        stock_data_list = []
        print(f"Gathering data for {len(all_tickers)} stocks...")
        
        for ticker in all_tickers:
            p_data = portfolio_map.get(ticker)
            s_config = config_map.get(ticker)
            
            name = (p_data.get('name') if p_data else None) or (s_config.get('name') if s_config else None) or ticker
            market_value = p_data.get('market_value', 0) if p_data else -1
            
            # Decide fetch range based on summary_frequency (Daily=1day, Weekly=7days)
            freq = s_config.get('summary_frequency', 'weekly') if s_config else 'weekly'
            fetch_days = 2 if freq == 'daily' else 7 # Use 2 days for daily to catch weekend/late news if needed, or 1 for strictness
            
            print(f"Fetching {fetch_days} days of news for {name} ({ticker})...")
            articles = fetch_google_news(ticker, name, days=fetch_days)
            
            stock_data_list.append({
                'ticker': ticker, 
                'name': name, 
                'articles': articles, 
                'market_value': market_value,
                'portfolio': p_data
            })

        # Sort by market value (descending)
        stock_data_list.sort(key=lambda x: x['market_value'], reverse=True)

        if stock_data_list:
            batch_size = 3
            summaries = []
            for i in range(0, len(stock_data_list), batch_size):
                batch = stock_data_list[i : i + batch_size]
                print(f"Processing batch {i//batch_size + 1}...")
                batch_summary = summarize_batch_with_retry(gemini_api_key, batch)
                summaries.append(batch_summary)
                time.sleep(1)

            full_summary = "\n\n".join(summaries)
            header = f"## 💎 AI 投資戦略サマリー ({today.strftime('%Y/%m/%d')})\n\n"
            send_discord_notification(webhook_url, header + full_summary, is_embed=False)
        else:
            print("No data to summarize.")

    else:
        # Alert Mode
        new_state = state.copy()
        for s in config.get('stocks', []):
            ticker, name = s['ticker'], s['name']
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
