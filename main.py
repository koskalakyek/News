"""
News Agent: fetches RSS feeds from a curated list of trusted sources
(Finance / Edge of Science / International Trade), filters new items,
asks Claude to pick the most important ones and summarize them in
Persian, then sends the result to a Telegram chat.

Designed to be run hourly (e.g. via GitHub Actions cron).
State (which links were already sent) is stored in seen_links.json
so the same article is never sent twice.
"""

import os
import json
import time
import requests
import feedparser
from datetime import datetime, timedelta, timezone

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
# gemini-3.5-flash: مدل فعلی رایگان گوگل (نسخه‌های 2.5 منقضی شده‌اند).
# می‌توانید با گذاشتن متغیر GEMINI_MODEL در workflow آن را عوض کنید
# (مثلاً gemini-3.5-flash-lite برای سهمیه بیشتر).
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

# How far back to look for "new" items on the very first run / if an
# item has no reliable publish date.
LOOKBACK_HOURS = int(os.environ.get("LOOKBACK_HOURS", "3"))

SEEN_FILE = "seen_links.json"

# Curated, high-trust sources grouped by topic.
# NOTE: Bloomberg, Ray Dalio's posts, WTO and UNCTAD do not publish
# reliable public RSS feeds, so they are intentionally left out here.
# See README for how to add sources manually if you find good feeds.
FEEDS = {
    "Finance": [
        ("The Economist - Finance", "https://www.economist.com/finance-and-economics/rss.xml"),
        ("Financial Times", "https://www.ft.com/world?format=rss"),
        ("Wall Street Journal - Markets", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
    ],
    "Edge of Science": [
        ("MIT Technology Review", "https://www.technologyreview.com/feed/"),
        ("Nature - Current", "https://www.nature.com/nature.rss"),
        ("Science (AAAS)", "https://www.science.org/rss/news_current.xml"),
        ("IEEE Spectrum", "https://spectrum.ieee.org/rss/fulltext"),
        ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
        ("Quanta Magazine", "https://api.quantamagazine.org/feed/"),
    ],
    "International Trade": [
        ("The Economist - Business", "https://www.economist.com/business/rss.xml"),
        ("Harvard Business Review", "https://hbr.org/feed"),
        ("Financial Times - World", "https://www.ft.com/world?format=rss"),
    ],
}


# ----------------------------------------------------------------------
# STATE HANDLING
# ----------------------------------------------------------------------

def load_seen():
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_seen(seen):
    # Keep the file from growing forever: cap at the most recent 2000 links.
    trimmed = list(seen)[-2000:]
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------
# FETCHING
# ----------------------------------------------------------------------

def parse_entry_time(entry):
    for key in ("published_parsed", "updated_parsed"):
        val = entry.get(key)
        if val:
            return datetime(*val[:6], tzinfo=timezone.utc)
    return None


def fetch_new_articles(seen):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    new_articles = []

    for topic, sources in FEEDS.items():
        for source_name, url in sources:
            try:
                feed = feedparser.parse(url)
            except Exception as e:
                print(f"[WARN] Failed to fetch {source_name}: {e}")
                continue

            for entry in feed.entries[:15]:
                link = entry.get("link")
                if not link or link in seen:
                    continue

                pub_time = parse_entry_time(entry)
                if pub_time and pub_time < cutoff:
                    continue  # too old

                new_articles.append({
                    "topic": topic,
                    "source": source_name,
                    "title": entry.get("title", "").strip(),
                    "summary": (entry.get("summary", "") or "")[:600],
                    "link": link,
                })
                seen.add(link)

    return new_articles


# ----------------------------------------------------------------------
# GEMINI (رایگان): SELECT + SUMMARIZE IN PERSIAN
# ----------------------------------------------------------------------

def summarize_with_gemini(articles):
    if not articles:
        return None

    articles_text = "\n\n".join(
        f"[{a['topic']}] منبع: {a['source']}\nعنوان: {a['title']}\n"
        f"توضیح: {a['summary']}\nلینک: {a['link']}"
        for a in articles
    )

    system_prompt = (
        "تو یک دستیار خبری دقیق و بی‌طرف هستی. وظیفه‌ات انتخاب مهم‌ترین و "
        "معتبرترین اخبار از فهرست داده‌شده و خلاصه‌سازی آن‌ها به فارسی روان است. "
        "قوانین مهم:\n"
        "1. فقط از مقالاتی که در فهرست داده شده استفاده کن؛ هیچ خبر یا لینک جدیدی نساز.\n"
        "2. لینک هر خبر را دقیقاً همان‌طور که داده شده، بدون تغییر بازنویسی کن.\n"
        "3. برای هر موضوع (Finance / Edge of Science / International Trade) "
        "حداکثر ۳ تا از مهم‌ترین و تاثیرگذارترین خبرها را انتخاب کن. اگر خبر مهمی "
        "در آن موضوع نبود، آن بخش را خالی بگذار.\n"
        "4. خلاصه هر خبر باید ۲ تا ۳ جمله، دقیق، بدون اغراق و بدون حدس و گمان باشد.\n"
        "5. خروجی را با فرمت HTML ساده و سازگار با تلگرام بنویس: از <b> برای عنوان "
        "بخش‌ها و <a href='لینک'>عنوان خبر</a> برای لینک‌ها استفاده کن. از Markdown "
        "استفاده نکن.\n"
        "6. در انتهای هر خبر نام منبع را داخل پرانتز بنویس.\n"
        "7. اگر هیچ خبر مهمی وجود نداشت، فقط بنویس: 'در این ساعت خبر مهمی یافت نشد.'"
    )

    body = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"فهرست خبرهای این ساعت:\n\n{articles_text}"}],
            }
        ],
        "generationConfig": {"maxOutputTokens": 2000, "temperature": 0.3},
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )

    resp = requests.post(
        url,
        headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        json=body,
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates") or []
    if not candidates:
        print(f"[WARN] Gemini returned no candidates: {data}")
        return None

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "\n".join(p.get("text", "") for p in parts).strip()
    return text or None


# ----------------------------------------------------------------------
# TELEGRAM
# ----------------------------------------------------------------------

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # Telegram hard limit is 4096 chars per message; chunk safely.
    max_len = 3800
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)] or [text]

    for chunk in chunks:
        resp = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": False,
        }, timeout=30)
        if not resp.ok:
            print(f"[ERROR] Telegram send failed: {resp.status_code} {resp.text}")
        time.sleep(1)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    seen = load_seen()
    articles = fetch_new_articles(seen)
    print(f"Found {len(articles)} new candidate articles.")

    if not articles:
        save_seen(seen)
        print("No new articles this run. Nothing sent.")
        return

    summary = summarize_with_gemini(articles)
    save_seen(seen)

    if not summary:
        print("Gemini returned no summary.")
        return

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"🗞 <b>خلاصه اخبار مهم — {now_str}</b>\n\n"
    send_telegram_message(header + summary)
    print("Sent summary to Telegram.")


if __name__ == "__main__":
    main()
