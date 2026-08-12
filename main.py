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
import re
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
LOOKBACK_HOURS = int(os.environ.get("LOOKBACK_HOURS", "6"))

SEEN_FILE = "seen_links.json"
WEEKLY_LOG_FILE = "weekly_log.json"

# Curated, high-trust sources grouped by topic.
# NOTE: USPTO/WIPO patent grants and the World Bank do not publish a
# simple, reliable public RSS feed for general search results, so they
# are intentionally left out (see README for manual alternatives).
# Matt Levine (Bloomberg) and Stratechery do have public RSS feeds, but
# full article text is paywalled — same limitation as the FT feed below
# (title + short summary only, which is what gets sent to Telegram).
FEEDS = {
    "Finance": [
        ("The Economist - Finance", "https://www.economist.com/finance-and-economics/rss.xml"),
        ("Financial Times", "https://www.ft.com/world?format=rss"),
        ("Wall Street Journal - Markets", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
        # منبع دست‌اول: بیانیه‌های رسمی فدرال رزرو آمریکا (نه تفسیر رسانه‌ها)
        ("Federal Reserve - Press Releases", "https://www.federalreserve.gov/feeds/press_all.xml"),
        # منبع دست‌اول: بیانیه‌های رسمی بانک مرکزی اروپا
        ("ECB - Press Releases", "https://www.ecb.europa.eu/rss/press.xml"),
        # منبع دست‌اول: اعلامیه‌های رسمی کمیسیون بورس آمریکا (SEC)
        ("SEC - Press Releases", "https://www.sec.gov/news/pressreleases.rss"),
        # تحلیلگر مستقل معتبر در حوزه سرمایه‌گذاری و استراتژی بازار
        ("The Diff (Byrne Hobart)", "https://www.thediff.co/feed"),
        # تحلیل روزانه بازار و وال‌استریت (عنوان/خلاصه رایگان، متن کامل پولی)
        ("Money Stuff (Matt Levine)", "https://www.bloomberg.com/opinion/authors/ARbTQlRLRjE/matthew-s-levine.rss"),
    ],
    "Edge of Science": [
        ("MIT Technology Review", "https://www.technologyreview.com/feed/"),
        ("Nature - Current", "https://www.nature.com/nature.rss"),
        ("Science (AAAS)", "https://www.science.org/rss/news_current.xml"),
        ("IEEE Spectrum", "https://spectrum.ieee.org/rss/fulltext"),
        ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
        ("Quanta Magazine", "https://api.quantamagazine.org/feed/"),
        # منبع دست‌اول: خودِ مقالات علمی جدید هوش مصنوعی (قبل از پخش خبرش)
        ("arXiv - cs.AI (هوش مصنوعی)", "https://rss.arxiv.org/rss/cs.AI"),
        # منبع دست‌اول: مقالات جدید فیزیک کوانتوم
        ("arXiv - quant-ph (فیزیک کوانتوم)", "https://rss.arxiv.org/rss/quant-ph"),
        # خلاصه هفتگی معتبر و دقیق از پیشرفت‌های AI
        ("Import AI (Jack Clark)", "https://importai.substack.com/feed"),
        # تحلیل تکنولوژی و مدل‌های کسب‌وکار (بخشی رایگان، بخشی پولی)
        ("Stratechery (Ben Thompson)", "https://stratechery.com/feed/"),
    ],
    "International Trade": [
        ("The Economist - Business", "https://www.economist.com/business/rss.xml"),
        ("Harvard Business Review", "https://hbr.org/feed"),
        ("Financial Times - World", "https://www.ft.com/world?format=rss"),
        # منبع دست‌اول: گزارش‌های رسمی صندوق بین‌المللی پول
        ("IMF - News", "https://www.imf.org/en/News/rss"),
        # منبع دست‌اول: اخبار رسمی سازمان تجارت جهانی
        ("WTO - News", "http://www.wto.org/library/rss/latest_news_e.xml"),
    ],
}


# ----------------------------------------------------------------------
# STATE HANDLING
# ----------------------------------------------------------------------

def load_seen():
    if os.path.exists(SEEN_FILE):
        try:
            with open(SEEN_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return set()
                return set(json.loads(content))
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WARN] seen_links.json is invalid ({e}); starting fresh.")
            return set()
    return set()


def save_seen(seen):
    # Keep the file from growing forever: cap at the most recent 2000 links.
    trimmed = list(seen)[-2000:]
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, ensure_ascii=False, indent=2)


def append_weekly_log(articles):
    """Record every new article (regardless of whether the hourly digest
    picked it) so the weekly digest has full raw material to spot trends
    across the week, not just what got sent hourly."""
    if not articles:
        return

    log = []
    if os.path.exists(WEEKLY_LOG_FILE):
        try:
            with open(WEEKLY_LOG_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    log = json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WARN] weekly_log.json is invalid ({e}); starting fresh.")
            log = []

    now_iso = datetime.now(timezone.utc).isoformat()
    for a in articles:
        log.append({
            "date": now_iso,
            "topic": a["topic"],
            "source": a["source"],
            "title": a["title"],
            "link": a["link"],
        })

    # Keep only the last 9 days so the file doesn't grow forever.
    cutoff = datetime.now(timezone.utc) - timedelta(days=9)

    def entry_time(e):
        try:
            return datetime.fromisoformat(e["date"])
        except (ValueError, KeyError):
            return datetime.now(timezone.utc)

    log = [e for e in log if entry_time(e) >= cutoff]

    with open(WEEKLY_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


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
        "حداکثر ۴ تا از خبرهای موجود را انتخاب کن. سخت‌گیر نباش: تا وقتی خبر "
        "از یک منبع معتبر است و واقعاً به آن موضوع مرتبط است، آن را در خروجی "
        "بگنجان، حتی اگر خیلی 'بزرگ' یا 'تاریخ‌ساز' نباشد؛ فقط خبرهای کاملاً "
        "کم‌اهمیت یا تکراری (مثل گزارش‌های خیلی جزئی یا تبلیغاتی) را کنار "
        "بگذار.\n"
        "3b. تایید متقاطع (Cross-verification): اگر یک رویداد یا موضوع مشابه "
        "توسط دو یا چند منبع مستقل در فهرست پوشش داده شده، این یعنی خبر واقعاً "
        "مهم است — این خبر را در اولویت اول قرار بده و کنار آن علامت 🔥 بگذار "
        "و بنویس 'تایید شده توسط چند منبع'. برای منابع دست‌اول و رسمی (مثل "
        "Federal Reserve، arXiv) که مستقیماً از خودِ رویداد یا مقاله علمی "
        "خبر می‌دهند (نه گزارش رسانه‌ای از آن)، علامت 📌 و عبارت 'منبع دست‌اول' "
        "را کنار خبر بگذار.\n"
        "4. خلاصه هر خبر باید فقط ۱ تا ۲ جمله کوتاه، دقیق، بدون اغراق و بدون "
        "حدس و گمان باشد (برای اینکه پاسخ کامل کوتاه و مطمئن ارسال شود).\n"
        "5. خروجی را کاملاً به‌صورت متن ساده (Plain Text) بنویس — هیچ‌وقت از "
        "تگ HTML یا Markdown (مثل <b>، <a>، **، []()) استفاده نکن. برای عنوان "
        "هر بخش موضوعی، فقط نام موضوع را با یک ایموجی مناسب در ابتدای خط "
        "بنویس (مثلاً '💰 Finance'). لینک هر خبر را در یک خط جدا و به‌صورت "
        "کامل و دقیقاً همان‌طور که داده شده بنویس (تلگرام خودش آن را "
        "کلیک‌پذیر می‌کند).\n"
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
        "generationConfig": {"maxOutputTokens": 8000, "temperature": 0.3},
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

    finish_reason = candidates[0].get("finishReason")
    if finish_reason == "MAX_TOKENS":
        print("[WARN] Gemini response was cut off (hit max output tokens). "
              "Consider raising maxOutputTokens further or reducing article count.")

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
        # Plain text, no parse_mode: Telegram auto-links bare URLs on its
        # own, so we never depend on the model producing valid markup.
        resp = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
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
    append_weekly_log(articles)

    if not summary:
        print("Gemini returned no summary.")
        return

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"🗞 خلاصه اخبار مهم — {now_str}\n\n"
    send_telegram_message(header + summary)
    print("Sent summary to Telegram.")


if __name__ == "__main__":
    main()
