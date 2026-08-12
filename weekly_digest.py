"""
Weekly Trends Digest: reads the rolling log of every article collected
during the past week (weekly_log.json, written by main.py on every hourly
run) and asks Gemini to identify the handful of real trends connecting
those stories -- not just repeat headlines. Sends the result to the same
Telegram chat as a deeper, once-a-week digest.

Meant to run once a week (see .github/workflows/weekly.yml).
"""

import os
import json
import requests
from datetime import datetime, timezone

from main import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    send_telegram_message,
)

WEEKLY_LOG_FILE = "weekly_log.json"


def load_weekly_log():
    if not os.path.exists(WEEKLY_LOG_FILE):
        return []
    try:
        with open(WEEKLY_LOG_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return json.loads(content) if content else []
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[WARN] weekly_log.json is invalid ({e}); nothing to summarize.")
        return []


def summarize_week_with_gemini(entries):
    if not entries:
        return None

    by_topic = {}
    for e in entries:
        by_topic.setdefault(e["topic"], []).append(e)

    sections = []
    for topic, items in by_topic.items():
        lines = "\n".join(
            f"- ({item['source']}) {item['title']} — {item['link']}" for item in items
        )
        sections.append(f"=== {topic} ===\n{lines}")
    articles_text = "\n\n".join(sections)

    system_prompt = (
        "تو یک تحلیلگر ارشد هستی. وظیفه‌ات پیدا کردن روندهای واقعی هفته از "
        "میان ده‌ها خبر پراکنده‌ی زیر است، نه فقط فهرست کردن دوباره تیترها. "
        "فهرست شامل همه خبرهایی است که در هفته‌ی گذشته از منابع معتبر جمع‌آوری "
        "شده (گروه‌بندی‌شده بر اساس موضوع Finance / Edge of Science / "
        "International Trade).\n\n"
        "قوانین مهم:\n"
        "1. برای هر موضوع، ۲ تا ۴ روند اصلی (نه خبر تکی) را که در طول هفته "
        "تکرار شده یا چند خبر به‌هم مرتبط بوده‌اند شناسایی کن و هرکدام را در "
        "۲ تا ۴ جمله‌ی فارسی روان توضیح بده.\n"
        "2. زیر هر روند، لینک ۱ تا ۳ خبر مرتبط را که این روند را نشان "
        "می‌دهند دقیقاً همان‌طور که داده شده بیاور.\n"
        "3. فقط از خبرهای همین فهرست استفاده کن؛ هیچ ادعا یا لینک جدیدی "
        "نساز.\n"
        "4. خروجی کاملاً به‌صورت متن ساده باشد (بدون تگ HTML یا Markdown). "
        "برای عنوان هر موضوع، فقط نام موضوع را با یک ایموجی مناسب در ابتدای "
        "خط بیاور.\n"
        "5. اگر برای یک موضوع روند مشخصی پیدا نشد، آن بخش را خالی بگذار."
    )

    body = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {"role": "user", "parts": [{"text": f"خبرهای هفته‌ی گذشته:\n\n{articles_text}"}]}
        ],
        "generationConfig": {"maxOutputTokens": 8000, "temperature": 0.4},
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    resp = requests.post(
        url,
        headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates") or []
    if not candidates:
        print(f"[WARN] Gemini returned no candidates: {data}")
        return None

    if candidates[0].get("finishReason") == "MAX_TOKENS":
        print("[WARN] Weekly digest was cut off (hit max output tokens).")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "\n".join(p.get("text", "") for p in parts).strip()
    return text or None


def main():
    entries = load_weekly_log()
    print(f"Loaded {len(entries)} articles from the past week.")

    if not entries:
        print("No articles logged this week. Nothing to summarize.")
        return

    summary = summarize_week_with_gemini(entries)
    if not summary:
        print("Gemini returned no weekly summary.")
        return

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    header = f"📊 خلاصه هفتگی روندها — هفته منتهی به {now_str}\n\n"
    send_telegram_message(header + summary)
    print("Sent weekly digest to Telegram.")

    # Start the next week with a clean slate.
    with open(WEEKLY_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)


if __name__ == "__main__":
    main()
