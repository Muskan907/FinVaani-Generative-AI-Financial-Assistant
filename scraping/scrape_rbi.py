"""
scrape_rbi.py — Scraper for RBI (Reserve Bank of India) FAQ pages.

English FAQs:
  URL pattern: https://www.rbi.org.in/Scripts/FAQView.aspx?Id=<n>
  Structure:   Numbered <p> tags — "1. Question?" followed by answer <p>s
  Working IDs: 3, 13, 18, 49, 54, 60, 64, 65, 67, 68, 70, 75, 76,
               77, 79, 81, 83, 86, 92  (live-tested May 2026)

Hindi FAQs:
  URL pattern: https://www.rbi.org.in/hindi/Scripts/Faqs.aspx?ID=<n>
  Structure:   Same numbered <p> pattern but in Devanagari
  Working IDs: 101 IDs found on the Hindi FAQ index (live-tested May 2026)
  Note: Hindi uses Faqs.aspx (not FAQView.aspx) with uppercase ID=

Output: data/raw/rbi_raw.csv
"""

import os
import re
import time
import csv
import requests
from bs4 import BeautifulSoup

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rbi_raw.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

# ── English FAQ IDs (confirmed working, live-tested May 2026) ────────────────
ENGLISH_FAQ_IDS = [3, 13, 18, 49, 54, 60, 64, 65, 67, 68, 70, 75,
                   76, 77, 79, 81, 83, 86, 92]
ENGLISH_BASE = "https://www.rbi.org.in/Scripts/FAQView.aspx"

# ── Hindi FAQ IDs (all 101 IDs from Hindi FAQ index, live-tested May 2026) ───
# URL pattern is DIFFERENT from English: hindi/Scripts/Faqs.aspx?ID=<n>
HINDI_FAQ_IDS = [
    8, 10, 15, 16, 21, 23, 24, 26, 28, 29, 31, 33, 34, 44, 46, 48,
    49, 50, 51, 52, 55, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 69,
    72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
    89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
    104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    117, 118, 119, 121, 123, 124, 125, 126, 127, 128, 129, 130, 131,
    132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
]
HINDI_BASE = "https://www.rbi.org.in/hindi/Scripts/Faqs.aspx"

MIN_ANSWER_LEN   = 80
MIN_QUESTION_LEN = 15

# Matches numbered questions in both English ("1. What is...") and
# Hindi ("1. आरबीआई क्या है?")
QUESTION_PATTERN = re.compile(r"^\d+\.?\s+\S")


def clean_text(text: str) -> str:
    """Remove HTML artifacts and normalise whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    for pat in ["Home >", "FAQ >", "Back to top", "Read more",
                "Click here", "Download", "Print", "Skip to main content",
                "मुख्य सामग्री पर जाएं"]:
        text = text.replace(pat, "")
    return re.sub(r"\s+", " ", text).strip()


def extract_qa_from_paragraphs(paragraphs: list, url: str, language: str) -> list:
    """
    Extract Q&A pairs from a list of paragraph strings.

    RBI FAQ pages (both English and Hindi) use a consistent pattern:
      - Questions start with a number: "1.  What is..." / "1. क्या है..."
      - Answers follow in subsequent paragraphs until the next numbered item

    Args:
        paragraphs: List of cleaned paragraph text strings.
        url:        Source URL for attribution.
        language:   "en" or "hi"

    Returns:
        List of dicts with question, answer, source, language, url.
    """
    pairs = []
    i = 0
    while i < len(paragraphs) - 1:
        p = paragraphs[i]
        if QUESTION_PATTERN.match(p):
            question = re.sub(r"^\d+\.?\s+", "", p).strip()

            j = i + 1
            answer_parts = []
            while j < len(paragraphs) and not QUESTION_PATTERN.match(paragraphs[j]):
                answer_parts.append(paragraphs[j])
                j += 1

            answer = " ".join(answer_parts).strip()

            if len(question) >= MIN_QUESTION_LEN and len(answer) >= MIN_ANSWER_LEN:
                pairs.append({
                    "question": question,
                    "answer":   answer,
                    "source":   "RBI",
                    "language": language,
                    "url":      url,
                })
            i = j
        else:
            i += 1

    return pairs


def scrape_english_faq(faq_id: int) -> list:
    """
    Fetch a single English RBI FAQ page (FAQView.aspx?Id=<n>).

    Returns:
        List of Q&A pair dicts (empty on failure).
    """
    url = f"{ENGLISH_BASE}?Id={faq_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        r.encoding = "utf-8"
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = [
            clean_text(p.get_text())
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 10
        ]
        pairs = extract_qa_from_paragraphs(paragraphs, url, "en")
        print(f"  [RBI-EN] ID {faq_id:3d}: {len(pairs):3d} pairs")
        return pairs
    except requests.exceptions.RequestException as e:
        print(f"  [RBI-EN] WARNING — ID {faq_id}: {e}")
        return []


def scrape_hindi_faq(faq_id: int) -> list:
    """
    Fetch a single Hindi RBI FAQ page (hindi/Scripts/Faqs.aspx?ID=<n>).
    Note: Hindi uses uppercase ID= and Faqs.aspx (not FAQView.aspx).

    Returns:
        List of Q&A pair dicts (empty on failure).
    """
    url = f"{HINDI_BASE}?ID={faq_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        r.encoding = "utf-8"
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = [
            clean_text(p.get_text())
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 10
        ]
        # Keep only paragraphs that contain Devanagari (filter nav/English noise)
        paragraphs = [p for p in paragraphs
                      if re.search(r"[\u0900-\u097F]", p) or QUESTION_PATTERN.match(p)]
        pairs = extract_qa_from_paragraphs(paragraphs, url, "hi")
        print(f"  [RBI-HI] ID {faq_id:3d}: {len(pairs):3d} pairs")
        return pairs
    except requests.exceptions.RequestException as e:
        print(f"  [RBI-HI] WARNING — ID {faq_id}: {e}")
        return []


def scrape_rbi() -> list:
    """
    Main scraping function for RBI.
    Scrapes both English (FAQView.aspx) and Hindi (Faqs.aspx) FAQ pages.

    Returns:
        Combined list of all Q&A pair dicts.
    """
    all_pairs = []

    print(f"[RBI] Scraping {len(ENGLISH_FAQ_IDS)} English FAQ pages...")
    for faq_id in ENGLISH_FAQ_IDS:
        pairs = scrape_english_faq(faq_id)
        all_pairs.extend(pairs)
        time.sleep(2)

    print(f"\n[RBI] Scraping {len(HINDI_FAQ_IDS)} Hindi FAQ pages...")
    for faq_id in HINDI_FAQ_IDS:
        pairs = scrape_hindi_faq(faq_id)
        all_pairs.extend(pairs)
        time.sleep(2)

    return all_pairs


def save_to_csv(pairs: list, filepath: str) -> None:
    """Save list of Q&A pair dicts to a CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = ["question", "answer", "source", "language", "url"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)
    print(f"[RBI] Saved {len(pairs)} pairs → {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("FinVaani — RBI FAQ Scraper (English + Hindi)")
    print("=" * 60)

    pairs = scrape_rbi()

    # Deduplicate by question text
    seen = set()
    unique_pairs = []
    for p in pairs:
        key = p["question"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    save_to_csv(unique_pairs, OUTPUT_FILE)

    en_count = sum(1 for p in unique_pairs if p["language"] == "en")
    hi_count = sum(1 for p in unique_pairs if p["language"] == "hi")
    print(f"[RBI] Total unique pairs: {len(unique_pairs)} "
          f"(English: {en_count}, Hindi: {hi_count})")
