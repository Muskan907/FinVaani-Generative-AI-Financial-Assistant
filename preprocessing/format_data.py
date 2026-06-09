"""
format_data.py — Format Q&A pairs into model training format.
Adds formatted column with ### markers and <|endoftext|> tokens.
Input:  data/processed/cleaned_data.csv
Output: data/processed/formatted_data.csv
"""

import os
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
INPUT_FILE    = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
OUTPUT_FILE   = os.path.join(PROCESSED_DIR, "formatted_data.csv")


def format_english(question: str, answer: str) -> str:
    """
    Format an English Q&A pair into the mGPT training template.

    Template:
        ### Question: {question}
        ### Answer: {answer}<|endoftext|>
    """
    return f"### Question: {question}\n### Answer: {answer}<|endoftext|>"


def format_hindi(question: str, answer: str) -> str:
    """
    Format a Hindi Q&A pair into the mGPT training template.

    Template:
        ### सवाल: {question}
        ### जवाब: {answer}<|endoftext|>
    """
    return f"### सवाल: {question}\n### जवाब: {answer}<|endoftext|>"


def format_plain_english(question: str, answer: str) -> str:
    """Plain format without special tokens — used for evaluation reference."""
    return f"Question: {question}\nAnswer: {answer}"


def format_plain_hindi(question: str, answer: str) -> str:
    """Plain Hindi format without special tokens."""
    return f"सवाल: {question}\nजवाब: {answer}"


def format_row(row: pd.Series) -> tuple:
    """
    Format a single DataFrame row into (formatted, plain) strings.

    Args:
        row: DataFrame row with 'question', 'answer', 'language' columns.

    Returns:
        Tuple of (formatted_text, plain_text).
    """
    q = str(row["question"]).strip()
    a = str(row["answer"]).strip()
    lang = str(row.get("language", "en")).strip().lower()

    if lang == "hi":
        formatted = format_hindi(q, a)
        plain     = format_plain_hindi(q, a)
    else:
        formatted = format_english(q, a)
        plain     = format_plain_english(q, a)

    return formatted, plain


def main():
    """Main entry point for format_data.py."""
    print("=" * 60)
    print("FinVaani — Data Formatter")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Please run preprocessing/clean_data.py first."
        )

    df = pd.read_csv(INPUT_FILE, encoding="utf-8")
    print(f"[Format] Loaded {len(df)} rows from {INPUT_FILE}")

    # Apply formatting
    formatted_list = []
    plain_list     = []

    for _, row in df.iterrows():
        fmt, plain = format_row(row)
        formatted_list.append(fmt)
        plain_list.append(plain)

    df["formatted"] = formatted_list
    df["plain"]     = plain_list

    # Verify formatting
    en_sample = df[df["language"] == "en"]["formatted"].iloc[0] if len(df[df["language"] == "en"]) > 0 else ""
    hi_sample = df[df["language"] == "hi"]["formatted"].iloc[0] if len(df[df["language"] == "hi"]) > 0 else ""

    print("\n[Format] Sample English formatted entry:")
    print(en_sample[:200] + "..." if len(en_sample) > 200 else en_sample)

    if hi_sample:
        print("\n[Format] Sample Hindi formatted entry:")
        print(hi_sample[:200] + "..." if len(hi_sample) > 200 else hi_sample)

    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n[Format] Saved {len(df)} formatted rows → {OUTPUT_FILE}")

    # Stats
    avg_len_en = df[df["language"] == "en"]["formatted"].str.len().mean()
    avg_len_hi = df[df["language"] == "hi"]["formatted"].str.len().mean()
    print(f"[Format] Avg formatted length — English: {avg_len_en:.0f} chars, "
          f"Hindi: {avg_len_hi:.0f} chars")


if __name__ == "__main__":
    main()
