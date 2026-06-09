"""
clean_data.py — Clean and deduplicate the merged raw dataset.
Applies 10 cleaning steps in order.
Input:  data/processed/merged_raw.csv
Output: data/processed/cleaned_data.csv
"""

import os
import re
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
INPUT_FILE    = os.path.join(PROCESSED_DIR, "merged_raw.csv")
OUTPUT_FILE   = os.path.join(PROCESSED_DIR, "cleaned_data.csv")

# Navigation / boilerplate fragments to strip
NAV_PATTERNS = [
    "Home >", "FAQ >", "Back to top", "Read more", "Click here",
    "Download", "Print", "Skip to main content", "Screen Reader Access",
    "A+ A-", "A+", "A-", "Last Updated", "Source:", "© ",
]


def remove_html_tags(text: str) -> str:
    """Step 1: Remove all HTML tags using regex."""
    return re.sub(r"<[^>]+>", "", str(text))


def remove_nav_artifacts(text: str) -> str:
    """Step 2: Remove navigation and boilerplate fragments."""
    for pat in NAV_PATTERNS:
        text = text.replace(pat, "")
    return text


def normalize_whitespace(text: str) -> str:
    """Step 3: Collapse multiple whitespace characters into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def word_overlap_ratio(text1: str, text2: str) -> float:
    """
    Compute simple word overlap ratio between two strings.
    Used for near-duplicate detection.
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def remove_near_duplicates(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """
    Step 7: Remove near-duplicate questions using word overlap.
    Keeps the first occurrence when overlap > threshold.
    """
    keep = []
    kept_questions = []

    for idx, row in df.iterrows():
        q = row["question"].lower()
        is_near_dup = False
        for kept_q in kept_questions:
            if word_overlap_ratio(q, kept_q) > threshold:
                is_near_dup = True
                break
        if not is_near_dup:
            keep.append(idx)
            kept_questions.append(q)

    return df.loc[keep].reset_index(drop=True)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all 10 cleaning steps to the DataFrame.

    Steps:
        1. Remove HTML tags
        2. Remove navigation artifacts
        3. Normalize whitespace
        4. Filter short answers (< 80 chars)
        5. Filter short questions (< 15 chars)
        6. Remove exact duplicate questions
        7. Remove near-duplicate questions (word overlap > 90%)
        8. Preserve Devanagari encoding (ensured via utf-8 throughout)
        9. Add word_count column
        10. Save and report
    """
    initial_count = len(df)
    print(f"[Clean] Initial row count: {initial_count}")

    # Step 1 — Remove HTML tags
    df["question"] = df["question"].apply(remove_html_tags)
    df["answer"]   = df["answer"].apply(remove_html_tags)
    print("[Clean] Step 1 ✓ — HTML tags removed")

    # Step 2 — Remove navigation artifacts
    df["question"] = df["question"].apply(remove_nav_artifacts)
    df["answer"]   = df["answer"].apply(remove_nav_artifacts)
    print("[Clean] Step 2 ✓ — Navigation artifacts removed")

    # Step 3 — Normalize whitespace
    df["question"] = df["question"].apply(normalize_whitespace)
    df["answer"]   = df["answer"].apply(normalize_whitespace)
    print("[Clean] Step 3 ✓ — Whitespace normalized")

    # Step 4 — Filter short answers
    before = len(df)
    df = df[df["answer"].str.len() >= 80].reset_index(drop=True)
    print(f"[Clean] Step 4 ✓ — Short answers removed: {before - len(df)} rows dropped")

    # Step 5 — Filter short questions
    before = len(df)
    df = df[df["question"].str.len() >= 15].reset_index(drop=True)
    print(f"[Clean] Step 5 ✓ — Short questions removed: {before - len(df)} rows dropped")

    # Step 6 — Remove exact duplicate questions
    before = len(df)
    df = df.drop_duplicates(subset=["question"], keep="first").reset_index(drop=True)
    print(f"[Clean] Step 6 ✓ — Exact duplicates removed: {before - len(df)} rows dropped")

    # Step 7 — Remove near-duplicate questions
    before = len(df)
    df = remove_near_duplicates(df, threshold=0.90)
    print(f"[Clean] Step 7 ✓ — Near-duplicates removed: {before - len(df)} rows dropped")

    # Step 8 — Devanagari preservation (ensured by utf-8 I/O; verify here)
    hi_rows = df[df["language"] == "hi"]
    devanagari_ok = hi_rows["question"].str.contains(r"[\u0900-\u097F]", regex=True).sum()
    print(f"[Clean] Step 8 ✓ — Hindi rows with Devanagari: {devanagari_ok}/{len(hi_rows)}")

    # Step 9 — Add word_count column
    df["word_count"] = df["answer"].apply(lambda x: len(str(x).split()))
    print("[Clean] Step 9 ✓ — word_count column added")

    final_count = len(df)
    print(f"\n[Clean] Final row count: {final_count}")
    print(f"[Clean] Total removed: {initial_count - final_count} rows "
          f"({100 * (initial_count - final_count) / max(initial_count, 1):.1f}%)")

    return df


def main():
    """Main entry point for clean_data.py."""
    print("=" * 60)
    print("FinVaani — Data Cleaner")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Please run preprocessing/merge_raw.py first."
        )

    df = pd.read_csv(INPUT_FILE, encoding="utf-8")
    df = clean_dataframe(df)

    # Step 10 — Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n[Clean] Saved → {OUTPUT_FILE}")

    print("\n[Clean] Language breakdown after cleaning:")
    for lang, count in df["language"].value_counts().items():
        label = "English" if lang == "en" else "Hindi"
        print(f"  {label:<10}: {count:>4} rows")

    print("\n[Clean] Source breakdown after cleaning:")
    for source, count in df["source"].value_counts().items():
        print(f"  {source:<10}: {count:>4} rows")


if __name__ == "__main__":
    main()
