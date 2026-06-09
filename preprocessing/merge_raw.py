"""
merge_raw.py — Merge all raw scraped CSVs into a single DataFrame.
Reads from data/raw/, saves to data/processed/merged_raw.csv.
"""

import os
import pandas as pd

RAW_DIR       = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUTPUT_FILE   = os.path.join(PROCESSED_DIR, "merged_raw.csv")

EXPECTED_COLUMNS = ["question", "answer", "source", "language", "url"]


def merge_raw_csvs() -> pd.DataFrame:
    """
    Read all CSV files from data/raw/ and concatenate them into one DataFrame.

    Returns:
        Combined DataFrame with standardised columns.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith("_raw.csv")]
    if not csv_files:
        raise FileNotFoundError(
            f"No raw CSV files found in {RAW_DIR}. "
            "Please run scraping/scrape_all.py first."
        )

    frames = []
    print(f"[Merge] Found {len(csv_files)} raw CSV files:")

    for fname in sorted(csv_files):
        fpath = os.path.join(RAW_DIR, fname)
        try:
            df = pd.read_csv(fpath, encoding="utf-8")
            # Ensure all expected columns exist
            for col in EXPECTED_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            df = df[EXPECTED_COLUMNS]
            print(f"  {fname:<25}: {len(df):>4} rows")
            frames.append(df)
        except Exception as e:
            print(f"  WARNING — Could not read {fname}: {e}")

    if not frames:
        raise ValueError("No valid CSV files could be read.")

    merged = pd.concat(frames, ignore_index=True)
    return merged


def main():
    """Main entry point for merge_raw.py."""
    print("=" * 60)
    print("FinVaani — Raw Data Merger")
    print("=" * 60)

    merged = merge_raw_csvs()

    print(f"\n[Merge] Total rows after merge: {len(merged)}")
    print("\n[Merge] Per-source breakdown:")
    for source, count in merged["source"].value_counts().items():
        print(f"  {source:<10}: {count:>4} rows")

    print("\n[Merge] Language breakdown:")
    for lang, count in merged["language"].value_counts().items():
        label = "English" if lang == "en" else "Hindi"
        print(f"  {label:<10}: {count:>4} rows")

    merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n[Merge] Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
