"""
split_data.py — Split formatted data into train / val / test sets.
Uses stratified split by language for balanced representation.
Input:  data/processed/formatted_data.csv
Output: data/splits/train.csv, val.csv, test.csv
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
SPLITS_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "splits")
INPUT_FILE    = os.path.join(PROCESSED_DIR, "formatted_data.csv")

RANDOM_STATE = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15


def print_split_stats(name: str, df: pd.DataFrame) -> None:
    """Print statistics for a single data split."""
    en_count = (df["language"] == "en").sum()
    hi_count = (df["language"] == "hi").sum()
    avg_wc   = df["word_count"].mean() if "word_count" in df.columns else 0
    print(f"\n  [{name}]")
    print(f"    Total:              {len(df):>4}")
    print(f"    English:            {en_count:>4}")
    print(f"    Hindi:              {hi_count:>4}")
    print(f"    Avg answer words:   {avg_wc:.1f}")


def split_data(df: pd.DataFrame) -> tuple:
    """
    Split DataFrame into train / val / test with stratification by language.

    Args:
        df: Full formatted DataFrame.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_STATE,
        stratify=df["language"],
    )

    # Second split: val vs test (equal halves of the temp set)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_df["language"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def main():
    """Main entry point for split_data.py."""
    print("=" * 60)
    print("FinVaani — Data Splitter")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Please run preprocessing/format_data.py first."
        )

    df = pd.read_csv(INPUT_FILE, encoding="utf-8")
    print(f"[Split] Loaded {len(df)} rows from {INPUT_FILE}")

    # Ensure word_count column exists
    if "word_count" not in df.columns:
        df["word_count"] = df["answer"].apply(lambda x: len(str(x).split()))

    train_df, val_df, test_df = split_data(df)

    os.makedirs(SPLITS_DIR, exist_ok=True)

    train_path = os.path.join(SPLITS_DIR, "train.csv")
    val_path   = os.path.join(SPLITS_DIR, "val.csv")
    test_path  = os.path.join(SPLITS_DIR, "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path,     index=False, encoding="utf-8")
    test_df.to_csv(test_path,   index=False, encoding="utf-8")

    print("\n[Split] Split statistics:")
    print_split_stats("Train", train_df)
    print_split_stats("Val",   val_df)
    print_split_stats("Test",  test_df)

    total = len(train_df) + len(val_df) + len(test_df)
    print(f"\n[Split] Ratios — Train: {len(train_df)/total:.1%}, "
          f"Val: {len(val_df)/total:.1%}, Test: {len(test_df)/total:.1%}")

    print(f"\n[Split] Saved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    main()
