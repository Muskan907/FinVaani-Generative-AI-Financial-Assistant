"""
scrape_all.py — Master script that runs all five FinVaani scrapers sequentially.
Reports progress and total pairs per source.
"""

import os
import sys

# Ensure the scraping directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

from scrape_rbi  import scrape_rbi,  save_to_csv as save_rbi
from scrape_sebi import scrape_sebi, save_to_csv as save_sebi
from scrape_irdai import scrape_irdai, save_to_csv as save_irdai
from scrape_npci import scrape_npci, save_to_csv as save_npci
from scrape_ncfe import scrape_ncfe, save_to_csv as save_ncfe

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def run_scraper(name: str, scrape_fn, save_fn, output_file: str) -> int:
    """
    Run a single scraper safely, catching any exceptions so other scrapers
    can continue even if one fails.

    Returns the number of pairs collected (0 on failure).
    """
    print(f"\n{'=' * 60}")
    print(f"  Running {name} scraper...")
    print(f"{'=' * 60}")
    try:
        pairs = scrape_fn()
        save_fn(pairs, output_file)
        print(f"  ✓ {name}: {len(pairs)} pairs collected")
        return len(pairs)
    except Exception as e:
        print(f"  ✗ {name} scraper FAILED: {e}")
        return 0


def main():
    """Run all scrapers and print a summary report."""
    print("\n" + "=" * 60)
    print("  FinVaani — Master Scraper")
    print("  Collecting Indian financial Q&A data from 5 sources")
    print("=" * 60)

    os.makedirs(RAW_DIR, exist_ok=True)

    results = {}

    results["RBI"] = run_scraper(
        "RBI",
        scrape_rbi,
        save_rbi,
        os.path.join(RAW_DIR, "rbi_raw.csv"),
    )

    results["SEBI"] = run_scraper(
        "SEBI",
        scrape_sebi,
        save_sebi,
        os.path.join(RAW_DIR, "sebi_raw.csv"),
    )

    results["IRDAI"] = run_scraper(
        "IRDAI",
        scrape_irdai,
        save_irdai,
        os.path.join(RAW_DIR, "irdai_raw.csv"),
    )

    results["NPCI"] = run_scraper(
        "NPCI",
        scrape_npci,
        save_npci,
        os.path.join(RAW_DIR, "npci_raw.csv"),
    )

    results["NCFE"] = run_scraper(
        "NCFE",
        scrape_ncfe,
        save_ncfe,
        os.path.join(RAW_DIR, "ncfe_raw.csv"),
    )

    # ── Summary report ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SCRAPING SUMMARY")
    print("=" * 60)
    total = 0
    for source, count in results.items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {source:<8}: {count:>4} pairs")
        total += count
    print(f"  {'─' * 30}")
    print(f"  {'TOTAL':<8}: {total:>4} pairs")
    print("=" * 60)
    print("\nAll raw CSVs saved to data/raw/")
    print("Next step: Run preprocessing/merge_raw.py")


if __name__ == "__main__":
    main()
