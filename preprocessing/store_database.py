"""
store_database.py — Create SQLite database and populate all tables.
Creates data/database/project.db with 5 tables.
"""

import os
import sqlite3
import json
import pandas as pd
from datetime import datetime

DB_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "database")
DB_PATH  = os.path.join(DB_DIR, "project.db")
SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "splits")

DDL = """
CREATE TABLE IF NOT EXISTS train_data (
    id         INTEGER PRIMARY KEY,
    question   TEXT,
    answer     TEXT,
    formatted  TEXT,
    source     TEXT,
    language   TEXT,
    word_count INTEGER
);

CREATE TABLE IF NOT EXISTS val_data (
    id         INTEGER PRIMARY KEY,
    question   TEXT,
    answer     TEXT,
    formatted  TEXT,
    source     TEXT,
    language   TEXT,
    word_count INTEGER
);

CREATE TABLE IF NOT EXISTS test_data (
    id         INTEGER PRIMARY KEY,
    question   TEXT,
    answer     TEXT,
    formatted  TEXT,
    source     TEXT,
    language   TEXT,
    word_count INTEGER
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT,
    language        TEXT,
    bleu_score      REAL,
    rouge_l_score   REAL,
    perplexity      REAL,
    sparsity_percent REAL,
    pruning_round   INTEGER,
    timestamp       TEXT
);

CREATE TABLE IF NOT EXISTS pruning_masks (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    pruning_round         INTEGER,
    layer_name            TEXT,
    sparsity_percent      REAL,
    mask_json             TEXT,
    param_count_remaining INTEGER
);

CREATE TABLE IF NOT EXISTS generated_outputs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name       TEXT,
    question         TEXT,
    language         TEXT,
    generated_answer TEXT,
    reference_answer TEXT,
    is_hallucination INTEGER
);
"""


def insert_split(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    """Insert a DataFrame split into the given table."""
    rows = []
    for i, row in df.iterrows():
        rows.append((
            int(i),
            str(row.get("question", "")),
            str(row.get("answer", "")),
            str(row.get("formatted", "")),
            str(row.get("source", "")),
            str(row.get("language", "en")),
            int(row.get("word_count", 0)),
        ))
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} "
        "(id, question, answer, formatted, source, language, word_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def create_database() -> sqlite3.Connection:
    """Create the SQLite database and all tables."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(DDL)
    conn.commit()
    print(f"[DB] Database created at {DB_PATH}")
    return conn


def main():
    print("=" * 60)
    print("FinVaani — Database Builder")
    print("=" * 60)

    conn = create_database()

    for split in ["train", "val", "test"]:
        path = os.path.join(SPLITS_DIR, f"{split}.csv")
        if not os.path.exists(path):
            print(f"[DB] WARNING — {path} not found, skipping.")
            continue
        df = pd.read_csv(path, encoding="utf-8")
        insert_split(conn, f"{split}_data", df)
        print(f"[DB] {split}_data: {len(df)} rows inserted")

    # Verify row counts
    print("\n[DB] Table row counts:")
    for table in ["train_data", "val_data", "test_data",
                  "evaluation_results", "pruning_masks", "generated_outputs"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<25}: {count}")

    conn.close()
    print(f"\n[DB] Done. Database saved → {DB_PATH}")


if __name__ == "__main__":
    main()
