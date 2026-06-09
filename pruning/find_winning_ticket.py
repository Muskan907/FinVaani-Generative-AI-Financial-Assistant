"""
find_winning_ticket.py — Identify and copy the best pruning round checkpoint.
Reads evaluation_results from the database and finds the highest-sparsity
round where BLEU >= 85% of the fine-tuned baseline.
"""

import os
import shutil
import sqlite3
import pandas as pd

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
DB_PATH     = os.path.join(BASE_DIR, "data", "database", "project.db")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
QUALITY_FLOOR = 0.85


def find_winning_ticket():
    """Query the database and identify the winning ticket round."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM evaluation_results ORDER BY pruning_round",
        conn,
    )
    conn.close()

    if df.empty:
        print("[WinTicket] No evaluation results found. Run lth_pruning.py first.")
        return None

    print("[WinTicket] Evaluation results from database:")
    print(df[["model_name", "pruning_round", "bleu_score",
              "rouge_l_score", "perplexity", "sparsity_percent"]].to_string(index=False))

    # Get baseline BLEU (round 0 = fine-tuned model)
    baseline_row = df[df["pruning_round"] == 0]
    if baseline_row.empty:
        print("[WinTicket] No baseline (round 0) found.")
        return None

    baseline_bleu = baseline_row["bleu_score"].iloc[0]
    threshold     = baseline_bleu * QUALITY_FLOOR
    print(f"\n[WinTicket] Baseline BLEU: {baseline_bleu:.4f}")
    print(f"[WinTicket] Quality floor: {threshold:.4f} ({QUALITY_FLOOR:.0%} of baseline)")

    # Find pruning rounds that meet the quality floor
    pruned = df[df["pruning_round"] > 0].copy()
    passing = pruned[pruned["bleu_score"] >= threshold]

    if passing.empty:
        print("[WinTicket] No pruning round met the quality floor.")
        print("[WinTicket] Using round 1 as fallback winning ticket.")
        winner = pruned.iloc[0] if not pruned.empty else None
    else:
        # Pick the round with highest sparsity among passing rounds
        winner = passing.loc[passing["sparsity_percent"].idxmax()]

    if winner is None:
        print("[WinTicket] No pruning rounds found at all.")
        return None

    round_num = int(winner["pruning_round"])
    print(f"\n[WinTicket] ✓ Winning ticket: Round {round_num}")
    print(f"  BLEU:     {winner['bleu_score']:.4f} "
          f"({winner['bleu_score'] / max(baseline_bleu, 1e-6):.1%} of baseline)")
    print(f"  ROUGE-L:  {winner['rouge_l_score']:.4f}")
    print(f"  PPL:      {winner['perplexity']:.2f}")
    print(f"  Sparsity: {winner['sparsity_percent']:.1f}%")

    # Copy to winning_ticket/ if not already done
    source_dir = os.path.join(MODELS_DIR, f"pruned_round_{round_num}")
    target_dir = os.path.join(MODELS_DIR, "winning_ticket")

    if os.path.exists(source_dir):
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
        print(f"\n[WinTicket] Checkpoint copied → {target_dir}")
    else:
        print(f"[WinTicket] WARNING — Source checkpoint not found: {source_dir}")

    return round_num


if __name__ == "__main__":
    print("=" * 60)
    print("FinVaani — Find Winning Ticket")
    print("=" * 60)
    find_winning_ticket()
