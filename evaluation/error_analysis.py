"""
error_analysis.py — Hallucination and failure case detection.
Reads qualitative_outputs.csv and flags potential hallucinations.
Saves report to results/error_analysis.csv.
"""

import os
import re
import pandas as pd

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
INPUT_CSV   = os.path.join(RESULTS_DIR, "qualitative_outputs.csv")
OUTPUT_CSV  = os.path.join(RESULTS_DIR, "error_analysis.csv")

# Known Indian financial institutions
INSTITUTIONS = [
    "RBI", "SEBI", "IRDAI", "NPCI", "NCFE", "NSDL", "CDSL",
    "NSE", "BSE", "NABARD", "SIDBI", "NHB", "EXIM", "DICGC",
    "PFRDA", "IRDA", "AMFI", "CIBIL",
]

MODEL_COLS = ["raw_answer", "prompted_answer", "lora_answer", "winning_ticket_answer"]
MODEL_LABELS = {
    "raw_answer":            "Raw mGPT",
    "prompted_answer":       "Prompted mGPT",
    "lora_answer":           "LoRA Fine-tuned",
    "winning_ticket_answer": "Winning Ticket",
}


def extract_numbers(text: str) -> set:
    """Extract all numbers and percentages from text."""
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", str(text)))


def extract_institutions(text: str) -> set:
    """Extract institution names mentioned in text."""
    return {inst for inst in INSTITUTIONS if inst in str(text)}


def is_hallucination(answer: str, reference: str, question: str) -> tuple:
    """
    Flag a response as potential hallucination based on 4 rules.

    Returns:
        (is_hallucination: bool, reasons: list of str)
    """
    reasons = []
    answer = str(answer).strip()
    reference = str(reference).strip()
    question = str(question).strip()

    # Rule 1: Contains a number/percentage not in reference
    ans_nums = extract_numbers(answer)
    ref_nums = extract_numbers(reference)
    novel_nums = ans_nums - ref_nums
    if novel_nums:
        reasons.append(f"Novel numbers not in reference: {novel_nums}")

    # Rule 2: Mentions institution not in question or reference
    ans_insts = extract_institutions(answer)
    ref_insts = extract_institutions(reference)
    q_insts   = extract_institutions(question)
    novel_insts = ans_insts - ref_insts - q_insts
    if novel_insts:
        reasons.append(f"Novel institutions not in reference: {novel_insts}")

    # Rule 3: Answer too short (evasion / failure)
    word_count = len(answer.split())
    if word_count < 20:
        reasons.append(f"Very short answer ({word_count} words) — possible evasion")

    # Rule 4: Answer repeats question without adding information
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    if len(q_words) > 5 and len(a_words) > 0:
        overlap = len(q_words & a_words) / len(a_words)
        if overlap > 0.7 and word_count < 40:
            reasons.append(f"Answer mostly repeats question (overlap={overlap:.0%})")

    return bool(reasons), reasons


def run_error_analysis():
    """Analyse all model outputs for hallucinations and failures."""
    if not os.path.exists(INPUT_CSV):
        print(f"[Error] Input file not found: {INPUT_CSV}")
        print("[Error] Run qualitative_analysis.py first.")
        return

    df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    print(f"[Error] Loaded {len(df)} rows from {INPUT_CSV}")

    # Check which model columns exist
    available_cols = [c for c in MODEL_COLS if c in df.columns]
    if not available_cols:
        print("[Error] No model answer columns found in input CSV.")
        return

    report_rows = []
    summary = {}

    for col in available_cols:
        label = MODEL_LABELS.get(col, col)
        hallucination_count = 0
        evasion_count = 0
        domain_confusion_count = 0
        total = 0

        for _, row in df.iterrows():
            answer    = str(row.get(col, ""))
            reference = str(row.get("reference_answer", ""))
            question  = str(row.get("question", ""))

            is_hall, reasons = is_hallucination(answer, reference, question)
            is_evasion = any("short answer" in r for r in reasons)
            is_domain  = any("institution" in r for r in reasons)

            if is_hall:
                hallucination_count += 1
            if is_evasion:
                evasion_count += 1
            if is_domain:
                domain_confusion_count += 1
            total += 1

            report_rows.append({
                "model":             label,
                "question":          question[:80],
                "language":          row.get("language", ""),
                "answer":            answer[:200],
                "reference":         reference[:200],
                "is_hallucination":  int(is_hall),
                "reasons":           "; ".join(reasons) if reasons else "OK",
            })

        hall_rate   = 100 * hallucination_count / max(total, 1)
        evasion_rate = 100 * evasion_count / max(total, 1)
        domain_rate  = 100 * domain_confusion_count / max(total, 1)

        summary[label] = {
            "total":                total,
            "hallucination_count":  hallucination_count,
            "hallucination_rate_%": round(hall_rate, 1),
            "evasion_count":        evasion_count,
            "evasion_rate_%":       round(evasion_rate, 1),
            "domain_confusion_%":   round(domain_rate, 1),
        }

        print(f"\n[Error] {label}:")
        print(f"  Hallucination rate : {hall_rate:.1f}%  ({hallucination_count}/{total})")
        print(f"  Evasion rate       : {evasion_rate:.1f}%  ({evasion_count}/{total})")
        print(f"  Domain confusion   : {domain_rate:.1f}%  ({domain_confusion_count}/{total})")

    # Save detailed report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n[Error] Detailed report saved → {OUTPUT_CSV}")

    # Save summary
    summary_df = pd.DataFrame(summary).T
    summary_path = os.path.join(RESULTS_DIR, "error_summary.csv")
    summary_df.to_csv(summary_path, encoding="utf-8")
    print(f"[Error] Summary saved → {summary_path}")

    return summary_df


if __name__ == "__main__":
    print("=" * 60)
    print("FinVaani — Error Analysis")
    print("=" * 60)
    run_error_analysis()
