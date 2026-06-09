"""
baseline_raw.py — Model 1: Zero-shot mGPT baseline (no fine-tuning, no prompting).
Loads ai-forever/mGPT and generates answers directly from the test set.
Saves outputs to results/qualitative_outputs.csv.
"""

import os
import sys
import csv
import time
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from generate import generate_answer, get_device

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.join(os.path.dirname(__file__), "..")
TEST_CSV     = os.path.join(BASE_DIR, "data", "splits", "test.csv")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
OUTPUT_CSV   = os.path.join(RESULTS_DIR, "qualitative_outputs.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
INFO_FILE    = os.path.join(MODEL_DIR, "raw_model_info.txt")

MODEL_NAME   = "ai-forever/mGPT"
MODEL_LABEL  = "raw_mgpt"

# 5 English + 5 Hindi sample questions
SAMPLE_EN = [
    "What is the Cash Reserve Ratio (CRR)?",
    "What is a Non-Banking Financial Company (NBFC)?",
    "What is UPI and how does it work?",
    "What is the difference between NEFT, RTGS, and IMPS?",
    "What is the role of SEBI in regulating the stock market?",
]
SAMPLE_HI = [
    "नकद आरक्षित अनुपात (CRR) क्या है?",
    "रेपो रेट क्या है?",
    "UPI क्या है और यह कैसे काम करता है?",
    "म्यूचुअल फंड क्या है?",
    "वित्तीय समावेशन क्या है?",
]


def load_model():
    """Load mGPT tokenizer and model."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[Raw] Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )
    model.eval()
    return model, tokenizer


def run_baseline_raw():
    """Run zero-shot generation on 10 sample questions and save results."""
    torch.manual_seed(42)
    device = get_device()
    print(f"[Raw] Device: {device}")

    model, tokenizer = load_model()
    model = model.to(device)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Write model info file
    with open(INFO_FILE, "w", encoding="utf-8") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write("Type: Zero-shot baseline (no fine-tuning)\n")
        f.write("HuggingFace: https://huggingface.co/ai-forever/mGPT\n")
        f.write("Parameters: ~117M\n")

    results = []
    questions = [(q, "en") for q in SAMPLE_EN] + [(q, "hi") for q in SAMPLE_HI]

    print(f"\n[Raw] Generating answers for {len(questions)} questions...\n")
    for i, (question, lang) in enumerate(questions, 1):
        t0 = time.time()
        answer = generate_answer(
            model, tokenizer, question, lang,
            max_new_tokens=150, temperature=0.7, device=device,
        )
        elapsed_ms = (time.time() - t0) * 1000

        print(f"[{i:02d}] [{lang.upper()}] Q: {question[:60]}")
        print(f"       A: {answer[:120]}")
        print(f"       Time: {elapsed_ms:.0f}ms\n")

        results.append({
            "model_name":  MODEL_LABEL,
            "question":    question,
            "language":    lang,
            "answer":      answer,
            "response_ms": round(elapsed_ms, 1),
        })

    # Save to CSV (append if file exists)
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "question", "language",
                                                "answer", "response_ms"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"[Raw] Saved {len(results)} outputs → {OUTPUT_CSV}")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("FinVaani — Model 1: Raw mGPT Baseline")
    print("=" * 60)
    run_baseline_raw()
