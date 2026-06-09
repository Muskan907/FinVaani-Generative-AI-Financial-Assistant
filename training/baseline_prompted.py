"""
baseline_prompted.py — Model 2: Few-shot prompted mGPT baseline.
Uses fixed few-shot examples to guide the model without any fine-tuning.
Saves outputs to results/qualitative_outputs.csv.
"""

import os
import sys
import csv
import time
import torch

sys.path.insert(0, os.path.dirname(__file__))
from generate import generate_answer, get_device

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_CSV  = os.path.join(RESULTS_DIR, "qualitative_outputs.csv")
MODEL_NAME  = "ai-forever/mGPT"
MODEL_LABEL = "prompted_mgpt"

# ── Fixed few-shot examples (same for all runs) ───────────────────────────────
FEW_SHOT_EN = [
    ("What is CRR?",
     "CRR or Cash Reserve Ratio is the minimum percentage of a bank's total "
     "deposits that must be kept as reserves with RBI."),
    ("What does SEBI regulate?",
     "SEBI or Securities and Exchange Board of India regulates the securities "
     "and capital markets in India."),
    ("What is a repo rate?",
     "Repo rate is the rate at which the Reserve Bank of India lends money to "
     "commercial banks for short-term requirements."),
]

FEW_SHOT_HI = [
    ("सीआरआर क्या है?",
     "सीआरआर यानी नकद आरक्षित अनुपात वह न्यूनतम प्रतिशत है जो बैंकों को अपनी "
     "कुल जमा राशि का RBI के पास रखना होता है।"),
    ("सेबी क्या करती है?",
     "सेबी यानी भारतीय प्रतिभूति और विनिमय बोर्ड भारत में प्रतिभूति बाजार को "
     "नियंत्रित करती है।"),
    ("रेपो रेट क्या होता है?",
     "रेपो रेट वह दर है जिस पर RBI वाणिज्यिक बैंकों को अल्पकालिक जरूरतों के "
     "लिए ऋण देता है।"),
]

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
    print(f"[Prompted] Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    return model, tokenizer


def run_baseline_prompted():
    """Run few-shot prompted generation on 10 sample questions."""
    torch.manual_seed(42)
    device = get_device()
    print(f"[Prompted] Device: {device}")

    model, tokenizer = load_model()
    model = model.to(device)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []
    questions = [(q, "en", FEW_SHOT_EN) for q in SAMPLE_EN] + \
                [(q, "hi", FEW_SHOT_HI) for q in SAMPLE_HI]

    print(f"\n[Prompted] Generating answers for {len(questions)} questions...\n")
    for i, (question, lang, examples) in enumerate(questions, 1):
        t0 = time.time()
        answer = generate_answer(
            model, tokenizer, question, lang,
            few_shot_examples=examples,
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

    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "question", "language",
                                                "answer", "response_ms"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"[Prompted] Saved {len(results)} outputs → {OUTPUT_CSV}")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("FinVaani — Model 2: Few-Shot Prompted mGPT")
    print("=" * 60)
    run_baseline_prompted()
