"""
qualitative_analysis.py — Side-by-side output comparison for all 4 models.
Selects 25 questions (10 easy EN, 5 hard EN, 5 easy HI, 5 hard HI).
Handles both standard LoRA and pruned (weight_mask) adapters.
Saves to results/qualitative_outputs.csv.
"""

import os
import sys
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from generate import generate_answer, get_device

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
TEST_CSV    = os.path.join(BASE_DIR, "data", "splits", "test.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_CSV  = os.path.join(RESULTS_DIR, "qualitative_outputs.csv")
MODEL_NAME  = "ai-forever/mGPT"

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


def classify_difficulty(question: str, answer: str) -> str:
    """Simple heuristic: long answers with technical terms = hard."""
    technical_terms = [
        "MCLR", "SLR", "LAF", "OMO", "NDTL", "CRAR", "NPA", "IBC",
        "FEMA", "PMLA", "SARFAESI", "DICGC", "NBFC-MFI",
        "एनपीए", "एसएलआर", "एलएएफ",
    ]
    word_count = len(answer.split())
    has_technical = any(t in question or t in answer for t in technical_terms)
    return "hard" if (word_count > 120 or has_technical) else "easy"


def load_pruned_model(adapter_dir: str, device: str):
    """Load a pruned LoRA model with weight_mask tensors applied."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from safetensors import safe_open

    tok = AutoTokenizer.from_pretrained(adapter_dir)
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, adapter_dir)

    adapter_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        with safe_open(adapter_path, framework="pt") as f:
            keys = list(f.keys())
        mask_keys = [k for k in keys if k.endswith("weight_mask")]
        if mask_keys:
            with torch.no_grad():
                with safe_open(adapter_path, framework="pt") as f:
                    for mk in mask_keys:
                        wk = mk.replace("weight_mask", "weight")
                        if wk in keys:
                            mask   = f.get_tensor(mk)
                            weight = f.get_tensor(wk)
                            for name, param in model.named_parameters():
                                if "base_model.model." + name == wk or name == wk:
                                    param.data.copy_(
                                        weight.to(param.device) * mask.to(param.device))
                                    break
    return model.eval().to(device), tok


def load_all_models(device: str) -> dict:
    """Load all 4 models. Returns dict of {key: (model, tokenizer, few_shot)}."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    models_dir = os.path.join(BASE_DIR, "models")
    loaded = {}

    print("[Qual] Loading Raw mGPT...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    raw = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32).eval().to(device)
    loaded["raw"]      = (raw, tok, None)
    loaded["prompted"] = (raw, tok, {"en": FEW_SHOT_EN, "hi": FEW_SHOT_HI})

    print("[Qual] Loading LoRA fine-tuned...")
    lora_dir = os.path.join(models_dir, "lora_finetuned")
    if os.path.exists(lora_dir):
        tok3 = AutoTokenizer.from_pretrained(lora_dir)
        tok3.pad_token = tok3.eos_token
        base3 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
        m3 = PeftModel.from_pretrained(base3, lora_dir).eval().to(device)
        loaded["lora"] = (m3, tok3, None)
    else:
        loaded["lora"] = (raw, tok, None)

    print("[Qual] Loading Winning Ticket...")
    wt_dir = os.path.join(models_dir, "winning_ticket")
    if os.path.exists(wt_dir) and os.listdir(wt_dir):
        m4, tok4 = load_pruned_model(wt_dir, device)
        loaded["winning_ticket"] = (m4, tok4, None)
    else:
        loaded["winning_ticket"] = (raw, tok, None)

    return loaded


def select_25_questions(test_df: pd.DataFrame) -> pd.DataFrame:
    """Select 25 representative questions."""
    test_df = test_df.copy()
    test_df["difficulty"] = test_df.apply(
        lambda r: classify_difficulty(str(r["question"]), str(r["answer"])), axis=1)

    en_df = test_df[test_df["language"] == "en"]
    hi_df = test_df[test_df["language"] == "hi"]

    easy_en = en_df[en_df["difficulty"] == "easy"].sample(
        min(10, len(en_df[en_df["difficulty"] == "easy"])), random_state=42)
    hard_en = en_df[en_df["difficulty"] == "hard"].sample(
        min(5, len(en_df[en_df["difficulty"] == "hard"])), random_state=42)
    easy_hi = hi_df[hi_df["difficulty"] == "easy"].sample(
        min(5, len(hi_df[hi_df["difficulty"] == "easy"])), random_state=42)
    hard_hi = hi_df[hi_df["difficulty"] == "hard"].sample(
        min(5, len(hi_df[hi_df["difficulty"] == "hard"])), random_state=42)

    selected = pd.concat([easy_en, hard_en, easy_hi, hard_hi]).reset_index(drop=True)
    print(f"[Qual] Selected {len(selected)} questions: "
          f"{len(easy_en)} easy EN, {len(hard_en)} hard EN, "
          f"{len(easy_hi)} easy HI, {len(hard_hi)} hard HI")
    return selected


def run_qualitative_analysis():
    """Generate side-by-side outputs for all 4 models on 25 questions."""
    torch.manual_seed(42)
    device = get_device()
    print(f"[Qual] Device: {device}")

    test_df  = pd.read_csv(TEST_CSV, encoding="utf-8")
    selected = select_25_questions(test_df)
    models   = load_all_models(device)

    rows = []
    for i, row in selected.iterrows():
        q    = str(row["question"])
        lang = str(row["language"])
        ref  = str(row["answer"])
        diff = str(row["difficulty"])

        print(f"[{i+1:02d}] [{lang.upper()}][{diff}] {q[:60]}...")

        raw_ans = generate_answer(
            *models["raw"][:2], q, lang, max_new_tokens=100, device=device)

        fs = models["prompted"][2]
        prompted_ans = generate_answer(
            *models["prompted"][:2], q, lang,
            few_shot_examples=fs[lang] if fs else None,
            max_new_tokens=100, device=device)

        lora_ans = generate_answer(
            *models["lora"][:2], q, lang, max_new_tokens=100, device=device)

        wt_ans = generate_answer(
            *models["winning_ticket"][:2], q, lang, max_new_tokens=100, device=device)

        rows.append({
            "question":              q,
            "language":              lang,
            "difficulty":            diff,
            "raw_answer":            raw_ans,
            "prompted_answer":       prompted_ans,
            "lora_answer":           lora_ans,
            "winning_ticket_answer": wt_ans,
            "reference_answer":      ref,
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n[Qual] Saved {len(rows)} rows → {OUTPUT_CSV}")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("FinVaani — Qualitative Analysis")
    print("=" * 60)
    run_qualitative_analysis()
