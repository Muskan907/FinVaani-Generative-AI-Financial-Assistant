"""
compute_metrics.py — Compute BLEU, ROUGE-L, Perplexity, speed for all 4 models.
Handles both standard LoRA adapters and pruned adapters (weight_mask format).
Saves results to results/metrics_table.csv and the SQLite database.
"""

import os
import sys
import math
import time
import sqlite3
import pandas as pd
import torch
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from generate import generate_answer, get_device

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
TEST_CSV    = os.path.join(BASE_DIR, "data", "splits", "test.csv")
VAL_CSV     = os.path.join(BASE_DIR, "data", "splits", "val.csv")
DB_PATH     = os.path.join(BASE_DIR, "data", "database", "project.db")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_CSV = os.path.join(RESULTS_DIR, "metrics_table.csv")

MODEL_NAME    = "ai-forever/mGPT"
SPEED_SAMPLES = 10   # questions for speed benchmark (reduced for MPS)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_pruned_model(adapter_dir: str, device: str):
    """
    Load a pruned LoRA model that uses PyTorch weight_mask re-parametrization.
    The safetensors file contains both 'weight' and 'weight_mask' tensors.
    We apply the mask manually after loading.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from safetensors import safe_open

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, adapter_dir)

    # Apply weight masks manually
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
                            # Navigate to the parameter and apply mask
                            parts = mk.replace("base_model.model.", "").split(".")
                            # parts like: transformer.h.0.attn.c_attn.lora_A.weight_mask
                            param_name = mk.replace("weight_mask", "weight")
                            for name, param in model.named_parameters():
                                clean = "base_model.model." + name
                                if clean == param_name or name == param_name:
                                    param.data.copy_(
                                        weight.to(param.device) * mask.to(param.device)
                                    )
                                    break

    return model.eval().to(device), tokenizer


def load_model_for_eval(model_type: str, device: str):
    """Load a model by type. Returns (model, tokenizer)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    models_dir = os.path.join(BASE_DIR, "models")

    if model_type in ("raw_mgpt", "prompted_mgpt"):
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
        return m.eval().to(device), tok

    elif model_type == "lora_finetuned":
        lora_dir = os.path.join(models_dir, "lora_finetuned")
        tok = AutoTokenizer.from_pretrained(lora_dir)
        tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
        m = PeftModel.from_pretrained(base, lora_dir)
        return m.eval().to(device), tok

    elif model_type == "winning_ticket":
        wt_dir = os.path.join(models_dir, "winning_ticket")
        if not os.path.exists(wt_dir) or not os.listdir(wt_dir):
            wt_dir = os.path.join(models_dir, "lora_finetuned")
            print("  [Metrics] winning_ticket not found, using lora_finetuned")
        return load_pruned_model(wt_dir, device)

    raise ValueError(f"Unknown model type: {model_type}")


# ── Metric computation ────────────────────────────────────────────────────────

def compute_bleu_rouge(predictions: list, references: list) -> dict:
    """Compute BLEU and ROUGE-L scores."""
    import evaluate
    bleu_metric  = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    # Filter empty predictions
    valid = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    if not valid:
        return {"bleu": 0.0, "rouge_l": 0.0}
    preds, refs = zip(*valid)

    bleu  = bleu_metric.compute(predictions=list(preds),
                                 references=[[r] for r in refs])
    rouge = rouge_metric.compute(predictions=list(preds),
                                  references=list(refs))
    return {
        "bleu":    round(bleu["bleu"], 4),
        "rouge_l": round(rouge["rougeL"], 4),
    }


def compute_perplexity(model, tokenizer, texts: list, device: str) -> float:
    """Compute average perplexity over a list of formatted text strings."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for text in texts[:30]:   # 30 samples is enough for a reliable estimate
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=128).to(device)
            labels = inputs["input_ids"].clone()
            out = model(**inputs, labels=labels)
            total_loss += out.loss.item()
            count += 1
    avg_loss = total_loss / max(count, 1)
    return round(math.exp(min(avg_loss, 20)), 2)


def measure_speed(model, tokenizer, questions: list, device: str) -> dict:
    """Measure average inference time in ms over N questions."""
    times = []
    for q in questions[:SPEED_SAMPLES]:
        t0 = time.time()
        generate_answer(model, tokenizer, q, "en",
                        max_new_tokens=100, device=device)
        times.append((time.time() - t0) * 1000)
    mean_ms = round(sum(times) / len(times), 1) if times else 0
    std_ms  = round((sum((t - mean_ms)**2 for t in times) / max(len(times), 1))**0.5, 1)
    return {"mean_ms": mean_ms, "std_ms": std_ms}


def get_sparsity_from_adapter(adapter_dir: str) -> float:
    """Read sparsity from the adapter's weight_mask tensors."""
    from safetensors import safe_open
    path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(path):
        return 0.0
    try:
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
        mask_keys = [k for k in keys if k.endswith("weight_mask")]
        if not mask_keys:
            return 0.0
        total = zeros = 0
        with safe_open(path, framework="pt") as f:
            for mk in mask_keys:
                m = f.get_tensor(mk)
                total += m.numel()
                zeros += (m == 0).sum().item()
        return round(100.0 * zeros / max(total, 1), 1)
    except Exception:
        return 0.0


def evaluate_model_on_test(model, tokenizer, test_df: pd.DataFrame,
                            device: str, model_label: str,
                            few_shot_en=None, few_shot_hi=None) -> dict:
    """Run full evaluation for one model on the test set."""
    predictions_all, references_all = [], []
    predictions_en,  references_en  = [], []
    predictions_hi,  references_hi  = [], []

    print(f"  Generating answers for {len(test_df)} test questions...")
    for _, row in test_df.iterrows():
        lang = str(row["language"])
        few_shot = few_shot_en if lang == "en" else few_shot_hi
        pred = generate_answer(
            model, tokenizer,
            str(row["question"]), lang,
            few_shot_examples=few_shot,
            max_new_tokens=100, device=device,
        )
        ref = str(row["answer"])
        predictions_all.append(pred)
        references_all.append(ref)
        if lang == "en":
            predictions_en.append(pred); references_en.append(ref)
        else:
            predictions_hi.append(pred); references_hi.append(ref)

    scores_all = compute_bleu_rouge(predictions_all, references_all)
    scores_en  = compute_bleu_rouge(predictions_en,  references_en)
    scores_hi  = compute_bleu_rouge(predictions_hi,  references_hi)

    ppl   = compute_perplexity(model, tokenizer,
                                test_df["formatted"].tolist(), device)
    speed = measure_speed(model, tokenizer,
                          test_df["question"].tolist(), device)

    total_params  = sum(p.numel() for p in model.parameters())
    active_params = sum((p != 0).sum().item() for p in model.parameters())

    return {
        "model":           model_label,
        "bleu_all":        scores_all["bleu"],
        "bleu_en":         scores_en["bleu"],
        "bleu_hi":         scores_hi["bleu"],
        "rouge_l_all":     scores_all["rouge_l"],
        "rouge_l_en":      scores_en["rouge_l"],
        "rouge_l_hi":      scores_hi["rouge_l"],
        "perplexity":      ppl,
        "total_params_M":  round(total_params  / 1e6, 1),
        "active_params_M": round(active_params / 1e6, 1),
        "speed_mean_ms":   speed["mean_ms"],
        "speed_std_ms":    speed["std_ms"],
    }


# ── Few-shot examples for prompted model ─────────────────────────────────────
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


def run_all_evaluations():
    """Evaluate all 4 models and save results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    print(f"[Metrics] Device: {device}")

    test_df = pd.read_csv(TEST_CSV, encoding="utf-8")
    # Use 20 samples (10 EN + 10 HI) — full 60-sample eval takes ~40min on MPS
    en_sample = test_df[test_df["language"] == "en"].sample(
        min(10, len(test_df[test_df["language"] == "en"])), random_state=42)
    hi_sample = test_df[test_df["language"] == "hi"].sample(
        min(10, len(test_df[test_df["language"] == "hi"])), random_state=42)
    test_sample = pd.concat([en_sample, hi_sample]).reset_index(drop=True)
    print(f"[Metrics] Evaluating on {len(test_sample)} samples "
          f"({len(en_sample)} EN + {len(hi_sample)} HI)")

    conn = sqlite3.connect(DB_PATH)
    all_results = []

    models_to_eval = [
        ("raw_mgpt",       "Raw mGPT",        None,        None),
        ("prompted_mgpt",  "Prompted mGPT",   FEW_SHOT_EN, FEW_SHOT_HI),
        ("lora_finetuned", "LoRA Fine-tuned",  None,        None),
        ("winning_ticket", "Winning Ticket",   None,        None),
    ]

    for model_type, label, fs_en, fs_hi in models_to_eval:
        print(f"\n[Metrics] ── Evaluating: {label} ──")
        try:
            model, tokenizer = load_model_for_eval(model_type, device)

            # Get sparsity for pruned models
            models_dir = os.path.join(BASE_DIR, "models")
            if model_type == "winning_ticket":
                sparsity = get_sparsity_from_adapter(
                    os.path.join(models_dir, "winning_ticket"))
            else:
                sparsity = 0.0

            metrics = evaluate_model_on_test(
                model, tokenizer, test_sample, device, label, fs_en, fs_hi)
            metrics["sparsity_pct"] = sparsity
            all_results.append(metrics)

            # Save to DB
            conn.execute(
                "INSERT INTO evaluation_results "
                "(model_name, language, bleu_score, rouge_l_score, perplexity, "
                "sparsity_percent, pruning_round, timestamp) VALUES (?,?,?,?,?,?,?,?)",
                (label, "all", metrics["bleu_all"], metrics["rouge_l_all"],
                 metrics["perplexity"], sparsity, 0, datetime.now().isoformat()),
            )
            conn.commit()

            print(f"  BLEU={metrics['bleu_all']:.4f}  "
                  f"ROUGE-L={metrics['rouge_l_all']:.4f}  "
                  f"PPL={metrics['perplexity']:.1f}  "
                  f"Speed={metrics['speed_mean_ms']:.0f}ms")

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR evaluating {label}: {e}")
            import traceback; traceback.print_exc()
            all_results.append({"model": label, "error": str(e)})

    conn.close()

    # Save metrics table
    results_df = pd.DataFrame([r for r in all_results if "error" not in r])
    results_df.to_csv(METRICS_CSV, index=False, encoding="utf-8")
    print(f"\n[Metrics] Saved → {METRICS_CSV}")

    # Print comparison table
    print("\n" + "=" * 85)
    print(f"{'Model':<20} {'BLEU':>6} {'ROUGE-L':>8} {'PPL':>8} "
          f"{'Params':>8} {'Sparsity':>9} {'Speed':>8}")
    print("-" * 85)
    for r in all_results:
        if "error" not in r:
            print(f"{r['model']:<20} {r['bleu_all']:>6.4f} {r['rouge_l_all']:>8.4f} "
                  f"{r['perplexity']:>8.1f} {r['total_params_M']:>7.1f}M "
                  f"{r.get('sparsity_pct', 0):>8.1f}% "
                  f"{r['speed_mean_ms']:>6.0f}ms")
    print("=" * 85)

    return results_df


if __name__ == "__main__":
    print("=" * 60)
    print("FinVaani — Evaluation: All Models")
    print("=" * 60)
    run_all_evaluations()
