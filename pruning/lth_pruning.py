"""
lth_pruning.py — Model 4: Iterative Magnitude Pruning (Lottery Ticket Hypothesis).
Loads the LoRA fine-tuned model and applies 5 rounds of IMP.
"""

import os
import sys
import copy
import shutil
import sqlite3
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from pruning_utils import (
    apply_magnitude_pruning,
    get_sparsity_percent,
    get_param_count,
    save_pruning_mask,
    reset_to_original_weights,
    make_pruning_permanent,
)
from generate import generate_answer, get_device

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.join(os.path.dirname(__file__), "..")
LORA_DIR      = os.path.join(BASE_DIR, "models", "lora_finetuned")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
DB_PATH       = os.path.join(BASE_DIR, "data", "database", "project.db")
TRAIN_CSV     = os.path.join(BASE_DIR, "data", "splits", "train.csv")
VAL_CSV       = os.path.join(BASE_DIR, "data", "splits", "val.csv")
TEST_CSV      = os.path.join(BASE_DIR, "data", "splits", "test.csv")

MODEL_NAME    = "ai-forever/mGPT"
MAX_ROUNDS    = 5
PRUNE_AMOUNT  = 0.20   # Prune 20% of remaining weights per round
QUALITY_FLOOR = 0.85   # Stop if BLEU drops below 85% of baseline
SEED          = 42
MAX_LENGTH    = 512


def load_and_tokenize(tokenizer, csv_path: str) -> Dataset:
    """Tokenize a CSV split for training."""
    df = pd.read_csv(csv_path, encoding="utf-8").dropna(subset=["formatted"])

    def tokenize(batch):
        tokens = tokenizer(batch["formatted"], truncation=True,
                           max_length=MAX_LENGTH, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    ds = Dataset.from_pandas(df[["formatted"]])
    ds = ds.map(tokenize, batched=True, remove_columns=["formatted"])
    ds.set_format("torch")
    return ds


def evaluate_model(model, tokenizer, device: str) -> dict:
    """
    Quick evaluation: compute average cross-entropy loss on validation set
    and convert to perplexity. Also compute a simple BLEU proxy.
    Returns dict with bleu, rouge_l, perplexity.
    """
    import math
    from torch.utils.data import DataLoader

    val_ds = load_and_tokenize(tokenizer, VAL_CSV)
    loader = DataLoader(val_ds, batch_size=4)

    model.eval()
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            out = model(input_ids=input_ids, labels=labels)
            total_loss += out.loss.item()
            n_batches  += 1
            if n_batches >= 20:   # Cap at 20 batches for speed
                break

    avg_loss   = total_loss / max(n_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))   # Cap to avoid overflow

    # Proxy BLEU: inverse of normalised perplexity (for relative comparison)
    # Real BLEU is computed in evaluation/compute_metrics.py
    proxy_bleu  = max(0.0, 1.0 - avg_loss / 10.0)
    proxy_rouge = max(0.0, 1.0 - avg_loss / 12.0)

    return {
        "bleu":       round(proxy_bleu,  4),
        "rouge_l":    round(proxy_rouge, 4),
        "perplexity": round(perplexity,  2),
        "eval_loss":  round(avg_loss,    4),
    }


def retrain_one_epoch(model, tokenizer, output_dir: str, device: str):
    """Fine-tune the pruned model for 1 epoch (LTH retrain step)."""
    train_ds = load_and_tokenize(tokenizer, TRAIN_CSV)
    val_ds   = load_and_tokenize(tokenizer, VAL_CSV)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=50,
        weight_decay=0.01,
        learning_rate=2e-4,
        fp16=(device == "cuda"),
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    return model


def save_metrics(conn, model_name, metrics, sparsity, round_num):
    """Insert evaluation metrics into the database."""
    from datetime import datetime
    conn.execute(
        "INSERT INTO evaluation_results "
        "(model_name, language, bleu_score, rouge_l_score, perplexity, "
        "sparsity_percent, pruning_round, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (model_name, "all",
         metrics["bleu"], metrics["rouge_l"], metrics["perplexity"],
         sparsity, round_num,
         datetime.now().isoformat()),
    )
    conn.commit()


def run_lth():
    """Full Iterative Magnitude Pruning loop."""
    torch.manual_seed(SEED)
    device = get_device()
    print(f"[LTH] Device: {device}")

    if not os.path.exists(LORA_DIR):
        raise FileNotFoundError(
            f"LoRA checkpoint not found at {LORA_DIR}. "
            "Run training/lora_finetune.py first."
        )

    conn = sqlite3.connect(DB_PATH)

    # ── Load LoRA fine-tuned model ────────────────────────────────────────────
    print(f"[LTH] Loading LoRA model from {LORA_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model = model.to(device)

    # ── Save original weights (CRITICAL for LTH) ─────────────────────────────
    print("[LTH] Saving original LoRA weights for LTH reset...")
    original_state_dict = copy.deepcopy(model.state_dict())

    # ── Baseline evaluation ───────────────────────────────────────────────────
    print("\n[LTH] Evaluating baseline (round 0)...")
    baseline = evaluate_model(model, tokenizer, device)
    print(f"[LTH] Baseline — BLEU: {baseline['bleu']:.4f}, "
          f"ROUGE-L: {baseline['rouge_l']:.4f}, PPL: {baseline['perplexity']:.2f}")
    save_metrics(conn, "lora_finetuned", baseline, 0.0, 0)

    best_round   = 0
    best_metrics = baseline
    best_sparsity = 0.0

    # ── Pruning loop ──────────────────────────────────────────────────────────
    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'=' * 50}")
        print(f"[LTH] PRUNING ROUND {round_num}/{MAX_ROUNDS}")
        print(f"{'=' * 50}")

        # a) Prune 20% of remaining weights
        model = apply_magnitude_pruning(model, amount=PRUNE_AMOUNT)
        sparsity = get_sparsity_percent(model)
        print(f"[LTH] Sparsity after pruning: {sparsity:.1f}%")

        # b) LTH RESET: restore original weight values, keep mask
        model = reset_to_original_weights(model, original_state_dict)

        # c) Retrain for 1 epoch
        round_dir = os.path.join(MODELS_DIR, f"pruned_round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)
        print(f"[LTH] Retraining for 1 epoch...")
        model = retrain_one_epoch(model, tokenizer, round_dir, device)

        # d) Evaluate
        metrics = evaluate_model(model, tokenizer, device)
        bleu_retained = metrics["bleu"] / max(baseline["bleu"], 1e-6)
        print(f"[LTH] Round {round_num} — "
              f"BLEU: {metrics['bleu']:.4f} ({bleu_retained:.1%} of baseline), "
              f"ROUGE-L: {metrics['rouge_l']:.4f}, "
              f"PPL: {metrics['perplexity']:.2f}, "
              f"Sparsity: {sparsity:.1f}%")

        # e) Save checkpoint and masks
        make_pruning_permanent(model)
        model.save_pretrained(round_dir)
        tokenizer.save_pretrained(round_dir)
        save_pruning_mask(model, round_num, conn)
        save_metrics(conn, f"pruned_round_{round_num}", metrics, sparsity, round_num)

        # f) Track best
        if bleu_retained >= QUALITY_FLOOR:
            best_round    = round_num
            best_metrics  = metrics
            best_sparsity = sparsity
            print(f"[LTH] ✓ Quality maintained. Continuing pruning.")
        else:
            print(f"[LTH] ✗ Quality degraded below {QUALITY_FLOOR:.0%}. "
                  f"Round {round_num - 1} is the winning ticket.")
            break

    # ── Save winning ticket ───────────────────────────────────────────────────
    winning_dir = os.path.join(MODELS_DIR, "winning_ticket")
    best_source = os.path.join(MODELS_DIR, f"pruned_round_{best_round}")

    if os.path.exists(best_source):
        if os.path.exists(winning_dir):
            shutil.rmtree(winning_dir)
        shutil.copytree(best_source, winning_dir)
        print(f"\n[LTH] Winning ticket (round {best_round}) saved → {winning_dir}")

    # ── Final summary ─────────────────────────────────────────────────────────
    params = get_param_count(model)
    print(f"\n{'=' * 50}")
    print(f"[LTH] FINAL SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Winning ticket round : {best_round}")
    print(f"  Sparsity achieved    : {best_sparsity:.1f}%")
    print(f"  Active LoRA params   : {params['active_lora_params']:,}")
    print(f"  BLEU retained        : {best_metrics['bleu'] / max(baseline['bleu'], 1e-6):.1%}")
    print(f"  Final perplexity     : {best_metrics['perplexity']:.2f}")

    conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("FinVaani — Model 4: Lottery Ticket Hypothesis Pruning")
    print("=" * 60)
    run_lth()
