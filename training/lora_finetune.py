"""
lora_finetune.py — Model 3: LoRA fine-tuning of mGPT on Indian finance Q&A.

NOTE: ai-forever/mGPT is 1.4B parameters (not 117M — that's a different variant).

Optimised for LOCAL training on Apple Silicon (MPS) with 16-17GB RAM.
Key memory optimisations for 1.4B model on 17GB RAM:
  - gradient_checkpointing=True  (trades compute for memory, ~40% RAM reduction)
  - batch_size=1, grad_accum=16  (effective batch=16, minimal peak memory)
  - max_length=128               (shorter sequences = less memory per step)
  - LoRA r=4                     (half the adapters vs r=8)
  - Only lora_A/B, no full layer  (0.15% trainable params)

Also works on CUDA (Colab T4) and CPU.
Saves checkpoint to models/lora_finetuned/.
"""

import os
import sys
import math
import time
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
from peft import LoraConfig, TaskType, get_peft_model

sys.path.insert(0, os.path.dirname(__file__))
from generate import generate_answer, get_device

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
TRAIN_CSV  = os.path.join(BASE_DIR, "data", "splits", "train.csv")
VAL_CSV    = os.path.join(BASE_DIR, "data", "splits", "val.csv")
TEST_CSV   = os.path.join(BASE_DIR, "data", "splits", "test.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "lora_finetuned")

MODEL_NAME = "ai-forever/mGPT"
SEED       = 42


def get_hardware_config(device: str) -> dict:
    """
    Return training hyperparameters tuned for the available hardware.
    mGPT is 1.4B params — memory is the primary constraint.

    MPS (Apple Silicon 17GB):
      - max_length=128, batch=1, grad_accum=16 → effective batch=16
      - gradient_checkpointing=True saves ~40% RAM
      - Estimated: ~5 min/epoch, ~15 min total

    CUDA (Colab T4 16GB):
      - max_length=256, batch=4, fp16=True → much faster
      - Estimated: ~3 min/epoch, ~9 min total

    CPU (fallback):
      - max_length=64, 1 epoch only (very slow, ~2h)
    """
    if device == "cuda":
        return dict(
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            fp16=True,
            bf16=False,
            dtype=torch.float16,
            max_length=256,
            num_epochs=3,
            gradient_checkpointing=True,
        )
    elif device == "mps":
        return dict(
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            fp16=False,   # MPS doesn't support fp16 training reliably
            bf16=False,
            dtype=torch.float32,
            max_length=128,
            num_epochs=3,
            gradient_checkpointing=True,
        )
    else:  # CPU
        return dict(
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            fp16=False,
            bf16=False,
            dtype=torch.float32,
            max_length=64,
            num_epochs=1,
            gradient_checkpointing=False,
        )


def load_and_tokenize(tokenizer, csv_path: str, max_length: int) -> Dataset:
    """Load a CSV split and tokenize the 'formatted' column."""
    df = pd.read_csv(csv_path, encoding="utf-8").dropna(subset=["formatted"])

    def tokenize(batch):
        tokens = tokenizer(
            batch["formatted"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    ds = Dataset.from_pandas(df[["formatted"]])
    ds = ds.map(tokenize, batched=True, remove_columns=["formatted"])
    ds.set_format("torch")
    return ds


def configure_lora(model):
    """
    Apply LoRA adapters to the mGPT (GPT-2 architecture) model.
    Uses r=4 (half of standard) to reduce memory on 1.4B model.
    """
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,            # Reduced from 8 for memory efficiency on 1.4B model
        lora_alpha=16,  # Keep alpha/r ratio = 4
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / total

    print(f"\n[LoRA] Trainable parameters: {trainable:,} / {total:,} ({pct:.3f}%)")
    print(f"[LoRA] Base model size: {(total - trainable) * 4 / 1e9:.2f} GB (fp32)")
    return model


def train():
    """Full LoRA fine-tuning pipeline — runs locally on MPS/CUDA/CPU."""
    torch.manual_seed(SEED)
    device = get_device()
    hw     = get_hardware_config(device)

    print("=" * 60)
    print(f"[LoRA] Device             : {device.upper()}")
    print(f"[LoRA] Batch size         : {hw['per_device_train_batch_size']} "
          f"(effective: {hw['per_device_train_batch_size'] * hw['gradient_accumulation_steps']})")
    print(f"[LoRA] Epochs             : {hw['num_epochs']}")
    print(f"[LoRA] Precision          : {'fp16' if hw['fp16'] else 'fp32'}")
    print(f"[LoRA] Max seq len        : {hw['max_length']}")
    print(f"[LoRA] Grad checkpointing : {hw['gradient_checkpointing']}")
    print("=" * 60)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print(f"\n[LoRA] Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[LoRA] Loading model {MODEL_NAME} (1.4B params)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=hw["dtype"],
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    print(f"[LoRA] Model loaded in {time.time()-t0:.1f}s")

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    model = configure_lora(model)

    # Enable gradient checkpointing BEFORE moving to device
    if hw["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print("[LoRA] Gradient checkpointing enabled")

    model = model.to(device)

    # ── Tokenize datasets ─────────────────────────────────────────────────────
    print("\n[LoRA] Tokenizing training data...")
    train_ds = load_and_tokenize(tokenizer, TRAIN_CSV, hw["max_length"])
    val_ds   = load_and_tokenize(tokenizer, VAL_CSV,   hw["max_length"])
    print(f"[LoRA] Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # Estimate training time
    steps_per_epoch = math.ceil(
        len(train_ds) / (hw["per_device_train_batch_size"] * hw["gradient_accumulation_steps"])
    )
    total_steps = steps_per_epoch * hw["num_epochs"]
    secs_per_step = 8 if device == "mps" else (1 if device == "cuda" else 30)
    est_mins = (total_steps * secs_per_step) / 60
    print(f"[LoRA] Steps: {total_steps} (~{est_mins:.0f} min estimated on {device.upper()})")

    # ── Training arguments ────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=hw["num_epochs"],
        per_device_train_batch_size=hw["per_device_train_batch_size"],
        per_device_eval_batch_size=hw["per_device_eval_batch_size"],
        gradient_accumulation_steps=hw["gradient_accumulation_steps"],
        gradient_checkpointing=hw["gradient_checkpointing"],
        warmup_steps=50,
        weight_decay=0.01,
        learning_rate=2e-4,
        fp16=hw["fp16"],
        bf16=hw["bf16"],
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=SEED,
        dataloader_num_workers=0,
        dataloader_pin_memory=(device == "cuda"),
        optim="adamw_torch",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[LoRA] Starting training on {device.upper()}...")
    t_start = time.time()
    trainer.train()
    elapsed = time.time() - t_start
    print(f"\n[LoRA] Training complete in {elapsed/60:.1f} minutes")

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[LoRA] Model saved → {OUTPUT_DIR}")

    # ── Quick validation ──────────────────────────────────────────────────────
    print("\n[LoRA] Quick validation on 5 test questions:")
    test_df = pd.read_csv(TEST_CSV, encoding="utf-8")
    samples = test_df.sample(5, random_state=SEED)
    model.eval()
    for _, row in samples.iterrows():
        ans = generate_answer(
            model, tokenizer,
            str(row["question"]), str(row["language"]),
            max_new_tokens=60, device=device,
        )
        lang_tag = "[EN]" if row["language"] == "en" else "[HI]"
        print(f"  {lang_tag} Q: {str(row['question'])[:65]}")
        print(f"       A: {ans[:100]}\n")

    return model, tokenizer


if __name__ == "__main__":
    print("=" * 60)
    print("FinVaani — Model 3: LoRA Fine-tuning (Local MPS/CUDA/CPU)")
    print("=" * 60)
    train()
