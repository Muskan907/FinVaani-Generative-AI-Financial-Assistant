# FinVaani — Colab Training & Transfer Guide

## Why Colab instead of your Mac?

| | Your Mac (MPS) | Colab T4 (free) |
|---|---|---|
| Hardware | Apple M-series, 17GB RAM | NVIDIA T4, 16GB VRAM |
| mGPT full training (3 epochs) | ~4–6 hours | **~45–60 min** |
| Precision | float32 only | float16 (2× faster) |
| LoRA fine-tune | Works, slow | Works, fast |
| LTH pruning (5 rounds) | ~2–3 hours | **~30–40 min** |

> **Note on TPUs:** Colab's free TPU is NOT recommended here.
> LoRA + HuggingFace Trainer requires CUDA. TPU needs a full XLA rewrite.
> Stick with T4 GPU — it's free and works with zero code changes.

---

## What gets transferred (it's tiny)

The LoRA adapter is just the **delta weights** on top of the frozen base model.

```
lora_finetuned/
  adapter_config.json     ~  1 KB
  adapter_model.bin       ~ 20 MB   ← only this matters
  tokenizer files         ~  3 MB
  
winning_ticket/
  adapter_config.json     ~  1 KB
  adapter_model.bin       ~  4 MB   ← 80% pruned, even smaller
  tokenizer files         ~  3 MB
```

**Total transfer: ~25 MB** (not 5.6GB — the base model stays in HuggingFace cache on both machines)

---

## Step-by-step: Train on Colab

### 1. Upload your data splits to Colab

On your Mac, your splits are at:
```
finvaani/data/splits/train.csv   (1194 rows)
finvaani/data/splits/val.csv     (256 rows)
finvaani/data/splits/test.csv    (256 rows)
```

In Colab, click the **folder icon** in the left sidebar → upload these 3 files to `/content/`.

### 2. Open the notebook

Upload `finvaani/notebooks/training_colab.ipynb` to Colab, or open it directly from Google Drive.

### 3. Set runtime to T4 GPU

`Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save`

### 4. Run all cells in order

The notebook handles everything:
- Installs dependencies
- Loads mGPT in fp16
- Applies LoRA (r=8, ~8.6M trainable params)
- Trains 3 epochs on full 1194 samples
- Runs LTH pruning (5 rounds)
- Saves everything to Google Drive

---

## Step-by-step: Transfer to your Mac

### Option A — Direct download (simplest)

The last notebook cell zips the winning ticket and triggers a browser download.

After downloading `finvaani_winning_ticket.zip`:
```bash
# On your Mac, from the finvaani/ directory:
unzip ~/Downloads/finvaani_winning_ticket.zip -d models/winning_ticket/

# Also download and unzip lora_finetuned:
unzip ~/Downloads/finvaani_lora_finetuned.zip -d models/lora_finetuned/
```

### Option B — Google Drive (recommended for large files)

1. The notebook auto-saves to `MyDrive/finvaani_models/` on your Drive
2. Install Google Drive desktop app on your Mac (drive.google.com/drive/download)
3. The folder syncs automatically — files appear at:
   ```
   ~/Google Drive/My Drive/finvaani_models/lora_finetuned/
   ~/Google Drive/My Drive/finvaani_models/winning_ticket/
   ```
4. Copy them to your project:
   ```bash
   cp -r ~/Google\ Drive/My\ Drive/finvaani_models/lora_finetuned/ \
         /path/to/finvaani/models/lora_finetuned/
   
   cp -r ~/Google\ Drive/My\ Drive/finvaani_models/winning_ticket/ \
         /path/to/finvaani/models/winning_ticket/
   ```

---

## After transfer: verify it works on your Mac

```bash
cd /path/to/finvaani

/opt/anaconda3/bin/python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Base model is already cached — loads instantly
tokenizer = AutoTokenizer.from_pretrained('ai-forever/mGPT')
base = AutoModelForCausalLM.from_pretrained('ai-forever/mGPT', dtype=torch.float32)

# Load your trained adapter
model = PeftModel.from_pretrained(base, 'models/winning_ticket').eval()

# Test
prompt = '### Question: What is the repo rate?\n### Answer:'
inputs = tokenizer(prompt, return_tensors='pt')
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.7)
print(tokenizer.decode(out[0], skip_special_tokens=True))
"
```

---

## Launch the Streamlit app

Once models are in place:
```bash
/opt/anaconda3/bin/pip install streamlit plotly
/opt/anaconda3/bin/streamlit run finvaani/frontend/app.py
```

The app loads the winning ticket adapter on top of the cached base model — inference on MPS is fast (~2–3 seconds per answer).
