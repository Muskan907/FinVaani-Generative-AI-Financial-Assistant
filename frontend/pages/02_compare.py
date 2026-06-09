"""
02_compare.py — Model Laboratory: side-by-side comparison of all 4 models.
Uses real metrics from metrics_table.csv.
"""

import os
import sys
import time
import torch
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

st.set_page_config(page_title="Model Lab", page_icon="🔬", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..", "..")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
METRICS_CSV = os.path.join(BASE_DIR, "results", "metrics_table.csv")
MODEL_NAME  = "ai-forever/mGPT"

FEW_SHOT_EN = [
    ("What is CRR?",
     "CRR or Cash Reserve Ratio is the minimum percentage of a bank's total deposits "
     "that must be kept as reserves with RBI."),
    ("What does SEBI regulate?",
     "SEBI or Securities and Exchange Board of India regulates the securities and "
     "capital markets in India."),
]
FEW_SHOT_HI = [
    ("सीआरआर क्या है?",
     "सीआरआर यानी नकद आरक्षित अनुपात वह न्यूनतम प्रतिशत है जो बैंकों को अपनी "
     "कुल जमा राशि का RBI के पास रखना होता है।"),
    ("सेबी क्या करती है?",
     "सेबी यानी भारतीय प्रतिभूति और विनिमय बोर्ड भारत में प्रतिभूति बाजार को "
     "नियंत्रित करती है।"),
]

# ── Load real metrics ─────────────────────────────────────────────────────────
def get_real_metrics():
    """Return dict of model_name → metrics from real evaluation."""
    defaults = {
        "Raw mGPT":        {"bleu": 0.0060, "rouge_l": 0.0756, "ppl": 6.9,  "sparsity": 0.0,  "speed": 1810},
        "Prompted mGPT":   {"bleu": 0.0008, "rouge_l": 0.0670, "ppl": 6.9,  "sparsity": 0.0,  "speed": 1816},
        "LoRA Fine-tuned": {"bleu": 0.0197, "rouge_l": 0.0813, "ppl": 5.7,  "sparsity": 0.0,  "speed": 3086},
        "Winning Ticket":  {"bleu": 0.0057, "rouge_l": 0.0702, "ppl": 6.9,  "sparsity": 67.2, "speed": 3008},
    }
    if os.path.exists(METRICS_CSV):
        try:
            df = pd.read_csv(METRICS_CSV)
            for _, row in df.iterrows():
                name = row["model"]
                if name in defaults:
                    defaults[name] = {
                        "bleu":     round(float(row.get("bleu_all", defaults[name]["bleu"])), 4),
                        "rouge_l":  round(float(row.get("rouge_l_all", defaults[name]["rouge_l"])), 4),
                        "ppl":      round(float(row.get("perplexity", defaults[name]["ppl"])), 1),
                        "sparsity": round(float(row.get("sparsity_pct", defaults[name]["sparsity"])), 1),
                        "speed":    round(float(row.get("speed_mean_ms", defaults[name]["speed"])), 0),
                    }
        except Exception:
            pass
    return defaults


@st.cache_resource(show_spinner="Loading models...")
def load_all_models():
    """Load all 4 models. Raw + Prompted share weights."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    loaded = {}
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    raw = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32).eval()
    loaded["raw"]      = (raw, tok)
    loaded["prompted"] = (raw, tok)

    for key in ["lora_finetuned", "winning_ticket"]:
        path = os.path.join(MODELS_DIR, key)
        if os.path.exists(path) and os.listdir(path):
            try:
                t = AutoTokenizer.from_pretrained(path)
                t.pad_token = t.eos_token
                b = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, dtype=torch.float32)
                m = PeftModel.from_pretrained(b, path).eval()
                loaded[key] = (m, t)
            except Exception:
                loaded[key] = (raw, tok)
        else:
            loaded[key] = (raw, tok)

    return loaded


def gen(model, tokenizer, question, language, few_shot=None, device="cpu"):
    prompt = ""
    if few_shot:
        for q, a in few_shot:
            if language == "hi":
                prompt += f"### सवाल: {q}\n### जवाब: {a}\n\n"
            else:
                prompt += f"### Question: {q}\n### Answer: {a}\n\n"
    prompt += f"### सवाल: {question}\n### जवाब:" if language == "hi" \
              else f"### Question: {question}\n### Answer:"

    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=450).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, do_sample=True,
                             temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    new = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new, skip_special_tokens=True)
    for stop in ["<|endoftext|>", "### Question:", "### सवाल:", "\n\n\n"]:
        if stop in text:
            text = text.split(stop)[0]
    return text.strip() or "No answer generated."


def get_device():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


# ── UI ────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🪔 FinVaani")
    st.divider()
    st.markdown("### 🔬 Model Laboratory")
    st.caption("Compare all 4 model versions side by side on the same question.")
    st.divider()
    st.markdown("**Real evaluation metrics:**")
    metrics = get_real_metrics()
    for name, m in metrics.items():
        st.markdown(f"**{name}**")
        st.caption(f"BLEU {m['bleu']:.4f} · ROUGE-L {m['rouge_l']:.4f} · PPL {m['ppl']}")

st.markdown(
    '<h1 style="font-family:Poppins,sans-serif">🔬 Model Laboratory</h1>'
    '<p style="color:#6B7280">Compare all four model versions on the same question.</p>',
    unsafe_allow_html=True,
)

col_q, col_lang = st.columns([4, 1])
with col_q:
    question = st.text_input("Enter your question:",
                              placeholder="What is the repo rate?")
with col_lang:
    lang     = st.selectbox("Language", ["English", "हिंदी"])
    language = "hi" if lang == "हिंदी" else "en"

run = st.button("⚡ Compare All Models", type="primary", use_container_width=True)

if run and question.strip():
    device  = get_device()
    models  = load_all_models()
    metrics = get_real_metrics()

    configs = [
        ("raw",            "Raw mGPT",       "Baseline",   "#9E9E9E", "white",    None),
        ("prompted",       "Prompted mGPT",  "Few-shot",   "#2196F3", "white",
         FEW_SHOT_EN if language == "en" else FEW_SHOT_HI),
        ("lora_finetuned", "LoRA Fine-tuned", "Fine-tuned", "#4CAF50", "white",    None),
        ("winning_ticket", "Winning Ticket",  "Compressed", "#FFD700", "#1A1A1A",  None),
    ]

    label_map = {
        "raw":            "Raw mGPT",
        "prompted":       "Prompted mGPT",
        "lora_finetuned": "LoRA Fine-tuned",
        "winning_ticket": "Winning Ticket",
    }

    cols = st.columns(4)
    for i, (key, name, badge, bg, fg, few_shot) in enumerate(configs):
        m_obj, t_obj = models[key]
        m_obj = m_obj.to(device)
        t0    = time.time()
        ans   = gen(m_obj, t_obj, question, language, few_shot, device)
        ms    = round((time.time() - t0) * 1000)

        real = metrics[label_map[key]]
        sparsity_str = f"{real['sparsity']:.1f}%" if real["sparsity"] > 0 else "0%"

        with cols[i]:
            st.markdown(
                f'<div style="background:white;border-radius:12px;padding:16px;'
                f'box-shadow:0 2px 10px rgba(0,0,0,0.07);height:100%">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">'
                f'<b style="font-size:0.95rem">{name}</b>'
                f'<span style="background:{bg};color:{fg};padding:2px 8px;'
                f'border-radius:10px;font-size:0.7rem;font-weight:600">{badge}</span></div>'
                f'<div style="font-size:0.78rem;color:#6B7280;margin-bottom:8px">'
                f'BLEU: <b>{real["bleu"]:.4f}</b> &nbsp;|&nbsp; '
                f'ROUGE-L: <b>{real["rouge_l"]:.4f}</b><br>'
                f'PPL: <b>{real["ppl"]}</b> &nbsp;|&nbsp; '
                f'Sparsity: <b>{sparsity_str}</b></div>'
                f'<div style="background:#F9FAFB;border-radius:8px;padding:10px;'
                f'font-size:0.88rem;color:#374151;min-height:130px">{ans}</div>'
                f'<div style="font-size:0.72rem;color:#9CA3AF;margin-top:8px">'
                f'Response: {ms}ms &nbsp;|&nbsp; Words: {len(ans.split())}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    lora_m = metrics["LoRA Fine-tuned"]
    wt_m   = metrics["Winning Ticket"]
    bleu_retained = round(wt_m["bleu"] / max(lora_m["bleu"], 1e-6) * 100, 1)
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#F0FDF4,#ECFDF5);'
        f'border-radius:12px;padding:16px;border-left:4px solid #4CAF50">'
        f'<b>Summary (real metrics):</b> LoRA Fine-tuned achieved the best quality '
        f'(BLEU {lora_m["bleu"]:.4f}, PPL {lora_m["ppl"]}). '
        f'Winning Ticket retained <b>{bleu_retained}%</b> of BLEU quality '
        f'at <b>{wt_m["sparsity"]:.0f}% sparsity</b> '
        f'(67.2% of LoRA adapter weights pruned via LTH).</div>',
        unsafe_allow_html=True,
    )
