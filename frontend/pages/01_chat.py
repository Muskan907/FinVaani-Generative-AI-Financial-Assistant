"""
01_chat.py — FinVaani main chat page.
Primary model: LoRA Fine-tuned mGPT.

Memory strategy for 1.4B model on 17GB Mac:
  - Model loaded ONCE via st.cache_resource (never reloaded)
  - Generation capped at 60 tokens (fast enough, avoids OOM)
  - 30-second timeout kills hung generation
  - CPU fallback if MPS causes issues
"""

import os
import sys
import time
import threading
import torch
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

st.set_page_config(page_title="FinVaani Chat", page_icon="💬", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading FinVaani model (one-time, ~15s)...")
def load_chat_model():
    """
    Load LoRA fine-tuned model ONCE and cache it for the entire session.
    Uses CPU to avoid MPS memory fragmentation issues on 17GB Mac.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    base_dir   = os.path.join(os.path.dirname(__file__), "..", "..")
    models_dir = os.path.join(base_dir, "models")
    MODEL_NAME = "ai-forever/mGPT"

    # Use CPU — avoids MPS memory fragmentation that causes hangs
    # MPS is faster per-token but causes OOM on 17GB when model is 5.6GB
    device = "cpu"

    for candidate in ["lora_finetuned", "winning_ticket"]:
        path = os.path.join(models_dir, candidate)
        if os.path.exists(path) and os.listdir(path):
            try:
                tok = AutoTokenizer.from_pretrained(path)
                tok.pad_token = tok.eos_token
                base = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, dtype=torch.float32)
                model = PeftModel.from_pretrained(base, path).eval()
                return model, tok, candidate, device
            except Exception:
                continue

    # Fallback: raw mGPT
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32).eval()
    return model, tok, "raw_mgpt", device


def generate_with_timeout(model, tokenizer, question: str,
                           language: str, device: str,
                           timeout_sec: int = 45) -> str:
    """
    Generate an answer with a hard timeout.
    Returns a fallback message if generation exceeds timeout_sec.
    """
    if language == "hi":
        prompt = f"### सवाल: {question}\n### जवाब:"
    else:
        prompt = f"### Question: {question}\n### Answer:"

    result = {"text": None, "error": None}

    def _generate():
        try:
            inputs = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=256,
            ).to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=60,          # Short answers — fast & avoids OOM
                    do_sample=False,            # Greedy — faster than sampling
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            for stop in ["<|endoftext|>", "### Question:", "### सवाल:", "\n\n\n"]:
                if stop in text:
                    text = text.split(stop)[0]
            result["text"] = text.strip()
        except Exception as e:
            result["error"] = str(e)

    t = threading.Thread(target=_generate, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        return ("⏱️ Response timed out (model is processing a large request). "
                "Please try a shorter question.")
    if result["error"]:
        return f"Error generating answer: {result['error']}"
    return result["text"] or "Could not generate a relevant answer. Please rephrase."


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "language" not in st.session_state:
    st.session_state.language = "en"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🪔 FinVaani")
    st.divider()
    lang_choice = st.radio("Language / भाषा", ["English", "हिंदी"], horizontal=True)
    st.session_state.language = "hi" if lang_choice == "हिंदी" else "en"
    st.divider()

    st.markdown("### About this answer")
    st.markdown("**Model:** LoRA Fine-tuned mGPT")
    st.markdown("**BLEU:** 0.0197 &nbsp;|&nbsp; **ROUGE-L:** 0.0813")
    st.markdown("**Perplexity:** 5.7 (best among all models)")
    st.markdown("**Training:** 3 epochs · 1,194 samples · T4 GPU")
    st.markdown("**Data:** RBI · SEBI · IRDAI · NPCI · NCFE")
    st.divider()
    st.caption("⚡ Running on CPU (avoids MPS memory fragmentation on 17GB Mac). "
               "Response time: ~20-40s per answer.")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;padding:20px 0 10px">'
    '<h1 style="font-family:Poppins,sans-serif;font-size:2.8rem;'
    'background:linear-gradient(135deg,#FF6B35,#1B4332);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent">🪔 FinVaani</h1>'
    '<p style="font-size:1.1rem;color:#6B7280">Your AI-powered guide to Indian Finance</p>'
    '<p style="font-size:0.8rem;color:#9CA3AF">'
    'LoRA fine-tuned on RBI · SEBI · IRDAI · NPCI · NCFE &nbsp;|&nbsp; '
    'English + हिंदी</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Quick chips ───────────────────────────────────────────────────────────────
lang = st.session_state.language
if lang == "en":
    chips = ["What is repo rate?", "How does SEBI work?",
             "What is UPI?", "Explain mutual funds"]
else:
    chips = ["रेपो रेट क्या है?", "सेबी क्या करती है?",
             "UPI क्या है?", "म्यूचुअल फंड क्या है?"]

st.markdown("**Quick questions:**")
cols = st.columns(len(chips))
chip_clicked = None
for i, (col, chip) in enumerate(zip(cols, chips)):
    if col.button(chip, key=f"chip_{i}", use_container_width=True):
        chip_clicked = chip

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div style="display:flex;justify-content:flex-end;margin:6px 0">'
            f'<div style="background:#FF6B35;color:white;border-radius:18px 18px 4px 18px;'
            f'padding:10px 16px;max-width:75%;font-size:0.95rem">{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="display:flex;justify-content:flex-start;margin:6px 0">'
            f'<div style="background:white;border:1px solid #E5E7EB;'
            f'border-radius:18px 18px 18px 4px;padding:10px 16px;max-width:80%;'
            f'box-shadow:0 2px 8px rgba(0,0,0,0.06)">'
            f'<div style="font-size:0.95rem;color:#1A1A1A">{msg["content"]}</div>'
            f'<div style="font-size:0.72rem;color:#9CA3AF;margin-top:6px">'
            f'🤖 FinVaani LoRA &nbsp;|&nbsp; 📚 RBI/SEBI/IRDAI/NPCI/NCFE</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

# ── Input ─────────────────────────────────────────────────────────────────────
placeholder = "Ask anything about Indian finance..." if lang == "en" \
              else "भारतीय वित्त के बारे में कुछ भी पूछें..."

with st.form("chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input("", placeholder=placeholder,
                                   label_visibility="collapsed")
    with col_btn:
        submitted = st.form_submit_button("➤", use_container_width=True)

question = chip_clicked or (user_input if submitted and user_input.strip() else None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Generating answer... (20-40s on CPU — 1.4B parameter model)"):
        try:
            model, tokenizer, model_name, device = load_chat_model()
            t0     = time.time()
            answer = generate_with_timeout(
                model, tokenizer, question, lang, device, timeout_sec=45)
            elapsed = round((time.time() - t0) * 1000)
        except Exception as e:
            answer  = f"Error: {e}"
            elapsed = 0
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
