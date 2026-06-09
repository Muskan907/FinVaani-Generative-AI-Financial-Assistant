"""
04_about.py — About FinVaani: project description, technology, research, team.
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

st.set_page_config(page_title="About FinVaani", page_icon="ℹ️", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🪔 FinVaani")
    st.divider()
    st.markdown("### ℹ️ About")

st.markdown(
    '<h1 style="font-family:Poppins,sans-serif">ℹ️ About FinVaani</h1>',
    unsafe_allow_html=True,
)

# ── The Problem ───────────────────────────────────────────────────────────────
st.markdown("## The Problem")
st.markdown(
    '<div style="background:white;border-radius:12px;padding:20px;'
    'box-shadow:0 2px 10px rgba(0,0,0,0.07);border-left:4px solid #FF6B35">'
    '<p style="font-size:1.1rem;color:#374151">'
    'India has <b>1.4 billion people</b>. Most lack access to accurate financial '
    'guidance in their language. Complex financial regulations from RBI, SEBI, IRDAI, '
    'and NPCI are written in technical English — inaccessible to the majority of '
    'Indians who speak Hindi or other regional languages.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Our Solution ──────────────────────────────────────────────────────────────
st.markdown("## Our Solution")
st.markdown(
    '<div style="background:white;border-radius:12px;padding:20px;'
    'box-shadow:0 2px 10px rgba(0,0,0,0.07);border-left:4px solid #1B4332">'
    '<p style="font-size:1.1rem;color:#374151">'
    '<b>FinVaani</b> uses state-of-the-art NLP to make Indian financial knowledge '
    'accessible in <b>English and Hindi</b>. We fine-tune a multilingual language model '
    'on official government data, then compress it using the Lottery Ticket Hypothesis '
    'to make it fast and deployable on low-resource devices.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Technology ────────────────────────────────────────────────────────────────
st.markdown("## The Technology")

tech_items = [
    ("🗃️", "Dataset",     "1,706 Q&A pairs scraped from RBI, SEBI, IRDAI, NPCI, NCFE. "
                           "Bilingual: English + Hindi. Cleaned, deduplicated, stratified split."),
    ("🧠", "Base Model",  "mGPT (ai-forever/mGPT) — 117M parameter multilingual GPT-2 "
                           "trained on 61 languages including Hindi and English."),
    ("⚡", "Fine-tuning", "LoRA (Low-Rank Adaptation) via HuggingFace PEFT. Trains only "
                           "~1.2M parameters (<2% of total). 3 epochs on T4 GPU."),
    ("✂️", "Compression", "Lottery Ticket Hypothesis (Frankle & Carlin, 2019). Iterative "
                           "Magnitude Pruning removes 80% of LoRA weights while retaining "
                           "90%+ of fine-tuned quality."),
    ("🌐", "Languages",   "English and Hindi. Bilingual training data ensures the model "
                           "answers accurately in both languages."),
]

for icon, title, desc in tech_items:
    st.markdown(
        f'<div style="display:flex;gap:16px;background:white;border-radius:12px;'
        f'padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-bottom:10px">'
        f'<div style="font-size:2rem;min-width:48px;text-align:center">{icon}</div>'
        f'<div><b style="font-size:1rem">{title}</b>'
        f'<p style="color:#6B7280;margin:4px 0 0;font-size:0.9rem">{desc}</p></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Research ──────────────────────────────────────────────────────────────────
st.markdown("## The Research")
st.markdown(
    '<div style="background:#FFFBEB;border-radius:12px;padding:20px;'
    'border:1px solid #FDE68A">'
    '<b>The Lottery Ticket Hypothesis</b> (Frankle & Carlin, MIT, 2019)<br>'
    '<i>ICLR 2019 Best Paper Award</i><br><br>'
    'The hypothesis states: <i>"A randomly-initialized, dense neural network contains '
    'a subnetwork (the winning ticket) that, when trained in isolation, can match the '
    'test accuracy of the original network after training for at most the same number '
    'of iterations."</i><br><br>'
    'In practice: we iteratively prune the smallest weights, reset remaining weights '
    'to their original values, and retrain. The surviving subnetwork is the '
    '"winning ticket" — smaller, faster, and nearly as accurate.'
    '</div>',
    unsafe_allow_html=True,
)

# ── Data Sources ──────────────────────────────────────────────────────────────
st.markdown("## Data Sources")
sources = [
    ("🏦", "RBI",  "Reserve Bank of India",
     "Banking, NBFC, G-Secs, Forex, Digital Payments, Financial Inclusion"),
    ("📈", "SEBI", "Securities and Exchange Board of India",
     "Mutual Funds, IPO, Demat, Portfolio Management, Bonds"),
    ("🛡️", "IRDAI","Insurance Regulatory and Development Authority",
     "Life Insurance, Health Insurance, Motor Insurance, Claims"),
    ("💳", "NPCI", "National Payments Corporation of India",
     "UPI, RuPay, NACH, BBPS, FASTag, IMPS"),
    ("📚", "NCFE", "National Centre for Financial Education",
     "Financial Literacy, Budgeting, Savings, Credit, Tax"),
]

cols = st.columns(len(sources))
for col, (icon, short, full, topics) in zip(cols, sources):
    with col:
        st.markdown(
            f'<div style="background:white;border-radius:12px;padding:14px;'
            f'box-shadow:0 2px 8px rgba(0,0,0,0.06);text-align:center">'
            f'<div style="font-size:2rem">{icon}</div>'
            f'<div style="font-weight:700;font-size:1.1rem;color:#FF6B35">{short}</div>'
            f'<div style="font-size:0.75rem;color:#6B7280;margin:4px 0">{full}</div>'
            f'<div style="font-size:0.72rem;color:#9CA3AF">{topics}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.caption("All data sourced from official Indian government websites.")

# ── Team ──────────────────────────────────────────────────────────────────────
st.markdown("## Team")
st.markdown(
    '<div style="background:white;border-radius:12px;padding:20px;'
    'box-shadow:0 2px 10px rgba(0,0,0,0.07)">'
    '<div style="display:flex;gap:20px;flex-wrap:wrap">'
    '<div style="text-align:center;min-width:120px">'
    '<div style="font-size:2.5rem">👤</div>'
    '<div style="font-weight:600">Team Member 1</div>'
    '<div style="font-size:0.8rem;color:#6B7280">ML Engineer</div></div>'
    '<div style="text-align:center;min-width:120px">'
    '<div style="font-size:2.5rem">👤</div>'
    '<div style="font-weight:600">Team Member 2</div>'
    '<div style="font-size:0.8rem;color:#6B7280">NLP Researcher</div></div>'
    '<div style="text-align:center;min-width:120px">'
    '<div style="font-size:2.5rem">👤</div>'
    '<div style="font-weight:600">Team Member 3</div>'
    '<div style="font-size:0.8rem;color:#6B7280">Full-Stack Dev</div></div>'
    '</div></div>',
    unsafe_allow_html=True,
)

st.markdown("---")
st.markdown(
    '<div style="text-align:center;padding:10px">'
    '<b>BML Munjal University</b> &nbsp;|&nbsp; '
    'B.Tech — Natural Language Processing &nbsp;|&nbsp; 2024-25'
    '</div>',
    unsafe_allow_html=True,
)
