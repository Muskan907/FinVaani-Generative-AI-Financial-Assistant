"""
app.py — FinVaani main Streamlit application entry point.
Run with: streamlit run frontend/app.py
"""

import os
import sys
import streamlit as st
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinVaani — Indian Finance AI",
    page_icon="🪔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:10px 0">'
        '<span style="font-size:2.5rem">🪔</span><br>'
        '<span style="font-family:Poppins,sans-serif;font-size:1.6rem;'
        'font-weight:700;color:#FF6B35">FinVaani</span><br>'
        '<span style="font-size:0.8rem;color:#6B7280">India\'s Financial Intelligence, Compressed.</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### Navigation")
    st.page_link("app.py",              label="🏠 Home",            icon=None)
    st.page_link("pages/01_chat.py",    label="💬 Chat",            icon=None)
    st.page_link("pages/02_compare.py", label="🔬 Model Lab",       icon=None)
    st.page_link("pages/03_metrics.py", label="📊 Metrics",         icon=None)
    st.page_link("pages/04_about.py",   label="ℹ️ About",           icon=None)

    st.divider()
    st.markdown(
        '<div style="background:linear-gradient(135deg,#FF6B35,#FFD700);'
        'color:white;border-radius:8px;padding:6px 12px;font-size:0.75rem;'
        'font-weight:600;text-align:center">⚡ Powered by Lottery Ticket Hypothesis</div>',
        unsafe_allow_html=True,
    )
    st.caption("Built on mGPT | Fine-tuned on Indian Govt Data")

# ── Home page ─────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="font-family:Poppins,sans-serif;font-size:3rem;'
    'background:linear-gradient(135deg,#FF6B35,#1B4332);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
    'margin-bottom:0">🪔 FinVaani</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="font-size:1.2rem;color:#6B7280;margin-top:4px">'
    'India\'s Financial Intelligence, Compressed. &nbsp;|&nbsp; '
    'English + हिंदी</p>',
    unsafe_allow_html=True,
)

st.divider()

# Feature cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        '<div style="background:white;border-radius:12px;padding:20px;'
        'box-shadow:0 2px 10px rgba(0,0,0,0.07);text-align:center">'
        '<div style="font-size:2rem">💬</div>'
        '<div style="font-weight:600;margin:8px 0">Bilingual Chat</div>'
        '<div style="font-size:0.85rem;color:#6B7280">Ask in English or Hindi</div>'
        '</div>', unsafe_allow_html=True)
with c2:
    st.markdown(
        '<div style="background:white;border-radius:12px;padding:20px;'
        'box-shadow:0 2px 10px rgba(0,0,0,0.07);text-align:center">'
        '<div style="font-size:2rem">✂️</div>'
        '<div style="font-weight:600;margin:8px 0">80% Compressed</div>'
        '<div style="font-size:0.85rem;color:#6B7280">Lottery Ticket Hypothesis</div>'
        '</div>', unsafe_allow_html=True)
with c3:
    st.markdown(
        '<div style="background:white;border-radius:12px;padding:20px;'
        'box-shadow:0 2px 10px rgba(0,0,0,0.07);text-align:center">'
        '<div style="font-size:2rem">🏛️</div>'
        '<div style="font-weight:600;margin:8px 0">Govt Sources</div>'
        '<div style="font-size:0.85rem;color:#6B7280">RBI · SEBI · IRDAI · NPCI</div>'
        '</div>', unsafe_allow_html=True)
with c4:
    st.markdown(
        '<div style="background:white;border-radius:12px;padding:20px;'
        'box-shadow:0 2px 10px rgba(0,0,0,0.07);text-align:center">'
        '<div style="font-size:2rem">🎓</div>'
        '<div style="font-weight:600;margin:8px 0">Research Grade</div>'
        '<div style="font-size:0.85rem;color:#6B7280">B.Tech NLP Project</div>'
        '</div>', unsafe_allow_html=True)

st.divider()
st.markdown("### Real Evaluation Results")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Training Pairs", "1,706", "623 EN + 1,083 HI")
s2.metric("LoRA BLEU", "0.0197", "+229% vs raw mGPT")
s3.metric("LoRA Perplexity", "5.7", "vs 6.9 baseline")
s4.metric("LTH Sparsity", "67.2%", "5 pruning rounds")

st.divider()
st.info("👈 Use the sidebar to navigate. Start with **💬 Chat** to ask financial questions.")
