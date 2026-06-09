"""model_card.py — Model info display component."""
import streamlit as st

BADGE_STYLES = {
    "Baseline":   ("grey",   "#9E9E9E", "white"),
    "Few-shot":   ("blue",   "#2196F3", "white"),
    "Fine-tuned": ("green",  "#4CAF50", "white"),
    "Compressed": ("gold",   "#FFD700", "#1A1A1A"),
}


def render_model_card(model_name: str, badge: str, params: str,
                      sparsity: str, bleu: str, answer: str = ""):
    _, bg, fg = BADGE_STYLES.get(badge, ("grey", "#9E9E9E", "white"))
    st.markdown(
        f'<div style="background:white;border-radius:12px;padding:16px;'
        f'box-shadow:0 2px 10px rgba(0,0,0,0.07);height:100%">'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">'
        f'<span style="font-weight:700;font-size:1rem">{model_name}</span>'
        f'<span style="background:{bg};color:{fg};padding:2px 10px;border-radius:12px;'
        f'font-size:0.72rem;font-weight:600">{badge}</span></div>'
        f'<div style="font-size:0.8rem;color:#6B7280;margin-bottom:8px">'
        f'Params: <b>{params}</b> &nbsp;|&nbsp; Sparsity: <b>{sparsity}</b><br>'
        f'BLEU: <b>{bleu}</b></div>'
        + (f'<div style="background:#F9FAFB;border-radius:8px;padding:10px;'
           f'font-size:0.88rem;color:#374151;max-height:180px;overflow-y:auto">'
           f'{answer}</div>' if answer else "")
        + '</div>',
        unsafe_allow_html=True,
    )
