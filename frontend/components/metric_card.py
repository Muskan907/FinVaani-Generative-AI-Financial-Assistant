"""metric_card.py — Metric display component."""
import streamlit as st


def render_metric_card(label: str, value: str, delta: str = "", color: str = "#FF6B35"):
    st.markdown(
        f'<div style="background:white;border-radius:12px;padding:16px;'
        f'box-shadow:0 2px 10px rgba(0,0,0,0.07);text-align:center">'
        f'<div style="font-size:2rem;font-weight:700;color:{color}">{value}</div>'
        f'<div style="font-size:0.85rem;color:#6B7280;margin-top:4px">{label}</div>'
        + (f'<div style="font-size:0.75rem;color:#9CA3AF;margin-top:2px">{delta}</div>'
           if delta else "")
        + '</div>',
        unsafe_allow_html=True,
    )
