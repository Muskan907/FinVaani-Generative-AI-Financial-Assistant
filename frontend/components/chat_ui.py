"""chat_ui.py — Chat bubble rendering components."""
import streamlit as st


def render_user_bubble(text: str):
    st.markdown(
        f'<div style="display:flex;justify-content:flex-end;margin:6px 0">'
        f'<div style="background:#FF6B35;color:white;border-radius:18px 18px 4px 18px;'
        f'padding:10px 16px;max-width:75%;font-size:0.95rem">{text}</div></div>',
        unsafe_allow_html=True,
    )


def render_bot_bubble(text: str, model_tag: str = "FinVaani",
                      source: str = "RBI/SEBI/IRDAI"):
    st.markdown(
        f'<div style="display:flex;justify-content:flex-start;margin:6px 0">'
        f'<div style="background:white;border:1px solid #E5E7EB;border-radius:18px 18px 18px 4px;'
        f'padding:10px 16px;max-width:80%;box-shadow:0 2px 8px rgba(0,0,0,0.06)">'
        f'<div style="font-size:0.95rem;color:#1A1A1A">{text}</div>'
        f'<div style="margin-top:6px;font-size:0.72rem;color:#9CA3AF">'
        f'🤖 {model_tag} &nbsp;|&nbsp; 📚 {source}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
