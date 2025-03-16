import streamlit as st
import os

LOG_DIR = "logs"

log_files = {
    "CLI Chat Log": "chat.log",
    "Streamlit Chat Log": "streamlit_chat.log",
    "Document Processing Log": "process.log"
}

selected_log = st.sidebar.selectbox("Válassz logfájlt:", list(log_files.keys()))
log_path = os.path.join(LOG_DIR, log_files[selected_log])

st.header(f"📄 {selected_log} tartalma")

if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if lines:
        st.write("### Logfájl első 20 sora:")
        st.code("".join(lines[:20]))

        st.write("### Logfájl utolsó 20 sora:")
        st.code("".join(lines[-20:]))
    else:
        st.warning("A logfájl üres!")
else:
    st.error("A kiválasztott logfájl nem található!")