import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re

st.set_page_config(page_title="📊 Log Dashboard", layout="wide")
st.title("📊 Chatbot Log Dashboard")

LOG_DIR = "logs"

log_files = {
    "CLI Chat Log": "chat.log",
    "Streamlit Chat Log": "streamlit_chat.log",
    "Document Processing Log": "process.log"
}

selected_log = st.sidebar.selectbox("Válassz logfájlt elemzésre:", list(log_files.keys()))
log_path = os.path.join(LOG_DIR, log_files[selected_log])

@st.cache_data
def load_log(log_path):
    pattern = r'^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) \[(?P<level>\w+)\] (?P<message>.+)$'
    data = []
    with open(log_path, "r", encoding="utf-8") as file:
        for line in file:
            match = re.match(pattern, line.strip())
            if match:
                data.append(match.groupdict())
    df = pd.DataFrame(data)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
    return df

log_path = os.path.join(LOG_DIR, log_files[selected_log])

if os.path.exists(log_path):
    df = load_log(log_path)

    if df.empty:
        st.warning("A logfájl üres vagy nem megfelelő formátumú.")
    else:
        st.subheader(f"📄 {selected_log} tartalma (utolsó 100 sor)")
        st.dataframe(df.tail(100), height=250)

        if "Chat Log" in selected_log:
            questions = df[df["message"].str.contains("Kérdés:", na=False)]
            responses = df[df["message"].str.contains("Válasz:", na=False)]
            inference_times = df[df["message"].str.contains("Inferencia idő:", na=False)].copy()
            inference_times['time'] = inference_times['message'].str.extract(r"Inferencia idő: ([\d.]+)s").astype(float)

            st.subheader("🚀 Chat statisztikák")
            col1, col2, col3 = st.columns(3)
            col1.metric("Összes kérdés", len(questions))
            col2.metric("Összes válasz", len(responses))
            col3.metric("Átlag válaszidő (s)", round(inference_times['time'].mean(), 2))

            if not inference_times.empty:
                st.subheader("📈 Válaszidők eloszlása")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(inference_times['time'], bins=15, kde=True, ax=ax)
                ax.set_xlabel("Válaszidő (s)")
                ax.set_ylabel("Gyakoriság")
                st.pyplot(fig)

                st.subheader("📉 Válaszidő időbeli alakulása")
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                sns.lineplot(x=inference_times['timestamp'], y=inference_times['time'], marker='o', ax=ax2)
                ax2.set_xlabel("Időpont")
                ax2.set_ylabel("Válaszidő (s)")
                st.pyplot(fig2)
            else:
                st.warning("Nincs inferencia idő adat a logban.")

        elif selected_log == "Document Processing Log":
            processing_df = df[df.message.str.contains("Betöltve:", na=False)].copy()
            processing_df['filename'] = processing_df['message'].str.extract(r'Betöltve: ([^\|]+)').iloc[:,0]
            processing_df['processing_time'] = processing_df['message'].str.extract(r'Idő: ([\d.]+) s').astype(float)

            st.subheader("📚 Dokumentumfeldolgozási statisztikák")
            st.metric("Feldolgozott dokumentumok száma", len(processing_df))
            st.metric("Átlagos feldolgozási idő (s)", round(processing_df.processing_time.mean(), 2))

            fig3, ax3 = plt.subplots(figsize=(10, max(4, len(processing_df)*0.5)))
            sns.barplot(y='filename', x='processing_time', data=processing_df, ax=ax3)
            ax3.set_xlabel("Feldolgozási idő (s)")
            ax3.set_ylabel("Dokumentum")
            st.pyplot(fig3)

        else:
            st.info("Ehhez a logfájlhoz nincs részletes elemzés.")
else:
    st.error("A kiválasztott logfájl nem található.")

