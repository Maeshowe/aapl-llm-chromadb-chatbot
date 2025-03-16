# 📚 Dokumentum-alapú Chatbot (LlamaCpp + ChromaDB + Streamlit)

Ez a projekt egy helyben futtatható, dokumentum-alapú chatbotot valósít meg, amely LangChain keretrendszert, ChromaDB vektoros keresést, LlamaCpp LLM modellt, és Streamlit webes felületet használ.

## 🛠️ Technológiai stack

- **LangChain:** Az LLM és RAG (Retrieval-Augmented Generation) backend kezelésére.
- **ChromaDB:** Dokumentumok hatékony keresése embeddingek alapján.
- **LlamaCpp:** LLM futtatása CPU-n, kvantált GGUF formátumú LLM-modellekkel.
- **Streamlit:** Felhasználóbarát webes interfész.

---

## 📂 Projektstruktúra

```
LLM-MSFT
│
├── chat.py                  # CLI-alapú chatbot
├── process.py               # Dokumentumok feldolgozása
├── streamlit_app.py         # Streamlit webes felület
├── config.yaml              # Konfigurációs fájl
├── requirements.txt         # Python-függőségek
├── docs                     # Ide kerülnek feldolgozásra váró dokumentumok
├── models                   # LLM modellek helye (.gguf formátum)
├── logs                     # Logfájlok
└── vectorstore              # ChromaDB adatbázis
```

## 📦 Telepítés

### 1. Klónozd a repositoryt

```bash
git clone <repo_url>
cd LLM-MSFT
```

### 2. Virtuális környezet létrehozása

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/MacOS
.venv\Scripts\activate      # Windows
```

### 2. Függőségek telepítése

```bash
pip install -r requirements.txt
```

### 3. Konfiguráció

Módosítsd a `config.yaml` fájlt, hogy a saját modelljeid és beállításaid megfelelőek legyenek.

## 🚀 Használat

### 🔄 Dokumentumfeldolgozás (Embedding generálás)

```bash
python process.py
```

### 🤖 CLI-alapú Chatbot indítása

```bash
python chat.py
```

- Kilépés: írd be az `exit` vagy `quit` parancsot.

### 🌐 Streamlit webes felület futtatása

```bash
streamlit run streamlit_app.py --server.fileWatcherType none
```

Ezután nyisd meg a böngésződben a megadott helyi URL-t (általában: `http://localhost:8501`).

---

## 📂 Logok

A rendszer részletesen logolja a műveleteket a `logs` mappában.

- **process.log**: dokumentumfeldolgozási részletek.
- **chat.log**: CLI chatbot interakciók és válaszidők.
- **streamlit_chat.log**: webes felületen történt interakciók és részletek.

---

## 🚧 Hibaelhárítás

- Ellenőrizd a logokat a `logs` mappában, ha problémát tapasztalsz.
- Ügyelj arra, hogy a környezeted Python 3.12 alatt fusson a maximális kompatibilitás érdekében.

## 📌 Következő fejlesztési lépések

- 📊 Részletes dashboard készítése a logfájlok elemzéséhez.
- 🔒 Felhasználó autentikáció hozzáadása a webes felülethez.
- ⚡ Új modellek integrációja és teljesítmény-összehasonlítások.

---

## 📞 Kapcsolat
Ha kérdésed van vagy támogatásra van szükséged, fordulj hozzám bizalommal!

