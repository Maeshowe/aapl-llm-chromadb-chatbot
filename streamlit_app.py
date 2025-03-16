import streamlit as st
import os, yaml
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# GPU kikapcsolása (Apple Silicon)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_NO_MPS"] = "1"

# Konfiguráció betöltése
@st.cache_resource
def load_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

config = load_config()

@st.cache_resource
def load_chain():
    model_conf = config["llm"]["models"][config["llm"]["active_model"]]

    llm = LlamaCpp(
        model_path=model_conf["model_path"],
        temperature=model_conf["temperature"],
        max_tokens=model_conf["max_tokens"],
        n_ctx=model_conf["context_size"],
        n_batch=model_conf["n_batch"],
        n_threads=model_conf["n_threads"],
        verbose=config["llm"].get("verbose", False),
        n_gpu_layers=0
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=config["embedding"]["model"],
        model_kwargs={"device": "cpu"},
        encode_kwargs={"device": "cpu"}
    )

    db = Chroma(persist_directory=config["paths"]["vectorstore_dir"], embedding_function=embeddings)

    prompt = PromptTemplate.from_template(config["prompt"]["template"]) if config["prompt"]["use_custom_prompt"] else None

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": config["similarity_search"]["top_k"]}),
        combine_docs_chain_kwargs={"prompt": prompt} if prompt else {},
        return_source_documents=True,
        response_if_no_docs_found="Nincs elegendő információ a válaszadáshoz."
    )

chain = load_chain()

# Streamlit UI
st.title("📚 Lokális Dokumentum-alapú Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Kérdésed:", placeholder="Írd ide a kérdésed...")

if st.button("Küldés") and user_input:
    with st.spinner("Válasz generálása..."):
        response = chain.invoke({"question": user_input, "chat_history": st.session_state.chat_history})
        answer = response['answer']
        st.session_state.chat_history.append((user_input, answer))

        st.markdown("### 💬 Válasz")
        st.write(answer)

        if config["metadata"]["show_source_in_response"]:
            sources = set(os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in response.get('source_documents', []))
            st.markdown("---")
            st.markdown("**📖 Forrásdokumentumok:**")
            for source in sources:
                st.markdown(f"- `{source}`")

st.markdown("---")
st.markdown("### 🔄 Korábbi beszélgetés")

for question, answer in reversed(st.session_state.chat_history[-5:]):
    st.markdown(f"**Te:** {question}")
    st.markdown(f"**Bot:** {answer}")
    st.markdown("---")

