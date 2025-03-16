import os
import yaml
import logging
from tqdm import tqdm
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# GPU kikapcsolása (Apple Silicon)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_NO_MPS"] = "1"

# Konfiguráció betöltése
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

logging.basicConfig(
    filename=f'{config["paths"]["logs_dir"]}/process.log',
    level=logging.INFO if config["logging"]["enable"] else logging.ERROR
)

# Dokumentumok betöltése
def load_docs(directory):
    documents = []
    for filename in tqdm(os.listdir(directory), desc="Dokumentumok betöltése"):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif filename.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            continue

        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename  # <-- EZ FONTOS
        documents.extend(docs)

    return documents

# Fő folyamat
if __name__ == "__main__":
    docs = load_docs(config["paths"]["docs_dir"])
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    chunks = []
    for doc in tqdm(docs, desc="Dokumentumok darabolása"):
        chunks.extend(splitter.split_documents([doc]))

    embeddings = HuggingFaceEmbeddings(
        model_name=config["embedding"]["model"],
        model_kwargs={"device": "cpu"},
        encode_kwargs={"device": "cpu"}
    )

    # ChromaDB feltöltése embeddingekkel
    db = Chroma.from_documents(
        tqdm(chunks, desc="Embeddingek generálása"),
        embeddings,
        persist_directory=config["paths"]["vectorstore_dir"]
    )
    logging.info("A dokumentumfeldolgozás sikeresen befejeződött.")
