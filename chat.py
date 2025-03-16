import os, yaml, logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from tqdm import tqdm

# GPU kikapcsolása (Apple Silicon)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_NO_MPS"] = "1"

# Konfiguráció betöltése
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

logging.basicConfig(
    filename=f'{config["paths"]["logs_dir"]}/chat.log',
    level=logging.INFO if config["logging"]["enable"] else logging.ERROR
)

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
retriever = db.as_retriever(search_kwargs={"k": config["similarity_search"]["top_k"]})

# Prompt konfiguráció betöltése
if config["prompt"]["use_custom_prompt"]:
    prompt_template = config["prompt"]["template"]
    prompt = PromptTemplate.from_template(prompt_template)
else:
    prompt = None

# Top-k érték betöltése a konfigurációból
top_k = config["similarity_search"]["top_k"]

# Lánc inicializáció
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": top_k}),
    combine_docs_chain_kwargs={"prompt": prompt} if prompt else {},
    return_source_documents=True,
    response_if_no_docs_found="Nincs elegendő információ a válaszadáshoz."
)

chat_history = []

print("Chatbot készen áll. Kilépéshez írd: 'exit'")
while True:
    query = input("Kérdésed: ")
    if query.lower() in ('exit', 'quit'):
        break
    with tqdm(total=1, desc="Válasz generálása") as pbar:
        response = chain.invoke({"question": query, "chat_history": chat_history})
        pbar.update(1)

 # --- DEBUG kimenet hozzáadása ---
 #   print("\nDEBUG response:", response, "\n") 

    answer = response['answer']
    if config["metadata"]["show_source_in_response"]:
        sources = "\nForrások:\n" + "\n".join(
            set(os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in response.get('source_documents', []))
        )
        answer += sources
    print(answer)
    chat_history.append((query, answer))
    logging.info(f"Q: {query}\nA: {answer}\n")
