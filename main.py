from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import ollama
from pydantic import BaseModel
from agent import ask_agent
from rag.loader import load_and_split
from rag.qa_chain import create_qa_chain
from rag.vector_store import create_vectorstore, load_vectorstore  # Usiamo il nostro vector_store basato su HuggingFace
import os
from monitor import Watcher
import time

os.makedirs("logs", exist_ok=True)

# Variabili globali per il vector store e la chain di QA
vector_store = None
qa_chain = None

# Modello di input per le domande (non obbligatorio, sebbene utile)
class QuestionRequest(BaseModel):
    question: str
    mode: str  # "rag" per la modalità RAG, "default" per la modalità classica

# Funzione di lifecycle che inizializza automaticamente il sistema RAG all'avvio
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, qa_chain
    try:
        print("Inizializzazione automatica del sistema RAG in corso...")
        if os.path.exists("rag_index"):
            vector_store = load_vectorstore()
        else:
            docs = load_and_split("docs/")
            vector_store = create_vectorstore(docs)
        
        # Ottieni il retriever dall'indice
        retriever = vector_store.as_retriever()
        # Inizializza il modello LLM usando Ollama (local o tramite API del servizio Ollama)
        llm = OllamaLLM(model="gemma:2b")  
        # Crea la RetrievalQA chain che combina il retriever e il LLM generativo
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
        print("Sistema RAG pronto.")
        yield
    except Exception as e:
        print(f"Errore durante l'inizializzazione automatica del sistema RAG: {e}")
        yield

# Crea l'app FastAPI utilizzando il lifespan per l'inizializzazione automatica
app = FastAPI(lifespan=lifespan)

# Aggiungi CORS per consentire richieste da qualsiasi origine (modifica in produzione se necessario)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monta la cartella "static" per servire file statici (ad esempio index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Endpoint per la home page (serve il file index.html)
@app.get("/", response_class=HTMLResponse)
async def get_home():
    # Usa FileResponse per restituire il file statico index.html
    return FileResponse("static/index.html")

# Endpoint per inizializzare manualmente il sistema RAG
@app.post("/initialize_rag")
async def initialize_rag(force: bool = False):
    global vector_store, qa_chain

    if not force and os.path.exists("rag_index"):
        return {"message": "Il sistema RAG è già inizializzato."}
    
    docs = load_and_split("docs/")
    vector_store = create_vectorstore(docs)
    # retriever = vector_store.as_retriever()
    # llm = ollama(model="gemma:2b")
    qa_chain = create_qa_chain(vector_store)

    return {"message": "Sistema RAG inizializzato con successo."}

if __name__ == "__main__":
    folder_path = "docs"
    debounce_seconds = 300

    watcher = Watcher(folder_path, debounce_sec = debounce_seconds)
    try:
        watcher.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()
        print("Monitoraggio interrotto.")

def log_answer(question: str, answer: str, mode: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Crea una stringa di log con tutte le informazioni
    log_entry = f"[{timestamp}] MODE: {mode}\nQ: {question}\nA: {answer}\n\n"
    with open("logs/agent_output.log", "a", encoding="utf-8") as f:
        f.write(log_entry)

# Endpoint API per rispondere alle domande
@app.post("/ask")
async def ask(request: Request):
    global vector_store, qa_chain
    data = await request.json()
    question = data.get("question", "")
    # context = "Explain in details and specify the resource page number. The answer must be in italian."
    mode = data.get("mode")

    if mode == "rag":
        # Se il sistema non è inizializzato, restituisce un messaggio di errore
        if vector_store is None or qa_chain is None:
            return {"answer": "RAG system is not initialized. Upload a document first."}
        
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        log_answer(question, answer, mode="rag")
        return {"answer": result["result"]}
    
    elif mode == "default":
        # Modalità classica che usa il comportamento definito in ask_agent
        answer = ask_agent(question)
        log_answer(question, answer, mode="default")
        return {"answer": answer}