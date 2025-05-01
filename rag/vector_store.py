from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag.loader import load_and_split

# Crea e salva un database vettoriale a partire dai documenti
def create_vectorstore(folder_path, persist_path="rag_index"):

    docs = load_and_split(folder_path)
    # Inizializza il modello di embeddings usando un modello della libreria sentence-transformers
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Crea l'indice FAISS a partire dai documenti e dagli embeddings
    vectordb = FAISS.from_documents(docs, embeddings)
    
    # Salva l'indice in locale per poterlo riutilizzare senza ricalcolare tutto
    vectordb.save_local(persist_path)
    return vectordb

# Carica il database vettoriale da file locale
def load_vectorstore(persist_path="rag_index"):
    # Usa lo stesso modello di embeddings per mantenere coerenza con la fase di creazione
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Carica l'indice FAISS dal disco in modo sicuro (solo se il file è tuo)
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)