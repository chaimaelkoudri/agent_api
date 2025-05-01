from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_and_split(folder_path: str):

    all_docs = []

    # Scansiona tutti i file nella cartella
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            try:
                pdf_path = os.path.join(folder_path, filename)
                loader = PyPDFLoader(pdf_path)
                documents = loader.load() 
                all_docs.extend(documents)
            except Exception as e:
                print(f"Errore nel caricamento del file {filename}: {e}")

    # Divide i documenti in blocchi (chunk) gestibili
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Dimensione massima di ogni pezzo
        chunk_overlap=100    # Sovrapposizione per mantenere il contesto
    )

    split_docs = splitter.split_documents(all_docs)

    # Ritorna i documenti spezzettati
    return split_docs