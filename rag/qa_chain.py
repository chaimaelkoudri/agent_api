from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama  # Usa LLM locale di Ollama

# Crea una catena RetrievalQA che interroga i documenti vettoriali
def create_qa_chain(vectorstore):
    # Converte il database vettoriale in un "retriever"
    retriever = vectorstore.as_retriever(
        search_type="similarity", # Usa la similarità vettoriale per cercare documenti
        search_kwargs={"k": 3} # Ritorna i 3 documenti più simili alla query
    )

    # Carica un modello Ollama locale
    llm = ChatOllama(
        model="llama3.1",
        num_predict=1024
    )

    # Crea la catena domanda-risposta collegando il modello e il retriever
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )
