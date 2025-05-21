from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Funzione principale che gestisce la logica dell'agente
async def ask_agent(question: str) -> str:
    # Crea un'istanza del modello AI utilizzando Ollama
    llm = OllamaLLM(model="llama3.1")

    # Definisce un template semplice per il prompt che invieremo al modello
    prompt = ChatPromptTemplate.from_template("Domanda: {question}")

    # Crea una pipeline che prima applica il template e poi passa al modello
    chain = prompt | llm

    # Esegui la pipeline e restituisci la risposta
    result = await chain.invoke({"question": question})
    return result["result"]