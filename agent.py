from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from EasyJailbreak.easyjailbreak.attacker.attacker_base import ModelBase

class FastAPIAgent(ModelBase):  # Eredita da ModelBase per garantire la compatibilità
    def __init__(self, model="llama3.1"):
        # Inizializza il modello LLM
        self.model = model
        self.llm = OllamaLLM(model=self.model)  # Crea una nuova istanza di OllamaLLM

    # Metodo asincrono che invia una domanda al modello e ottiene la risposta
    async def ask_agent(self, question: str, mode: str):
        # Definisce un template di chat che prende una variabile 'question'
        prompt = ChatPromptTemplate.from_template("Domanda: {question}")
        
        # Crea una pipeline tra il template e il modello LLM
        # Il prompt viene applicato prima di inviare la domanda al modello
        chain = prompt | self.llm
        
        # Esegui la pipeline per ottenere la risposta (la chiamata al modello LLM)
        result = await chain.invoke({"question": question})

        # Restituisce la risposta ottenuta dal modello
        return result["result"]

# Esempio di utilizzo dell'agente:
# Creiamo una nuova istanza di FastAPIAgent
agent = FastAPIAgent()

# Invia una domanda al modello (nota: 'mode' non viene usato in questa versione)
response = agent.ask("Qual è la capitale della Francia?", mode="default")

# Stampa la risposta ottenuta dal modello
print(response)
