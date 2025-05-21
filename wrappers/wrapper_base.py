# Classe base astratta che tutti gli agenti LLM devono estendere
class LLMWrapper:
    def __init__(self, name="generic_llm"):
        # Nome descrittivo del modello (es. "openai_gpt-4", "fastapi_agent", ecc.)
        self.name = name

    async def __call__(self, prompt: str, **kwargs) -> str:
        """
        Metodo da implementare nei sottotipi.
        Deve restituire la risposta dell’LLM al prompt fornito.
        """
        raise NotImplementedError("Implementa __call__ nel tuo wrapper.")
