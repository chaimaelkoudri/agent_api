import openai
import os
from wrapper_base import LLMWrapper

class OpenAIAgent(LLMWrapper):
    def __init__(self, model="gpt-4"):
        super().__init__(name=f"openai_{model}")
        # Legge la chiave API da variabile d'ambiente
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model

    async def __call__(self, prompt: str, **kwargs) -> str:
        # Invia il prompt all'API OpenAI usando il formato "chat"
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=512
        )
        # Estrae la risposta testuale
        return response["choices"][0]["message"]["content"]
