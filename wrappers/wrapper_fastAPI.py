import aiohttp
from wrappers.wrapper_base import LLMWrapper

class FastAPIAgent(LLMWrapper):
    def __init__(self, endpoint: str):
        # Nome descrittivo
        super().__init__(name="fastapi_agent")
        self.endpoint = endpoint

    async def __call__(self, prompt: str, **kwargs) -> str:
        # Invia il prompt all’API FastAPI tramite POST
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json={"prompt": prompt}) as resp:
                data = await resp.json()
                # Ritorna il campo "response" come output del modello
                return data.get("response", "")
