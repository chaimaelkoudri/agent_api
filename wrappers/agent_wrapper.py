import requests
from EasyJailbreak.easyjailbreak.attacker.attacker_base import ModelBase
from agent import ask_agent

class FastAPIAgent(ModelBase):
    def __init__(self, base_url: str):
        self.url = base_url
        self.name = "fastapi_agent"

    async def ask_agent(self, prompt: str, mode: str = "default") -> str:
        return await ask_agent(prompt)