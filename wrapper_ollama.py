import aiohttp
import torch
from types import SimpleNamespace
from transformers import AutoTokenizer

from FuzzyAI.fuzzy.llm.providers.ollama.ollama import OllamaProvider
from easyjailbreak.models import ModelBase

class OllamaEasyJailbreakWrapper(ModelBase):
    def __init__(self, model_name: str, llm_address: str = "localhost", ollama_port: int = 11434):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        self.ollama = None  # inizializza dopo
        self.model = model_name
        self.llm_address = llm_address
        self.ollama_port = ollama_port
        self.session = None

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def _ensure_ollama(self):
        if self.ollama is None:
            self.ollama = OllamaProvider(
                model=self.model,
                llm_address=self.llm_address,
                ollama_port=self.ollama_port
            )
            await self.ollama.validate_models()

    async def generate(self, prompt: str, **kwargs):
        await self._ensure_session()
        await self._ensure_ollama()
        kwargs.pop("url", None)  # rimuovi "url" se presente
        response = await self.ollama.generate(prompt, **kwargs)
        return response.response if response else None

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
        if self.ollama:
            await self.ollama.close()

class OllamaAutoDANWrapper(ModelBase):
    def __init__(self, async_wrapper: OllamaEasyJailbreakWrapper):
        super().__init__()
        self.async_wrapper = async_wrapper
        self.generation_config = SimpleNamespace()
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        self.model = self  # AutoDAN expects `.model.generate()`
        self.device = torch.device('cpu')

    async def generate(self, input_ids, **kwargs):
        prompt = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return await self.async_wrapper.generate(prompt)
