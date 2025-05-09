import requests

class FastAPIAgent:
    def __init__(self, base_url: str):
        self.url = base_url
        self.name = "fastapi_agent"

    def ask(self, prompt: str, mode: str = "default") -> str:
        response = requests.post(self.url, json={"question": prompt, "mode": mode})
        response.raise_for_status()
        data = response.json()
        return data["answer"]["text"] if isinstance(data["answer"], dict) and "text" in data["answer"] else data["answer"]
