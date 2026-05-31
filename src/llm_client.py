import config
from src.ollama_client import OllamaClient
from src.openai_client import OpenAIClient

def get_llm_client():
    provider = getattr(config, "LLM_PROVIDER", "ollama").lower()
    if provider == "openai":
        return OpenAIClient()
    return OllamaClient()
