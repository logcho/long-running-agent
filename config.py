import os

# Ollama Settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Agent Limits
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "30"))
# Number of active messages in context before compressing (summarizing)
CONTEXT_WINDOW_LIMIT = int(os.getenv("CONTEXT_WINDOW_LIMIT", "8"))

# Storage Locations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE_PATH = os.path.join(BASE_DIR, "memory.json")
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")

# Create directories if they don't exist
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
