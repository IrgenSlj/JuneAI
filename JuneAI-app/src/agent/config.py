"""Configuration loader for JuneAI.

Reads environment variables from .env file.
"""

import os

from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "phi3:mini")
MEMORY_DIR = os.getenv("MEMORY_DIR", ".june_memory")
