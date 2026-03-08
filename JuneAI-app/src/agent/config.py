"""Configuration loader for JuneAI.

Reads environment variables from .env file.
"""

import os

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/llama-3.1-8b-instruct:free")
MEMORY_DIR = os.getenv("MEMORY_DIR", ".june_memory")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set. Add it to your .env file.")
