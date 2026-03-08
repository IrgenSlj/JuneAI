"""Configuration loader for JuneAI.

Reads environment variables from .env file.
"""

import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
MEMORY_DIR = os.getenv("MEMORY_DIR", ".june_memory")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file.")
