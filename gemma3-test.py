# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="google/gemma-3-1b-it")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)