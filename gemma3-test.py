# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="google/gemma-3-1b-it") #google/gemma-3-270m")
messages = [
    {"role": "user", "content": "Who is the current prime minister of Albania?"},
]
pipe(messages)