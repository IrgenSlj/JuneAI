# JuneAI 
(work in progress)

**JuneAI** is a privacy-first, offline AI assistant that remembers your conversations using local storage and Retrieval-Augmented Generation (RAG). Designed to be efficient, lightweight, and deployable anywhere — from personal laptops to Kubernetes clusters — JuneAI uses open-source models and a modular architecture built with Python, LangChain, and Hugging Face Transformers.

---

## Goal

The aim of **JuneAI** is to create a customizable, memory-capable AI assistant that:
- Runs fully **offline**, protecting user privacy
- Uses **RAG** to remember and retrieve past conversations
- Supports **interchangeable LLMs** from Hugging Face via Transformers
- Runs efficiently on **consumer hardware**
- Is containerized with **Docker** and scalable via **Kubernetes**
- Is extensible for both personal and professional applications

---

## Features

- **Persistent Memory** — Stores conversation history locally using embeddings
- **RAG Pipeline** — Retrieves past chats relevant to your current prompt
- **Pluggable Models** — Swap out LLMs from Hugging Face (small, efficient models)
- **Powered by LangChain** — Handles chaining, prompt assembly, and memory logic
- **Containerized** — Deploy anywhere with Docker and Kubernetes
- **Built with PyTorch** — Ensures native performance and model compatibility

---

## Tech Stack

| Component           | Technology                |
|---------------------|----------------------------|
| Language            | Python 3.10+               |
| LLMs                | Hugging Face Transformers |
| Frameworks          | LangChain, PyTorch         |
| Embeddings & RAG    | LangChain, FAISS, Pandas   |
| Containerization    | Docker                     |
| Orchestration       | Kubernetes                 |
| Memory Persistence  | FAISS / Chroma + CSV/Parquet via Pandas |
| Config Management   | `config.yaml` / `.env`     |

---

## Architecture & Workflow

User CLI  --->  Memory Retriever  --->  Prompt Assembler 

LLM Inference Engine                                         
Transformers + PyTorch            

Response Output  <----  Conversation Storage               

## How It Works

1. **Startup**  
   - Docker or Python script loads `config.yaml`  
   - Initializes LLM, embedding model, and FAISS/Chroma DB

2. **User Input**  
   - You send a message via CLI, Web UI, or API

3. **Memory Retrieval (RAG)**  
   - Embeddings of your query are computed  
   - LangChain fetches top-K relevant memory chunks using FAISS

4. **Prompt Assembly**  
   - Retrieved context + user input is formatted into a prompt  
   - Prompt passed to chosen Hugging Face model

5. **LLM Response Generation**  
   - LLM generates a context-aware response using PyTorch backend

6. **Memory Update**  
   - New interaction is embedded and stored locally using Pandas + FAISS

---

## Project Structure

JuneAI/

- src/
   - main.py           # Entry point
   - llm_engine.py     # Model loading and inference
   - memory_manager.py # Embedding & RAG logic
   - config.py         # YAML/ENV config loader
   - utils.py          # Misc utilities
   - retriever.py      # LangChain + FAISS wrapper

- memory/              # Local conversation DB

- models/              # Downloaded or cached HF models

- docker/
   - Dockerfile        # Build container

- kubernetes/
   - deployment.yaml   # K8s manifest

- requirements.txt

- config.yaml

- README.md

## Privacy & Offline Use

All data is:

Stored locally
Not sent to any external APIs
Fully deletable with one command: python src/clear_memory.py

## Contributing
Pull requests, issues, and ideas are welcome!
Please submit an issue first if you plan to make major changes.

## License
This project is licensed under the MIT License.

## Maintainer
JuneAI is maintained by repo admin.
A project aiming to make AI assistants offline-first, memory-aware, and open-source for everyone.
