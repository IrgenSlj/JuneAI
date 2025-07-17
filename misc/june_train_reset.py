import os
import re
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from june_transformer import MiniMoETransformer
from tqdm import tqdm

# Config
from june_config import (
    model_name, vocab_size, model_dir, hidden_size, num_layers, num_heads,
    num_experts, dropout, max_len, batch_size, epochs, lr, device
)

# Dataset
dataset = load_dataset("blended_skill_talk", split="train")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def encode(example):
    # Safely get the first utterance text if available
    msgs = example.get("free_messages", [])
    if isinstance(msgs, list) and len(msgs) > 0 and isinstance(msgs[0], dict):
        text = msgs[0].get("text", "")
    else:
        text = ""
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=64)
    return {"input_ids": tokens["input_ids"]}

dataset = dataset.map(encode)
dataset.set_format(type="torch", columns=["input_ids"])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model (random init, with dropout)
model = MiniMoETransformer(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_heads=num_heads,
    num_experts=num_experts,
    dropout=dropout
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop (as before)
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix({
            "Batch": step + 1,
            "Loss": f"{loss.item():.4f}",
            "AvgLoss": f"{running_loss / (step + 1):.4f}"
        })
    print(f"Epoch {epoch+1} completed. Average loss: {running_loss / len(dataloader):.4f}")

# Save model
save_path = os.path.join(model_dir, "june_0_0.pt")
torch.save(model.state_dict(), save_path)
print(f"[INFO] Model saved as {save_path}")