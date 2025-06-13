import os
import re
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from june_transformer import MiniMoETransformer
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
import random
import numpy as np
import matplotlib.pyplot as plt


# Config
from june_config import (
    model_name, vocab_size, model_dir, hidden_size, num_layers, num_heads,
    num_experts, dropout, max_len, batch_size, epochs, lr, device
)

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# Find latest model version
def get_latest_model_path(model_dir):
    pt_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    if not pt_files:
        return None, 0, 0
    # Get the latest file by modification time
    latest_file = max(pt_files, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
    # Try to extract n, m from filename if possible, else use 0, 0
    match = re.match(r"june_(\d+)_(\d+)\.pt", latest_file, re.IGNORECASE)
    if match:
        n, m = int(match.group(1)), int(match.group(2))
    else:
        n, m = 0, 0
    return os.path.join(model_dir, latest_file), n, m

latest_model_path, last_n, last_m = get_latest_model_path(model_dir)
if latest_model_path:
    print(f"[INFO] Loading latest model: {latest_model_path}")
else:
    print("[INFO] No previous model found. Training from scratch.")

# Data: OpenAssistant/oasst1 only
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("[INFO] Loading OpenAssistant/oasst1 dataset...")
dataset = load_dataset("OpenAssistant/oasst1", split="train", trust_remote_code=True)

def encode_oa(example):
    text = example.get("text", "")
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=max_len)
    return {"input_ids": tokens["input_ids"]}

dataset = dataset.map(encode_oa)
dataset.set_format(type="torch", columns=["input_ids"])

# Validation split
indices = list(range(len(dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.05, random_state=42)
train_dataset = dataset.select(train_indices)
val_dataset = dataset.select(val_indices)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model
model = MiniMoETransformer(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_heads=num_heads,
    num_experts=num_experts,
    dropout=dropout,
    max_len=max_len
).to(device)
if latest_model_path:
    model.load_state_dict(torch.load(latest_model_path, map_location=device))
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()

def log_training_run(model_path, extra_info=None):
    log_path = os.path.join(model_dir, "june_training_log.json")
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "dataset": "OpenAssistant/oasst1",
        "extra_info": extra_info or {}
    }
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = []
    log.append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

train_losses = []
val_losses = []

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix({
            "Batch": step + 1,
            "Loss": f"{loss.item():.4f}",
            "AvgLoss": f"{running_loss / (step + 1):.4f}"
        })
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1} completed. Average loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for val_batch in val_loader:
            input_ids = val_batch["input_ids"].to(device)
            labels = input_ids.clone()
            logits = model(input_ids)
            val_loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
            total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Validation loss: {avg_val_loss:.4f}")


# Plot losses after training
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()

# Save with incremented version
if last_m < 9:
    save_n, save_m = last_n, last_m + 1
else:
    save_n, save_m = last_n + 1, 1
save_path = os.path.join(model_dir, f"june_{save_n}_{save_m}.pt")
torch.save(model.state_dict(), save_path)
print(f"[INFO] Model saved as {save_path}")

# Log the training run
log_training_run(
    model_path=save_path,
    extra_info={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "device": device,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "max_len": max_len,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_experts": num_experts,
        "dropout": dropout,
    }
)

print("[INFO] Training complete. Exiting.")
exit(0)