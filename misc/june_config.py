import torch, os

# Model architecture

# Training

model_name = "distilbert-base-uncased"  # For tokenizer only
vocab_size = 30522
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

hidden_size = 256   # 128   # if device == "cpu" else 256
num_layers = 4      # 2     # if device == "cpu" else 4
num_heads = 4       # 2     # if device == "cpu" else 4
num_experts = 4     # 2     # if device == "cpu" else 4
batch_size = 8      # 4     # if device == "cpu" else 16
epochs = 5          # if device == "cpu" else 10
dropout = 0.2       # if device == "cpu" else 0.1
lr = 5e-4           # if device == "cpu" else 3e-4
max_len = 128       # 64    # if device == "cpu" else 128

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"