import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from june_transformer import MiniMoETransformer, MiniMoETransformerBlock
from june_config import (
    model_name, vocab_size, model_dir, hidden_size, num_layers, num_heads,
    num_experts, dropout, max_len, batch_size, epochs, lr, device
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MiniMoETransformer(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_heads=num_heads,
    num_experts=num_experts,
    dropout=dropout,
    max_len=max_len
)
model.load_state_dict(torch.load("models/june_0_0.pt", map_location=device))
model.to(device)
model.eval()

test_sentence = "The quick brown fox jumps over the lazy dog."
input_ids = torch.tensor([tokenizer.encode(test_sentence, max_length=32, truncation=True)]).to(device)
labels = input_ids.clone()

with torch.no_grad():
    logits = model(input_ids)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
    perplexity = torch.exp(loss)
    preds = logits.argmax(dim=-1)
    accuracy = (preds == labels).float().mean().item()

print(f"Loss: {loss.item():.4f}")
print(f"Perplexity: {perplexity.item():.4f}")
print(f"Token Accuracy: {accuracy:.4f}")

# --- Top-k Token Distribution ---
topk = 5
probs = torch.softmax(logits, dim=-1)
topk_probs, topk_indices = torch.topk(probs, topk, dim=-1)
plt.figure(figsize=(12, 4))
for i in range(min(5, input_ids.size(1))):  # Plot for first 5 tokens
    plt.subplot(1, 5, i+1)
    plt.bar(range(topk), topk_probs[0, i].cpu().numpy())
    plt.xticks(range(topk), tokenizer.convert_ids_to_tokens(topk_indices[0, i].cpu().numpy()), rotation=90)
    plt.title(f"Pos {i}")
plt.suptitle("Top-5 Token Probabilities (first 5 positions)")
plt.tight_layout()
plt.show()

# --- MoE Expert Utilization ---
with torch.no_grad():
    first_block = model.layers[0]
    if isinstance(first_block, MiniMoETransformerBlock):
        x = model.embed(input_ids) + model.pos_embed[:, :input_ids.size(1), :]
        attn_output, _ = first_block.attn(x, x, x)
        x = first_block.norm1(x + attn_output)
        moe = first_block.moe
        batch, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        gate_scores = torch.softmax(moe.gate(x_flat), dim=-1).cpu().numpy()
        expert_choices = np.argmax(gate_scores, axis=-1)
        plt.hist(expert_choices, bins=np.arange(num_experts+1)-0.5, rwidth=0.8)
        plt.xlabel("Expert Index")
        plt.ylabel("Token Count")
        plt.title("MoE Expert Utilization (First Block)")
        plt.xticks(range(num_experts))
        plt.show()

# --- Attention Entropy ---
with torch.no_grad():
    attn = first_block.attn
    attn_output, attn_weights = attn(x, x, x, need_weights=True)
    attn_weights = attn_weights[0, 0].cpu().numpy()  # [seq_len, seq_len]
    entropy = -np.sum(attn_weights * np.log(attn_weights + 1e-8), axis=-1)
    plt.plot(entropy)
    plt.xlabel("Token Position")
    plt.ylabel("Attention Entropy")
    plt.title("Attention Entropy (First Head, First Block)")
    plt.show()

# --- Activation Statistics ---
activations = []
def hook_fn(module, inp, out):
    activations.append(out.detach().cpu().numpy())
hook = model.layers[0].register_forward_hook(hook_fn)
with torch.no_grad():
    _ = model(input_ids)
hook.remove()
acts = activations[0].reshape(-1)
plt.hist(acts, bins=50)
plt.title("Activation Distribution (First Block Output)")
plt.xlabel("Activation Value")
plt.ylabel("Frequency")
plt.show()