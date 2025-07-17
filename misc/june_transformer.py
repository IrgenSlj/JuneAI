import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMoE(nn.Module):
    def __init__(self, hidden_size, num_experts=4, expert_hidden=128):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_size, expert_hidden),
            nn.ReLU(),
            nn.Linear(expert_hidden, hidden_size)
        ) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        # x: [batch, seq_len, hidden]
        batch, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)  # [batch*seq_len, hidden]
        gate_scores = F.softmax(self.gate(x_flat), dim=-1)  # [batch*seq_len, num_experts]
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)  # [batch*seq_len, num_experts, hidden]
        output = (gate_scores.unsqueeze(-1) * expert_outputs).sum(dim=1)  # [batch*seq_len, hidden]
        output = output.view(batch, seq_len, hidden)  # [batch, seq_len, hidden]
        return output

class MiniMoETransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_experts, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.moe = SimpleMoE(hidden_size, num_experts)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        moe_output = self.moe(x)
        x = self.norm2(x + self.dropout(moe_output))
        return x

class MiniMoETransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=4, num_heads=4, num_experts=4, max_len=128, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden_size))
        self.layers = nn.ModuleList([
            MiniMoETransformerBlock(hidden_size, num_heads, num_experts, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids) + self.pos_embed[:, :input_ids.size(1), :]
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits