from june_config import (
    model_name, vocab_size, model_dir, hidden_size, num_layers, num_heads,
    num_experts, dropout, max_len, device
)
from june_transformer import MiniMoETransformer
import torch
from transformers import AutoTokenizer

print("[INFO] Loading tokenizer and model...")
try:
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
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model or tokenizer: {e}")
    exit(1)

# Quick test inference to confirm model is ready
test_sentence = "Hello, June!"
try:
    test_ids = torch.tensor([tokenizer.encode(test_sentence, max_length=32, truncation=True)]).to(device)
    with torch.no_grad():
        test_logits = model(test_ids)
        if test_logits is None or test_logits.shape[1] == 0:
            raise ValueError("Model returned empty logits.")
    print("[INFO] Model passed quick test inference.")
except Exception as e:
    print(f"[ERROR] Model failed test inference: {e}")
    exit(1)

print("Type 'exit' to quit.")
history = []

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("[INFO] Exiting interactive session.")
        break

    print("[INFO] Preparing prompt and running inference...")
    history.append(user_input)
    prompt = " ".join(history)
    try:
        input_ids = torch.tensor([tokenizer.encode(prompt, max_length=32, truncation=True)]).to(device)
        print("Prompt:", prompt)
        print("Input IDs:", input_ids)
        print("Decoded:", tokenizer.decode(input_ids[0]))
        with torch.no_grad():
            logits = model(input_ids)
            if logits is None or logits.shape[1] == 0:
                raise ValueError("Model returned empty logits during chat.")
            next_token_id = logits[0, -1].argmax().unsqueeze(0)
            generated = [next_token_id.item()]
            for _ in range(9):
                new_input = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
                logits = model(new_input)
                if logits is None or logits.shape[1] == 0:
                    raise ValueError("Model returned empty logits during generation.")
                next_token_id = logits[0, -1].argmax().unsqueeze(0)
                generated.append(next_token_id.item())
            reply = tokenizer.decode(generated, skip_special_tokens=True)
            print("[INFO] Model generated a reply.")
            print("June:", reply)
            print("Generated token IDs:", generated)
            print("Decoded tokens:", tokenizer.convert_ids_to_tokens(generated))
            history.append(reply)
    except Exception as e:
        print(f"[ERROR] Model failed to generate a reply: {e}")
        print("[INFO] Please check your model, training, or input formatting.")