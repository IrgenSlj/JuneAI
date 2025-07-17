import os
from datasets import load_dataset

def load_blended_skill_talk(tokenizer, max_length=64, split="train", cache_dir="datasets"):
    dataset = load_dataset("blended_skill_talk", split=split, cache_dir=cache_dir)
    def encode(example):
        msgs = example.get("free_messages", [])
        if isinstance(msgs, list) and len(msgs) > 0 and isinstance(msgs[0], dict):
            text = msgs[0].get("text", "")
        else:
            text = ""
        tokens = tokenizer(text, truncation=True, padding="max_length", max_length=max_length)
        return {"input_ids": tokens["input_ids"]}
    dataset = dataset.map(encode)
    dataset.set_format(type="torch", columns=["input_ids"])
    return dataset

def load_daily_dialog(tokenizer, max_length=64, split="train", cache_dir="datasets"):
    dataset = load_dataset("daily_dialog", split=split, cache_dir=cache_dir)
    def encode(example):
        text = " ".join(example["dialog"])
        tokens = tokenizer(text, truncation=True, padding="max_length", max_length=max_length)
        return {"input_ids": tokens["input_ids"]}
    dataset = dataset.map(encode)
    dataset.set_format(type="torch", columns=["input_ids"])
    return dataset

def load_openassistant_oasst1(tokenizer, max_length=128, split="train", cache_dir="datasets"):
    dataset = load_dataset("OpenAssistant/oasst1", split=split, trust_remote_code=True, cache_dir=cache_dir)
    def encode(example):
        text = example.get("text", "")
        tokens = tokenizer(text, truncation=True, padding="max_length", max_length=max_length)
        return {"input_ids": tokens["input_ids"]}
    dataset = dataset.map(encode)
    dataset.set_format(type="torch", columns=["input_ids"])
    return dataset

def preview_dataset(dataset, n=5, tokenizer=None):
    print(f"Dataset size: {len(dataset)}")
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        print(f"Sample {i}:")
        print("  input_ids:", item["input_ids"])
        if tokenizer:
            print("  decoded:", tokenizer.decode(item["input_ids"]))

# --- Section to print the data and content of all datasets ---
def print_all_datasets(tokenizer, max_length=64, cache_dir="datasets"):
    print("\n--- Blended Skill Talk ---")
    bst = load_blended_skill_talk(tokenizer, max_length=max_length, cache_dir=cache_dir)
    preview_dataset(bst, n=3, tokenizer=tokenizer)

    print("\n--- DailyDialog ---")
    dd = load_daily_dialog(tokenizer, max_length=max_length, cache_dir=cache_dir)
    preview_dataset(dd, n=3, tokenizer=tokenizer)

    print("\n--- OpenAssistant/oasst1 ---")
    oa = load_openassistant_oasst1(tokenizer, max_length=128, cache_dir=cache_dir)
    preview_dataset(oa, n=3, tokenizer=tokenizer)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    # You may want to set your model name here or import from config
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print_all_datasets(tokenizer)