from datasets import load_dataset
import pandas as pd

# 1. Load the dataset
dataset = load_dataset("blended_skill_talk", split="train", cache_dir="Training/datasets")

# 2. Convert to pandas DataFrame
df = dataset.to_pandas()

# 3. Inspect the first few rows
print("First 5 rows:")
print(df.head())

# 4. Show all columns
print("\nColumns:")
print(df.columns)

# 5. Look at a sample conversation
print("\nFirst row content:")
print(df.iloc[0])