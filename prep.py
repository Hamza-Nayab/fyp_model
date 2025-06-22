# STEP 1: Make sure required packages are installed
# Run this in your terminal before running the script:
# pip install datasets transformers scikit-learn pandas torch tqdm

import pandas as pd
from datasets import load_dataset
from transformers import DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

# STEP 2: Load the unsplit emotion dataset
print("📥 Loading dataset...")
dataset = load_dataset("dair-ai/emotion", "unsplit")
df = pd.DataFrame(dataset["train"])
print(f"✅ Loaded {len(df)} samples.")

# STEP 3: Initialize tokenizer
print("\n🔧 Loading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# STEP 4: Tokenize data
print("\n✂️ Tokenizing text (this may take a moment)...")
encodings = tokenizer(
    list(tqdm(df["text"].tolist(), desc="Tokenizing")),
    truncation=True,
    padding="max_length",
    max_length=128,
    return_tensors="pt"
)

# STEP 5: Convert labels to tensor
print("\n📌 Converting labels to tensor...")
labels = torch.tensor(df["label"].tolist())

# STEP 6: Train-test split
print("\n🔀 Splitting into train/test sets...")
train_size = 0.9
X_train, X_test, y_train, y_test = train_test_split(
    encodings["input_ids"], labels, train_size=train_size, random_state=42
)
train_attention, test_attention = train_test_split(
    encodings["attention_mask"], train_size=train_size, random_state=42
)

# STEP 7: Save everything locally
print("\n💾 Saving to 'emotion_dataset_preprocessed.pt'...")
torch.save({
    "train_input_ids": X_train,
    "train_attention_mask": train_attention,
    "train_labels": y_train,
    "test_input_ids": X_test,
    "test_attention_mask": test_attention,
    "test_labels": y_test,
}, "emotion_dataset_preprocessed.pt")

print("\n✅ All done! Preprocessed data saved as 'emotion_dataset_preprocessed.pt'")
