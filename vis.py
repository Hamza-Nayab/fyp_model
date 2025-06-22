import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Path to your file (adjust if needed)
FILE_PATH = "emotion_dataset_preprocessed.pt"

# Load the .pt file
print("📂 Loading data...")
data = torch.load(FILE_PATH, map_location=torch.device("cpu"))

# Combine all labels from train and test
labels = torch.cat([data["train_labels"], data["test_labels"]]).tolist()

# Define emotion label names (based on 'dair-ai/emotion' dataset)
label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Count label frequencies
label_counts = Counter(labels)
counts = [label_counts.get(i, 0) for i in range(len(label_names))]

# Plot label distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=label_names, y=counts, palette="pastel")
plt.title("Emotion Label Distribution")
plt.ylabel("Number of Samples")
plt.xlabel("Emotion")
plt.tight_layout()
plt.show()

# Optional: Plot token length distribution
token_lengths = (
    torch.cat([data["train_attention_mask"], data["test_attention_mask"]], dim=0)
    .sum(dim=1)
    .tolist()
)

plt.figure(figsize=(8, 5))
sns.histplot(token_lengths, bins=30, kde=True, color="skyblue")
plt.title("Token Length Distribution")
plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
