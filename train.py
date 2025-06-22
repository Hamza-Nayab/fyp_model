# Make sure required libraries are installed:
# pip install torch transformers scikit-learn tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizerFast
from torch.optim import AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm

# STEP 1: Load preprocessed data
print("📦 Loading preprocessed data...")
data = torch.load("emotion_dataset_preprocessed.pt")

train_dataset = TensorDataset(
    data["train_input_ids"], data["train_attention_mask"], data["train_labels"]
)
test_dataset = TensorDataset(
    data["test_input_ids"], data["test_attention_mask"], data["test_labels"]
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# STEP 2: Define model
class SentimentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]  # CLS token
        return self.classifier(hidden_state)

# STEP 3: Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentClassifier(num_classes=6).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# STEP 4: Training loop
print("🚀 Training model...")
epochs = 3
model.train()

for epoch in range(epochs):
    total_loss = 0
    for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"📉 Epoch {epoch+1} Loss: {avg_loss:.4f}")

# STEP 5: Evaluation
print("\n📊 Evaluating model...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\n✅ Classification Report:")
print(classification_report(all_labels, all_preds))
