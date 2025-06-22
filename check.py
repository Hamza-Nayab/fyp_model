import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertModel

# Define the model class (same as during training)
class SentimentClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(SentimentClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0])

# Load model
model = SentimentClassifier()
model.load_state_dict(torch.load("best_model.pt", map_location=torch.device("cpu")))
model.eval()

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Label mapping (from dair-ai/emotion)
label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Interactive loop
print("🔁 Enter a sentence to analyze (type 'exit' to quit):")
while True:
    text = input("You: ")
    if text.strip().lower() == "exit":
        print("👋 Exiting sentiment analyzer.")
        break

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
        predicted_label = torch.argmax(outputs, dim=1).item()

    print(f"🧠 Predicted Emotion: {label_map[predicted_label]}")