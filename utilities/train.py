from intent_net import IntentNet  # BiLSTM model
from intent_dataset import IntentDataset
from intent_classifier import IntentClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "model"
os.makedirs(save_dir, exist_ok=True)

# === Step 1: Load and preprocess data ===
classifier = IntentClassifier()
patterns, labels = classifier.get_intents("../resources/intents/intents_uncle_cheffington.json")

vocab = classifier.build_vocab(patterns)
encoded_labels, label2idx, idx2label = classifier.encode_labels(labels)

max_len = 50
X_train = classifier.encode_patterns(patterns, vocab, max_len)
y_train = encoded_labels

# === Step 2: Dataset and Dataloader ===
dataset = IntentDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# === Step 3: Model, Loss, Optimizer ===
model = IntentNet(
    vocab_size=len(vocab),
    embedding_dim=50,
    hidden_dim=64,
    output_dim=len(label2idx)  # should be 2 for greeting and recipe_request
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Step 4: Training Loop ===
n_epochs = 15

for epoch in range(n_epochs):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{n_epochs} | Loss: {avg_loss:.4f}")

# === Step 5: Save Model and Metadata ===
torch.save(model, os.path.join(save_dir, "intent_model_full.pth"))

with open(os.path.join(save_dir, "vocab.json"), "w") as f:
    json.dump(vocab, f)

with open(os.path.join(save_dir, "labels.json"), "w") as f:
    json.dump({
        "label2idx": label2idx,
        "idx2label": idx2label
    }, f)

print("✅ Model and metadata saved.")
