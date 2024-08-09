from x_transformers import TransformerWrapper, Encoder

import torch
from torch import nn
import numpy as np
import zipfile  

# unzip imbd_classification\data\data.zip
with zipfile.ZipFile('data/data.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

# Load data from data/IMDB Dataset.csv
import pandas as pd
data = pd.read_csv('data/IMDB Dataset.csv')

# delete csv file
import os
os.remove('data/IMDB Dataset.csv')

avg_len=0
max_len=0
for i in data['review']:
    max_len=max(max_len,len(i))
    avg_len+=len(i)
print(max_len)
print(avg_len/len(data['review']))

# preprocess data, transform positive/negative to 1/0 for sentiment column
data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# split data into train and test
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

import re
vocab = {}

# Updated encode_text function for word tokenization
def encode_text(text, vocab, max_len=1024):
    text = text.lower()  # Convert text to lowercase
    words = re.findall(r'\w+|[^\w\s]', text)  # Split into words and punctuation
    # Create a vocabulary mapping words to unique integers
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab) + 1  # Start token indices from 1

    # Encode the words using the vocabulary
    encoded_text = [vocab[word] for word in words if word in vocab]
    
    # Truncate or pad to max_len
    encoded_text = encoded_text[:max_len]
    mask = [True] * len(encoded_text) + [False] * (max_len - len(encoded_text))
    return np.array(encoded_text + [0] * (max_len - len(encoded_text))), np.array(mask)

# Create dataset
class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_len=1024):
        self.data = data
        self.max_len = max_len
        self.text = data['review'].values
        self.target = data['sentiment'].values
        self.tokenized = []

        # Encode all reviews at initialization
        for t in self.text:
            tokenized_text, mask = encode_text(t, vocab, max_len)
            self.tokenized.append((tokenized_text, mask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokenized_text, mask = self.tokenized[idx]
        return torch.tensor(tokenized_text), torch.tensor(self.target[idx]), torch.tensor(mask)

# Create dataloaders
train_dataset = IMDBDataset(train_data)
test_dataset = IMDBDataset(test_data)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

print("vocab size:", len(vocab))

# Create model
model = TransformerWrapper(
    num_tokens=len(vocab) + 1,
    max_seq_len=1024,
    logits_dim=1,
    use_cls_token=True,
    attn_layers=Encoder(
        dim=128,
        depth=4,
        heads=4,
        ff_glu=True,
        attn_flash=True,
        
    )
)
print("Params: ", sum(p.numel() for p in model.parameters()))

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Create loss function
loss_fn = nn.BCEWithLogitsLoss()

# Training loop
def train_loop(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    total_correct = 0
    for i, (texts, sentiments, masks) in enumerate(train_loader):
        texts = texts.to(device)
        sentiments = sentiments.to(device).float()
        masks = masks.to(device)
        optimizer.zero_grad()

        outputs = model(texts, mask = masks)
        loss = loss_fn(outputs, sentiments.unsqueeze(1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += ((outputs > 0) == sentiments.unsqueeze(1)).sum().item()

        if i % 100 == 0:
            print(f"Step {i}, Loss: {total_loss / (i + 1)}, Accuracy: {total_correct / ((i + 1) * len(texts))}")

    return model

# Evaluation loop
def test_loop(model, test_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for texts, sentiments, masks in test_loader:
            texts = texts.to(device)
            sentiments = sentiments.to(device).float()
            masks = masks.to(device)

            outputs = model(texts, mask = masks)
            loss = loss_fn(outputs, sentiments.unsqueeze(1))

            total_loss += loss.item()
            total_correct += ((outputs > 0) == sentiments.unsqueeze(1)).sum().item()
    print(f"Test Loss: {total_loss / len(test_loader)}, Test Accuracy: {total_correct / len(test_loader.dataset)}")

# Train model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
for epoch in range(3):
    print(f"Epoch {epoch}")
    model = train_loop(model, train_loader, optimizer, loss_fn, device)
    test_loop(model, test_loader, loss_fn, device)