import json

import numpy as np
import torch
import torch.nn as nn
from neo4j import GraphDatabase
from torch.utils.data import DataLoader, Dataset

from model import NeuralNet
from nltk_utils import stem, tokenize

# Connect to Neo4j
uri = "bolt://localhost:7687"
username = "neo4j"
password = "your_password"
driver = GraphDatabase.driver(uri, auth=(username, password))

# Read intents data from JSON file
with open('intents.json') as f:
    intents_data = json.load(f)

all_words = []
tags = []

# Loop through each intent in intents data
for intent in intents_data['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)

# Stem and remove duplicates from all_words
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create training data
X_train = []
y_train = []
for intent in intents_data['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        bag = [1 if stem(word) in w else 0 for word in all_words]
        X_train.append(bag)
        y_train.append(tags.index(tag))

# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Define Hyperparameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

# Define dataset class
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = torch.tensor(X_train, dtype=torch.float32)
        self.y_data = torch.tensor(y_train, dtype=torch.long)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Create DataLoader
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = NeuralNet(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'model.pth')

# Close the Neo4j driver
driver.close()

print('Training complete.')
