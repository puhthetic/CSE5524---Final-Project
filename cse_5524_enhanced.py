# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
import kagglehub
kagglehub.login()

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

animal_clef_2025_path = kagglehub.competition_download('animal-clef-2025')

print('Data source import complete.')

import timm
import torch
import torch.nn as nn
import torch.optim as optim

num_classes = 1102  # Adjust based on your dataset
model = timm.create_model('convnext_base_in22k', pretrained=True)
model.head = nn.Linear(model.head.in_features, num_classes)  # Adjust for your dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)  # Adjust T_max based on epochs

"""# RUN THIS"""

import os
import numpy as np
import pandas as pd
import torch
import timm
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

from wildlife_datasets.datasets import AnimalCLEF2025

def create_sample_submission(dataset_query, predictions, file_name='sample_submission.csv'):
    df = pd.DataFrame({
        'image_id': dataset_query.metadata['image_id'],
        'identity': predictions
    })
    df.to_csv(file_name, index=False)

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
root = animal_clef_2025_path

# Transforms
transform_display = T.Compose([
    T.Resize([384, 384]),
])
transform = T.Compose([
    *transform_display.transforms,
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Augmentation transform (used for training)
transform_train = T.Compose([
    *transform_display.transforms,
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Load datasets
dataset = AnimalCLEF2025(root, transform=transform, load_label=True)
dataset_database = dataset.get_subset(dataset.metadata['split'] == 'database')
dataset_query = dataset.get_subset(dataset.metadata['split'] == 'query')

# Apply augmentation only to training set
dataset_database.transform = transform_train

# Label processing
unique_labels = sorted(set(dataset_database.labels_string))
label_to_idx = {label: i for i, label in enumerate(unique_labels)}
idx_to_label = {i: label for label, i in label_to_idx.items()}

# Patch labels in dataset to use indices
dataset_database.labels = [label_to_idx[l] for l in dataset_database.labels_string]

# Dataloaders
train_loader = DataLoader(dataset_database, batch_size=16, shuffle=True, num_workers=2)
query_loader = DataLoader(dataset_query, batch_size=16, shuffle=False, num_workers=2)

# Load and configure model
model_name = 'convnext_base_in22k'
model = timm.create_model(model_name, num_classes=num_classes, pretrained=True)
model.to(device)

# Training setup
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)
epochs = 10  # Adjust as needed

# Training loop
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images = images.to(device)  # Move images to the device

        # Convert each label string to an integer index using label_to_idx
        labels = [label_to_idx[label] for label in labels]  # Convert tuple of strings to a list of integers

        # Convert the list of labels to a tensor and move it to the device
        labels = torch.tensor(labels).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}")

# Modified Inference with confidence threshold tuning
model.eval()
all_preds = []
all_probs = []  # Store probabilities for analysis

best_threshold = 0.6  

with torch.no_grad():
    for images, _ in tqdm(query_loader, desc="Inference"):
        images = images.to(device)
        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        max_probs, preds = torch.max(probabilities, dim=1)
        
        # Store probabilities for analysis
        all_probs.extend(max_probs.cpu().numpy())
        
        # Apply confidence threshold (using best_threshold)
        preds[max_probs <= best_threshold] = -1  # -1 will represent "new_individual"
        all_preds.extend(preds.cpu().numpy())

# Convert indices back to labels, handling "new_individual" case
predictions = []
for p in all_preds:
    if p == -1:
        predictions.append("new_individual")
    else:
        predictions.append(idx_to_label[p])

# Save submission
create_sample_submission(dataset_query, predictions, file_name='submission.csv')