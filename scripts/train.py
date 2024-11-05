import sys
import os

# Add the project root directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim

from dataloader.dataset import Dataset
from models.centernet import CenterNet
from loss import centerloss



# Load configuration from config.json
with open("dataloader/config.json", "r") as config_file:
    config = json.load(config_file)

# Define constants
MODEL_SCALE = config["MODEL_SCALE"]
input_size = config["input_size"]
batch_size = config["batch_size"]
in_scale = config["IN_SCALE"]
folder_path = "data"
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
])

# Dataset and data loaders
dataset = Dataset(input_size=input_size, 
                  model_scale=MODEL_SCALE, 
                  in_scale=in_scale, 
                  folder_path=folder_path, 
                  transform=train_transform)

print(len(dataset))
# Adjust train-test split sizes based on your dataset size
train_set, test_Set = torch.utils.data.random_split(dataset, [51,600])

print(len(train_set))

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=1, shuffle=True, num_workers=0
)
test_loader=torch.utils.data.DataLoader(
    test_Set, batch_size=1, shuffle=True, num_workers=0
)
# Model, device, and optimizer
model = CenterNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Initialize logging list and define number of epochs
logs = []
epochs = 10  # Define the number of epochs

# Best model saving mechanism
best_loss = float('inf')  # Initialize the best loss to infinity
best_epoch = 0

# Training function
def train(epoch):
    global best_loss, best_epoch
    model.train()
    print(f'Epoch {epoch+1}/{epochs}')
    running_loss = 0.0
    running_mask = 0.0
    running_regr = 0.0
    t = tqdm(train_loader)
    
    for idx, (img, hm, regr) in enumerate(t):       
        # Move data to device
        img = img.to(device)
        hm_gt = hm.to(device)
        regr_gt = regr.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        hm_pred, regr_pred = model(img)
        preds = torch.cat((hm_pred, regr_pred), 1)
        
        # Compute losses
        loss, mask_loss, regr_loss = centerloss(preds, hm_gt, regr_gt)
        
        # Accumulate loss for logging
        running_loss += loss.item()
        running_mask += mask_loss.item()
        running_regr += regr_loss.item()
        
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        
        # Update tqdm description
        t.set_description(f'Training (Loss={running_loss/(idx+1):.3f}, Mask={running_mask/(idx+1):.4f}, Regr={running_regr/(idx+1):.4f})')
        
    # Epoch loss logging
    avg_loss = running_loss / len(train_loader)
    avg_mask_loss = running_mask / len(train_loader)
    avg_regr_loss = running_regr / len(train_loader)
    print(f'Train Loss: {avg_loss:.4f}')
    print(f'Mask Loss: {avg_mask_loss:.4f}')
    print(f'Regression Loss: {avg_regr_loss:.4f}')
    
    # Save epoch logs
    log_epoch = {
        'epoch': epoch + 1,
        'lr': optimizer.state_dict()['param_groups'][0]['lr'],
        'loss': avg_loss,
        'mask': avg_mask_loss,
        'regr': avg_regr_loss
    }
    logs.append(log_epoch)

    # Save the model if it's the best so far (based on training loss)
    if avg_loss < best_loss:
        print(f"Saving model with improved loss ({best_loss:.4f} -> {avg_loss:.4f})")
        best_loss = avg_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "best_model.pth")  # Save the model with the lowest loss


for epoch in range(epochs):
    train(epoch)

print(f"Best model found at epoch {best_epoch} with loss {best_loss:.4f}")
