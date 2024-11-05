import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from tqdm import tqdm
import json

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from dataloader.dataset import Dataset
from models.centernet import CenterNet


# Load configuration from config.json
with open("dataloader\config.json", "r") as config_file:
    config = json.load(config_file)


# Define constants
MODEL_SCALE = config["MODEL_SCALE"]
input_size = config["input_size"]
batch_size = config["batch_size"]
in_scale = config["IN_SCALE"]
MODEL_PATH = config["MODEL_PATH"]
folder_path = "data"

# Define transforms
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

# Adjust train-test split sizes based on your dataset size
train_set, test_set = torch.utils.data.random_split(dataset, [51, len(dataset) - 51])

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=1, shuffle=True, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=True, num_workers=0
)


# Define pooling function
def pool(hm):
    # Apply a max-pooling layer to smooth the heatmap (non-maximum suppression)
    return torch.nn.functional.max_pool2d(torch.tensor(hm).unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze().numpy()

# Prediction to bounding box function
def pred2box(hm, regr, thresh=0.99):
    pred = hm > thresh
    pred_center = np.where(pred)
    pred_r = regr[:, pred].T

    boxes = []
    scores = hm[pred]
    for i, b in enumerate(pred_r):
        arr = np.array([
            pred_center[1][i] * MODEL_SCALE - b[0] * input_size // 2,
            pred_center[0][i] * MODEL_SCALE - b[1] * input_size // 2,
            int(b[0] * input_size),
            int(b[1] * input_size)
        ])
        arr = np.clip(arr, 0, input_size)
        boxes.append(arr)
    return np.asarray(boxes), scores

# Functions for visualizing results
def showbox(img, hm, regr, thresh=0.9):
    boxes, _ = pred2box(hm, regr, thresh=thresh)
    print("Predicted boxes:", boxes.shape)
    sample = img.copy()
    for box in boxes:
        cv2.rectangle(
            sample,
            (int(box[0]), int(box[1] + box[3])),
            (int(box[0] + box[2]), int(box[1])),
            (220, 0, 0), 3
        )
    return sample

def showgtbox(img, hm, regr, thresh=0.9):
    boxes, _ = pred2box(hm, regr, thresh=thresh)
    print("Ground Truth boxes:", boxes.shape)
    sample = img.copy()
    for box in boxes:
        cv2.rectangle(
            sample,
            (int(box[0]), int(box[1] + box[3])),
            (int(box[0] + box[2]), int(box[1])),
            (0, 220, 0), 3
        )
    return sample

model = CenterNet()
model = model.load_state_dict(torch.load(MODEL_PATH))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test loader (reuse train_set for demonstration, but ideally this should be a separate test set)
test_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)

# Inference loop
for img, hm_gt, regr_gt in test_loader:
    with torch.no_grad():
        # Forward pass
        hm, regr = model(img.to(device).float())
    
    # Process images and predictions for visualization
    img = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy(order='C')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detach and convert to numpy
    hm = torch.sigmoid(hm).cpu().numpy().squeeze(0).squeeze(0)
    regr = regr.cpu().numpy().squeeze(0)
    
    # Apply pooling to heatmap (non-maximum suppression)
    hm = pool(hm)

    # Plot thresholded heatmap
    plt.imshow(hm > 0.7, cmap='hot')
    plt.colorbar()
    plt.title("Thresholded Heatmap")
    plt.show()

    # Show predicted boxes on image
    sample = showbox(img, hm, regr, thresh=0.6)
    
    # Show ground truth boxes on the same image
    sample = showgtbox(sample, hm_gt.squeeze(0).cpu().numpy(), regr_gt.squeeze(0).cpu().numpy(), thresh=0.99)
    
    # Display final image with bounding boxes
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(sample)
    plt.show()
    break  # Remove this break to process all items in the test_loader
