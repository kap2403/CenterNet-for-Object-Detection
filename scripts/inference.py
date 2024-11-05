import cv2
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from dataloader.dataset import Dataset
from models.centernet import centernet  
from torchvision import transforms
import torch.nn.functional as F

# Load configuration from config.json
with open("dataloader/config.json", "r") as config_file:
    config = json.load(config_file)

# Extract model parameters from config
MODEL_PATH = config["MODEL_PATH"]
MODEL_SCALE = config["MODEL_SCALE"]
input_size = config["input_size"]
batch_size = config["batch_size"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = centernet().to(device)
if MODEL_PATH:
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: The model file at '{MODEL_PATH}' was not found.")
else:
    print("Error: MODEL_PATH is empty in config.json.")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
])

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

def showbox(img, hm, regr, thresh=0.9):
    boxes, _ = pred2box(hm, regr, thresh=thresh)
    for box in boxes:
        # Draw rectangle on image
        cv2.rectangle(img, (int(box[0]), int(box[1] + box[3])),
                      (int(box[0] + box[2]), int(box[1])),
                      (220, 0, 0), 3)
    return img

# Video inference
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to tensor
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            hm, regr = model(img)

        hm = torch.sigmoid(hm).cpu().numpy().squeeze(0).squeeze(0)
        regr = regr.cpu().numpy().squeeze(0)

        # Apply threshold and show bounding boxes
        frame = showbox(frame, hm, regr, thresh=0.6)

        # Display the frame
        cv2.imshow('Inference Output', frame)

        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Image inference
def process_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        hm, regr = model(img_tensor)

    hm = torch.sigmoid(hm).cpu().numpy().squeeze(0).squeeze(0)
    regr = regr.cpu().numpy().squeeze(0)

    # Display bounding boxes on image
    img = np.array(img)
    img_with_boxes = showbox(img, hm, regr, thresh=0.6)

    # Show image with bounding boxes
    plt.figure(figsize=(8, 8))
    plt.imshow(img_with_boxes)
    plt.show()

# Main function to handle both image and video inputs
def main(input_path):
    if input_path.endswith('.mp4') or input_path.endswith('.avi'):
        print("Processing video...")
        process_video(input_path)
    else:
        print("Processing image...")
        process_image(input_path)

# Example usage:
input_path = "your_image_or_video_path_here"  # Path to your image or video file
main(input_path)
