import os
import requests
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO

# Load the pre-trained ResNeXt-101 32x48d model from torchvision for classification
model_classification = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
model_classification.eval()  # Set the classification model to evaluation mode

# Define preprocessing transforms to match the model's requirements
preprocess_classification = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess an image for classification
os.chdir(r'trust_framework/school_data/street/')
image_url = 'street_2.jpeg'
image = cv2.imread(image_url)
original_image = image.copy()  # Make a copy for visualization
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert image to RGB
image = Image.fromarray(original_image_rgb)  # Convert NumPy array to PIL image
image = preprocess_classification(image)
image = image.unsqueeze(0)  # Add a batch dimension

# Perform inference for image classification
with torch.no_grad():
    outputs_classification = model_classification(image)

# Load class labels for ImageNet
LABELS_URL = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
labels = requests.get(LABELS_URL).json()

# Get the predicted class index and label
_, predicted_idx_classification = torch.max(outputs_classification, 1)
predicted_label_classification = labels[predicted_idx_classification.item()]

# Output the top 20 labels and their confidences
top_k = 20
top_confidences, top_indices = torch.topk(outputs_classification, top_k, 1)

print("Top 20 Labels and Confidences from Classification:")
for i in range(top_k):
    label_idx = top_indices[0][i].item()
    label_confidence = top_confidences[0][i].item()
    label = labels[label_idx]
    print(f"Label: {label}, Confidence: {label_confidence:.4f}")

# Original Paper Used YOLOv3; instead using YOLOv8 for object detection
model = YOLO('yolov8n.pt')
results = model(image_url)  # results list
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpeg')  # save image
