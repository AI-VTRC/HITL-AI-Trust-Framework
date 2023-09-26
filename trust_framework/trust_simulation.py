import cv2
import os
import json
import numpy as np
import random
import torch
import requests
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO
from tensorflow.keras.applications.resnet_v2 import ResNet101V2, preprocess_input, decode_predictions

# Load the pre-trained multi-label classification model (original model from the paper)
model_classification = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
model_classification.eval()  # Set the classification model to evaluation mode

# Load the pre-trained object detection model (original model was YOLOv3 from the paper)
model_object_detection = YOLO('yolov8n.pt')

# Define the trust threshold
trust_threshold = 0.5

# Define the trust propagation and trust fusion data structures
trust_recommendations = {}


# Function to calculate overlap between two bounding boxes
def calculate_overlap(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    top = max(y1, y2)
    bottom = min(y1 + h1, y2 + h2)

    if left < right and top < bottom:
        intersection_area = (right - left) * (bottom - top)
        area1 = w1 * h1
        area2 = w2 * h2
        return intersection_area / min(area1, area2)
    else:
        return 0.0


def are_objects_consistent(obj_1, obj_2):
    # Define logic to check consistency between two objects (e.g., based on location and type)
    # This is a simplified example; replace it with actual consistency checks
    location_tolerance = 10  # Tolerance for location consistency check (in pixels)

    # Check if the object labels are the same
    if obj_1['label'] == obj_2['label']:
        # Check location consistency (e.g., within a certain tolerance)
        x1, y1 = obj_1['location']
        x2, y2 = obj_2['location']
        if abs(x1 - x2) <= location_tolerance and abs(y1 - y2) <= location_tolerance:
            # Additional checks for consistency if needed
            # Replace with specific consistency checks

            # If all checks pass, consider the objects consistent
            return True

    # Objects are not consistent
    return False


# Simulate trust assessments for each CAV
def assess_trust(image_path, previous_trust_score, trust_scores, cav_name):
    # Simulate trust assessment based on the DC trust model
    # In this simplified example, we update trust based on received evidence and aij constant
    # Replace this logic with specific trust assessment rules

    # Generate random evidence counts (positive, negative, uncertain)
    positive_evidence = random.randint(0, 10)
    negative_evidence = random.randint(0, 10)
    uncertain_evidence = random.randint(0, 10)

    # Constants (aij) representing prior opinions (for example, based on hearsay)
    aij = random.uniform(0, 1)

    # Trust assessment logic
    alpha_ij = positive_evidence + aij
    beta_ij = negative_evidence + aij
    gamma_ij = uncertain_evidence + aij

    # Normalize the values to ensure they sum to 1
    total_evidence = alpha_ij + beta_ij + gamma_ij
    alpha_ij /= total_evidence
    beta_ij /= total_evidence
    gamma_ij /= total_evidence

    # Combine the updated trust values with previous trust (trust fusion)
    updated_trust_score = (
        alpha_ij + previous_trust_score[0],
        beta_ij + previous_trust_score[1],
        gamma_ij + previous_trust_score[2]
    )

    # Check trust with other CAVs and update trust accordingly
    for other_cav_name, trust_score in trust_scores.items():
        if other_cav_name != cav_name:
            trust_score_a = updated_trust_score
            trust_score_b = trust_score
            if trust_score_a[0] < 0.5 and trust_score_b[0] >= 0.5 and trust_score_b[0] < 1.0:
                updated_trust_score = (0.6, 0.2, 0.2)  # Set trust_score_a[0] to a higher value to trust the other CAV

    return updated_trust_score


# Function to process and classify an image using ResNet for scene classification
def classify_image_original(image_path):
    classification_model = ResNet101V2(weights='imagenet')

    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to the input size expected by ResNet101
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess the image

    # Perform scene classification
    predictions = classification_model.predict(img)
    decoded_predictions = decode_predictions(predictions, top=1)[0]

    # Return the top scene classification result and confidence
    # Outputs: ('fountain', 0.23353487)
    return decoded_predictions[0][1], decoded_predictions[0][2]


# Function to process and classify an image using ResNet for scene classification
def classify_image(image_path, model_classification):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None or img.size == 0:
        raise ValueError(f"Failed to load image or invalid image dimensions at path: {image_path}")

    preprocess_classification = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    original_image = img.copy()  # Make a copy for visualization
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert image to RGB
    image = Image.fromarray(original_image_rgb)  # Convert NumPy array to PIL image
    image = preprocess_classification(image)
    image = image.unsqueeze(0)  # Add a batch dimension

    # Perform inference for image classification
    with torch.no_grad():
        outputs_classification = model_classification(image)

    # Try to Load class labels locally first, if not available fetch from URL
    try:
        with open('imagenet-simple-labels.json', 'r') as f:
            labels = json.load(f)
    except FileNotFoundError:
        LABELS_URL = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
        response = requests.get(LABELS_URL)
        if response.status_code != 200:
            raise ConnectionError(f"Failed to fetch labels from {LABELS_URL}")
        labels = response.json()

    # Get the predicted class index and label
    _, predicted_idx_classification = torch.max(outputs_classification, 1)

    # Output the top 20 labels and their confidences
    top_k = 20
    top_confidences, top_indices = torch.topk(outputs_classification, top_k, 1)

    labels_result = [labels[idx] for idx in top_indices[0]]
    confidences = [conf.item() for conf in top_confidences[0]]

    scene_classification = {}
    for i in range(len(labels_result)):
        scene_classification[labels_result[i]] = confidences[i]

    # Return the top scene classification results (top_confidences and top_indices)
    first_key = list(scene_classification.keys())[0]
    first_value = scene_classification[first_key]

    return first_key, first_value


# Function to perform object detection using your specific object detection model
def detect_objects(image_path):
    # Perform object detection using YOLO
    results = model_object_detection(image_path)

    # Process YOLO predictions to extract object information
    detected_objects = []

    # Iterate over the results
    for result in results:
        # Extracting labels, confidences, and boxes
        for box in result.boxes:
            label_index = box.cls.item()  # Get class label as the index
            label_name = result.names[label_index]  # Map index to the corresponding name

            output = {
                'label': label_name,  # Replace with the mapped name
                'confidence': box.conf.item(),  # Confidence score of the detection
                'box': box.xyxy.cpu().tolist()  # Coordinates of the bounding box
            }
            detected_objects.append(output)  # Append each object inside the inner loop

    return detected_objects


# Class definition for CAV
class ConnectedAutonomousVehicle:
    def __init__(self, name, trust_scores, detected_objects=None):
        self.name = name
        self.trust_scores = trust_scores if trust_scores else {}
        self.detected_objects = detected_objects if detected_objects else []
        self.shared_info = {}

    def assess_trust(self, cav_name):
        # Simulate trust assessment based on the DC trust model
        # In this simplified example, we update trust based on received evidence and aij constant
        # Replace this logic with specific trust assessment rules

        # Generate random evidence counts (positive, negative, uncertain)
        positive_evidence = random.randint(0, 10)
        negative_evidence = random.randint(0, 10)
        uncertain_evidence = random.randint(0, 10)

        # Constants (aij) representing prior opinions (for example, based on hearsay)
        aij = random.uniform(0, 1)  # Replace with your own values

        # Trust assessment logic
        alpha_ij = positive_evidence + aij
        beta_ij = negative_evidence + aij
        gamma_ij = uncertain_evidence + aij

        # Normalize the values to ensure they sum to 1
        total_evidence = alpha_ij + beta_ij + gamma_ij
        alpha_ij /= total_evidence
        beta_ij /= total_evidence
        gamma_ij /= total_evidence

        # Combine the updated trust values with previous trust (trust fusion)
        updated_trust_score = (
            alpha_ij + self.trust_scores[0],
            beta_ij + self.trust_scores[1],
            gamma_ij + self.trust_scores[2]
        )

        # Check trust with other CAVs and update trust accordingly
        if isinstance(self.trust_scores, dict):
            for other_cav_name, trust_score in self.trust_scores.items():
                if other_cav_name != cav_name:
                    trust_score_a = updated_trust_score
                    trust_score_b = trust_score
                    if trust_score_a[0] < trust_threshold and trust_score_b[0] >= trust_threshold and trust_score_b[
                        0] < 1.0:
                        updated_trust_score = (
                            0.6, 0.2, 0.2)  # Set trust_score_a[0] to a higher value to trust the other CAV
        else:
            print(f"Expected a dict, but got {type(self.trust_scores)}: {self.trust_scores}")

        # this should be a single numeric value, not a tuple
        return updated_trust_score

    def share_info(self, other_cav, field_of_view):
        # Simulate capturing an image (Should replace this with capturing a real image)
        image_path = field_of_view  # Replace with the actual path to the image

        # Simulate scene classification
        scene_label, confidence = classify_image(image_path, model_classification)
        shared_info = {'scene_label': scene_label, 'confidence': confidence}

        # Simulate object detection (replace with your object detection logic)
        detected_objects = detect_objects(image_path)
        shared_info['detected_objects'] = detected_objects

        # Simulate information reception by other CAV and trust assessment
        received_info = shared_info  # In this simplified example, assume the information is received as shared
        received_scene_label = received_info['scene_label']
        received_confidence = received_info['confidence']

        # Assess trust and update trust scores
        self.trust_scores[other_cav.name] = self.assess_trust(other_cav.name)

        # Calculate overlap between FOVs (replace with your FOV logic)
        overlap = calculate_overlap(image_path, other_cav.image_path)

        # Check if there is overlap between FOVs
        if overlap > 0.0:
            # Check consistency of objects detected by both CAVs
            objects_detected_by_current_cav = self.detected_objects
            objects_detected_by_other_cav = received_info.get('detected_objects', [])

            consistent_objects = []
            for obj_1 in objects_detected_by_current_cav:
                for obj_2 in objects_detected_by_other_cav:
                    if obj_1['label'] == obj_2['label']:
                        # Check consistency based on object attributes (e.g., location, type)
                        if are_objects_consistent(obj_1, obj_2):
                            consistent_objects.append(obj_1)
                            break

            # Record consistent objects detected by both CAVs
            self.detected_objects += consistent_objects

            # Update trust recommendations based on trust assessment
            if self.name not in trust_recommendations:
                trust_recommendations[self.name] = {}
            trust_recommendations[self.name][other_cav.name] = self.trust_scores[other_cav.name]

            # Print objects and confidences between the two CAV images
            print(f"Overlap detected between {self.name} and {other_cav.name}.")
            print(f"Detected objects by {self.name}:")
            for obj in self.detected_objects:
                print(obj)
            print(f"Detected objects by {other_cav.name}:")
            for obj in other_cav.detected_objects:
                print(obj)
        else:
            # No FOV overlap, recommend trust to other CAV
            if self.name not in trust_recommendations:
                trust_recommendations[self.name] = {}
            trust_recommendations[self.name][other_cav.name] = self.trust_scores[other_cav.name]


def main():
    # List of CAV objects
    trust_scores_init = {f'cav{i}': (0.33, 0.33, 0.34) for i in range(1, 5)}
    detected_objects_init = {f'cav{i}': [] for i in range(1, 5)}

    cavs = [
        ConnectedAutonomousVehicle(
            name=f'cav{i}',
            trust_scores=trust_scores_init[f'cav{i}'],
            detected_objects=detected_objects_init[f'cav{i}']
        ) for i in range(1, 5)
    ]

    # Set directory for initial Field of View capture for each of the 4 simulated CAVs
    os.chdir(r'trust_framework/school_data/street/')
    image_paths = [
        'street_1.jpeg',
        'street_2.jpeg',
        'street_3.jpeg',
        'street_4.jpeg'
    ]

    # Process images and assess trust for each CAV
    for idx, cav in enumerate(cavs):
        print(f"Processing {cav.name}")

        image_path = image_paths[idx]

        # Update Trust Scores with Assess Trust Function
        cav.trust_scores = assess_trust(
            image_path,
            cav.trust_scores,
            trust_scores_init,
            cav.name
        )

        # Object Detection
        cav.detected_objects = detect_objects(image_path)

        # Classify Image
        labels, confidences = classify_image(image_path, model_classification)

        # Sharing information with other CAVs in the network
        for other_cav in cavs:
            if cav.name != other_cav.name:
                cav.share_info(other_cav, image_path)

        print(f"Trust Scores for {cav.name} are {cav.trust_scores}")
        print(f"Detected Objects by {cav.name} are {cav.detected_objects}")
        print(f"Classified Labels for {cav.name} are {labels} with confidences {confidences}")

    # Print Final Trust Recommendations
    print("Final Trust Recommendations:")
    print(json.dumps(trust_recommendations, indent=4))


if __name__ == "__main__":
    main()
