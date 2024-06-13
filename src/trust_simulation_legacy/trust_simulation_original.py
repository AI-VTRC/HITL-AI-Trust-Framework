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
from tensorflow.keras.applications.resnet_v2 import (
    ResNet101V2,
    preprocess_input,
    decode_predictions,
)
from cavs import ConnectedAutonomousVehicle

# Load the pre-trained multi-label classification model (original model from the paper)
model_classification = torch.hub.load(
    "facebookresearch/WSL-Images", "resnext101_32x48d_wsl"
)
model_classification.eval()  # Set the classification model to evaluation mode

# Load the pre-trained object detection model (original model was YOLOv3 from the paper)
model_object_detection = YOLO("yolov8n.pt")

# Define the trust threshold
trust_threshold = 0.5

# Define the trust propagation and trust fusion data structures
trust_recommendations = {}


def classify_image_original(image_path):
    """
    Function to process and classify an image using ResNet for scene classification
    """
    classification_model = ResNet101V2(weights="imagenet")

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


def classify_image(image_path, model_classification):
    """
    Function to process and classify an image using ResNet for scene classification
    Process and classify an image to predict the scene it represents using a given classification model.

    Parameters:
    - image_path (str): Path to the image file to be classified.
    - model_classification (torch.nn.Module): Pre-trained ResNet model for scene classification.

    Returns:
    - tuple (str, float): A tuple containing the top predicted class label and its corresponding confidence score.

    Raises:
    - ValueError: If the image can't be loaded or if its dimensions are invalid.
    - ConnectionError: If there's an issue fetching the class labels from the remote URL.

    Note:
    The function attempts to load class labels from a local file named 'imagenet-simple-labels.json'. If the file
    is not found, it fetches the labels from a URL.
    """

    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None or img.size == 0:
        raise ValueError(
            f"Failed to load image or invalid image dimensions at path: {image_path}"
        )

    preprocess_classification = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    original_image = img.copy()  # Make a copy for visualization
    original_image_rgb = cv2.cvtColor(
        original_image, cv2.COLOR_BGR2RGB
    )  # Convert image to RGB
    image = Image.fromarray(original_image_rgb)  # Convert NumPy array to PIL image
    image = preprocess_classification(image)
    image = image.unsqueeze(0)  # Add a batch dimension

    # Perform inference for image classification
    with torch.no_grad():
        outputs_classification = model_classification(image)

    # Try to Load class labels locally first, if not available fetch from URL
    try:
        with open("imagenet-simple-labels.json", "r") as f:
            labels = json.load(f)
    except FileNotFoundError:
        LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
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


def detect_objects(image_path):
    """
    Function to perform object detection using your specific object detection model
    Perform object detection on an image using a pre-defined object detection model (e.g., YOLO).

    Parameters:
    - image_path (str): Path to the image file on which object detection is to be performed.

    Returns:
    - list[dict]: A list of dictionaries, where each dictionary represents a detected object and contains:
        - 'label' (str): Name of the detected object.
        - 'confidence' (float): Confidence score of the detection.
        - 'box' (list[float]): Coordinates of the bounding box in the format [x1, y1, x2, y2].

    Note:
    The function relies on a globally-defined object detection model (`model_object_detection`) for predictions. Ensure
    that this model is properly initialized and loaded before calling this function.
    """
    # Perform object detection using YOLO
    results = model_object_detection(image_path)

    # Process YOLO predictions to extract object information
    detected_objects = []

    # Iterate over the results
    for result in results:
        # Extracting labels, confidences, and boxes
        for box in result.boxes:
            label_index = box.cls.item()  # Get class label as the index
            label_name = result.names[
                label_index
            ]  # Map index to the corresponding name

            output = {
                "label": label_name,  # Replace with the mapped name
                "confidence": box.conf.item(),  # Confidence score of the detection
                "box": box.xyxy.cpu().tolist(),  # Coordinates of the bounding box
            }
            detected_objects.append(output)  # Append each object inside the inner loop

    return detected_objects


def tuple_to_dict(trust_tuples, cav_names, obj_index):
    """Convert tuples to desired dictionary format"""
    obj_dict = {}

    # Get other objects except the current one
    other_objects = cav_names[:obj_index] + cav_names[obj_index + 1 :]

    for idx, other_obj in enumerate(other_objects):
        obj_dict[other_obj] = trust_tuples[obj_index][idx]

    return obj_dict


def create_cav_objects(num_cavs):
    """
    Creates a dictionary where each CAV has a tuple of trust scores for the other CAVs,
    ensuring each tuple sums to 1. Each CAV's trust scores are evenly distributed as much
    as possible, with adjustments for rounding to ensure the total sum for each tuple is 1.
    """
    # Initialize the dictionaries
    trust_scores_init = {}
    detected_objects_init = {f"cav{i+1}": [] for i in range(num_cavs)}

    # Iterate through each CAV to assign trust scores for the other CAVs
    for i in range(num_cavs):
        # Generate evenly distributed trust scores for n-1 CAVs
        scores = [1 / (num_cavs - 1) for _ in range(num_cavs - 1)]
        # Adjust the last element to ensure the sum is 1
        scores[-1] = 1 - sum(scores[:-1])
        # Round the scores to 2 decimal places and adjust for rounding errors if necessary
        scores = [round(score, 2) for score in scores]
        correction = 1.0 - sum(scores)
        scores[-1] += correction
        # Assign the scores to the current CAV
        trust_scores_init[f"cav{i+1}"] = tuple(scores)

    return trust_scores_init, detected_objects_init


def main():
    # List of CAV objects
    n_Agents = 4  # Number of CAVs that will connect together

    # Initialize trust values for connected agents
    trust_scores_init, detected_objects_init = create_cav_objects(n_Agents)

    # Set directory for initial Field of View capture for each of the 4 simulated CAVs
    os.chdir(r"Example/")
    image_paths = ["street_1.jpeg", "street_2.jpeg", "street_3.jpeg", "street_4.jpeg"]

    cavs = []
    for i in range(1, 5):
        cav = ConnectedAutonomousVehicle(
            name=f"cav{i}",
            fov=image_paths[i - 1],
            trust_scores=trust_scores_init[f"cav{i}"],
            detected_objects=detected_objects_init[f"cav{i}"],
            trust_threshold=trust_threshold,
            trust_recommendations=trust_recommendations,
        )
        trust_recommendations = cav.trust_recommendations
        cavs.append(cav)

    trust_scores_init = list(trust_scores_init.values())
    cav_names = [cav.name for cav in cavs]

    # Process FOVs for each CAV at the current time.
    for idx, cav in enumerate(cavs):
        print(f"Processing {cav.name}")

        image_path = image_paths[idx]

        cav.trust_scores = tuple_to_dict(trust_scores_init, cav_names, idx)

        # Object Detection
        cav.detected_objects = detect_objects(image_path)

        # Classify Image
        labels, confidences = classify_image(image_path, model_classification)
        cav.shared_info = {"scene_label": labels, "confidence": confidences}

    # Update each CAVs trust scores for each other based on the current shared information.
    for idx, cav in enumerate(cavs):
        for other_cav in cavs:
            if cav.name != other_cav.name:
                # Update Trust Scores with Assess Trust Function
                # ERROR HERE. THE ORIGINAL CAV TRUST VALUES ARE BEING RESET TO NOTHING
                new_trust_score = cav.assess_trust(other_cav.name)
                if (
                    new_trust_score is not None
                ):  # Assuming assess_trust returns None if no update is needed
                    cav.trust_scores[other_cav.name] = new_trust_score
                # cav trust scores after this get reset to empty.

                cav.share_info(other_cav)

        print("")
        print(f"Trust Scores for {cav.name} are {cav.trust_scores}")
        print(f"FOV Detected Objects for {cav.name} are {cav.detected_objects}")
        print(f"FOV Scene Description for {cav.name} are {cav.shared_info}")

    # Print Final Trust Recommendations
    print("Final Trust Recommendations:")
    print(json.dumps(trust_recommendations, indent=4))


if __name__ == "__main__":
    main()
