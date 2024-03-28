from builtins import round, range, sum, ValueError, FileNotFoundError, ConnectionError, len, list, float, max, min, \
    enumerate, print, any

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
def calculate_overlap(bboxes1, bboxes2):
    """
       Calculate the overlap between lists of bounding boxes. The overlap for each bounding box
       in bboxes1 with every bounding box in bboxes2 is computed and returned as a list.

       Parameters:
       - bboxes1 (list of tuples): A list of bounding boxes, where each bounding box is represented
                                   as a tuple in the format (x1, y1, x2, y2).
       - bboxes2 (list of tuples): A list of bounding boxes, where each bounding box is represented
                                   as a tuple in the format (x1, y1, x2, y2).

       Returns:
       - list of floats: A list containing the overlap ratios. Each overlap ratio is the ratio of the
                         intersection area to the smaller area of the two bounding boxes. If there's no
                         overlap, the ratio is 0.0.
    """

    overlaps = []
    for bbox1 in bboxes1:
        if not bboxes2:
            overlap = 0.0
            overlaps.append(overlap)
            break

        for bbox2 in bboxes2:
            x1, y1, x2, y2 = bbox1[0]  # Unpack the coordinates of bbox1
            x3, y3, x4, y4 = bbox2[0]  # Unpack the coordinates of bbox2

            left = max(x1, x3)
            right = min(x2, x4)
            top = max(y1, y3)
            bottom = min(y2, y4)

            if left < right and top < bottom:
                intersection_area = (right - left) * (bottom - top)
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (x4 - x3) * (y4 - y3)
                overlap = intersection_area / min(area1, area2)
            else:
                overlap = 0.0

            overlaps.append(overlap)

    return overlaps


def compute_iou(boxA, boxB):
    """
       Compute the Intersection over Union (IoU) between two bounding boxes.

       The IoU metric measures the overlap between two bounding boxes. It's the area of the intersection of the boxes
       divided by the area of the union of the boxes. The resulting value is between 0 (no overlap) and 1 (perfect overlap).

       Parameters:
       - boxA (list): A list containing the coordinates of the first bounding box in the format [x1, y1, x2, y2],
                      where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
       - boxB (list): A list containing the coordinates of the second bounding box in the same format as boxA.

       Returns:
       - float: The IoU value between the two bounding boxes.

       Note:
       The boxes are passed as lists containing a single list of coordinates. Only the first element ([0]) is considered.
    """

    # Determine the coordinates of the intersection rectangle
    boxA = boxA[0]
    boxB = boxB[0]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of the intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # Return the IoU
    return iou


def are_objects_consistent(objA, objB, iou_threshold=0.5):
    """
    Compare the consistency of two objects using their class and bounding box.

    Parameters:
    - objA: a dictionary representing an object with keys 'label' and 'box'
    - objB: a dictionary representing an object with keys 'label' and 'box'
    - iou_threshold: the threshold for the IoU to consider objects consistent

    Returns:
    - True if objects are consistent, otherwise False.
    """
    if objA['label'] != objB['label']:
        return False

    iou = compute_iou(objA['box'], objB['box'])

    return iou >= iou_threshold


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
    """
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
        with os.open('imagenet-simple-labels.json', 'r') as f:
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
    """
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
            label_name = result.names[label_index]  # Map index to the corresponding name

            output = {
                'label': label_name,  # Replace with the mapped name
                'confidence': box.conf.item(),  # Confidence score of the detection
                'box': box.xyxy.cpu().tolist()  # Coordinates of the bounding box
            }
            detected_objects.append(output)  # Append each object inside the inner loop

    return detected_objects


# Convert tuples to desired dictionary format
def tuple_to_dict(trust_tuples, cav_names, obj_index):
    obj_dict = {}

    # Get other objects except the current one
    other_objects = cav_names[:obj_index] + cav_names[obj_index + 1:]

    for idx, other_obj in enumerate(other_objects):
        obj_dict[other_obj] = trust_tuples[obj_index][idx]

    return obj_dict


def create_cav_objects(num_cavs):
    """
    Modifies the function to create a dictionary where each CAV has a tuple of three trust scores,
    evenly distributed as much as possible with adjustments for rounding to ensure the total
    sum of each tuple is as close to 1 as possible and evenly distributed across CAVs.
    """
    # Calculate base trust scores for three parts
    base_trust = round(1.0 / 3, 2)  # Base trust score for each part

    # Calculate corrections for rounding issues to ensure the sum of three parts is as close to 1 as possible
    correction = round(1.0 - (base_trust * 3), 2)

    # Apply corrections to distribute the rounding error across the three parts
    if correction == 0.01:
        trust_scores_tuple = (base_trust, base_trust, base_trust + correction)
    elif correction == 0.02:
        trust_scores_tuple = (base_trust, base_trust + 0.01, base_trust + 0.01)
    else:
        trust_scores_tuple = (base_trust, base_trust, base_trust)  # No correction needed

    # Create dictionaries for trust scores and detected objects
    trust_scores_init = {f'cav{i + 1}': trust_scores_tuple for i in range(num_cavs)}
    detected_objects_init = {f'cav{i + 1}': [] for i in range(num_cavs)}

    return trust_scores_init, detected_objects_init


# Class definition for CAV
class ConnectedAutonomousVehicle:
    """
        Represents a Connected Autonomous Vehicle (CAV) and its operations related to object detection, trust assessment,
        and information sharing with other CAVs.

        Attributes:
        - name (str): Unique identifier for the CAV.
        - fov (str): Field of View for the CAV.
        - trust_scores (dict): Dictionary containing trust scores for other CAVs.
        - detected_objects (list): List of objects detected by the CAV.
        - shared_info (dict): Information that the CAV chooses to share with others.
    """

    def __init__(self, name, fov, trust_scores, detected_objects=None):
        self.name = name
        self.fov = fov
        self.trust_scores = trust_scores if trust_scores else {}
        self.detected_objects = detected_objects if detected_objects else []
        self.shared_info = {}

    def assess_trust(self, cav_name):
        """
            Assess the trust score for a specific CAV based on the DC trust model.

            Parameters:
                - cav_name (str): The name of the CAV whose trust is being assessed.

            Returns:
                - float: Updated trust score for the given CAV.

            Note:
            In this implementation, a simplified trust assessment model is used which might not reflect
            real-world complexities.
        """
        # Simulate trust assessment based on the DC trust model
        # In this simplified example, we update trust based on received evidence and aij constant
        # Replace this logic with specific trust assessment rules
        if cav_name == self.name:  # or cav_name in self.trust_scores
            return {}

        # Generate random evidence counts (positive, negative, uncertain)
        positive_evidence = random.randint(0, 10)
        negative_evidence = random.randint(0, 10)
        uncertain_evidence = random.randint(0, 10)

        # Constants (aij) representing prior opinions (for example, based on hearsay)
        aij = random.uniform(0, 1)

        # Trust assessment logic
        alpha_ij = positive_evidence + aij * 10  # considering aij as a weight for pseudo count
        beta_ij = negative_evidence + (1 - aij) * 10  # considering (1-aij) as a weight for pseudo count
        gamma_ij = uncertain_evidence

        # Computing the trust value omega_ij as the expected value
        total_count = alpha_ij + beta_ij + gamma_ij
        omega_ij = alpha_ij / total_count  # Expected trust value (Trustworthiness)

        # Trust fusion with other CAVs in the system
        for other_cav_name, trust_score in self.trust_scores.items():
            if other_cav_name != cav_name:
                trust_score_a = omega_ij
                trust_score_b = trust_score
                if trust_score_a < trust_threshold and trust_threshold <= trust_score_b < 1.0:
                    omega_ij = 0.6  # Set to a higher value to trust the other CAV

        # Updating the trust score in the trust_scores dictionary
        self.trust_scores[cav_name] = omega_ij

        return omega_ij

    def share_info(self, other_cav):
        """
            Simulate sharing of detected objects and scene information with another CAV.

            Parameters:
                - other_cav (ConnectedAutonomousVehicle): The other CAV to share information with.

            Note:
            This function will also simulate the trust assessment and updating of the trust scores based on the
            shared information.
        """

        # Simulate information reception by other CAV and trust assessment
        received_info = other_cav.shared_info
        received_scene_label = received_info['scene_label']
        received_confidence = received_info['confidence']

        # Assess trust and update trust scores
        self.trust_scores[other_cav.name] = self.assess_trust(other_cav.name)

        # Get a list of all bounding boxes detected in FOV1
        cav1_detections = []
        for i in range(0, len(self.detected_objects)):
            cav1_detections.append(self.detected_objects[i]["box"])

        # Get a list of all bounding boxes detected in FOV2
        cav2_detections = []
        for i in range(0, len(other_cav.detected_objects)):
            cav2_detections.append(other_cav.detected_objects[i]["box"])

        # Calculate overlap between two FOVs
        overlap = calculate_overlap(cav1_detections, cav2_detections)

        # Check if there is overlap between FOVs
        if any(x > 0.0 for x in overlap):
            # Check consistency of objects detected by both CAVs
            objects_detected_by_current_cav = self.detected_objects
            objects_detected_by_other_cav = other_cav.detected_objects

            # INCLUDE logic that compares shared info as well.
            consistent_objects = []
            for obj_1 in objects_detected_by_current_cav:
                for obj_2 in objects_detected_by_other_cav:
                    if obj_1['label'] == obj_2['label']:
                        # Check consistency based on object attributes (e.g., location, type)
                        if are_objects_consistent(obj_1, obj_2):
                            consistent_objects.append(obj_1)
                            break

            print(f"Overlap detected between {self.name} and {other_cav.name}.")
            print(f"Detected objects shared by {self.name} include:")
            for obj in consistent_objects:
                print(obj)

            # Record consistent objects detected by both CAVs
            self.detected_objects += consistent_objects

            # Print objects and confidences between the two CAV images
            print(f"Overlap detected between {self.name} and {other_cav.name}.")
            print(f"Detected objects by {self.name}:")
            for obj in self.detected_objects:
                print(obj)
            print(f"Detected objects by {other_cav.name}:")
            for obj in other_cav.detected_objects:
                print(obj)

            # Update trust recommendations based on trust assessment
            if self.name not in trust_recommendations:
                trust_recommendations[self.name] = {}

            trust_recommendations[self.name][other_cav.name] = self.trust_scores[other_cav.name]  # ADD BREAK POINT HERE

            # Compare shared_info
            if self.shared_info['scene_label'] == received_scene_label:
                if received_confidence > self.shared_info['confidence']:
                    # Increase the trust value for the other CAV based on some criteria (e.g., by 10%)
                    trust_increment = 0.10
                    trust_value = trust_recommendations[self.name][other_cav.name]  # ADD BREAK POINT HERE
                    trust_value += trust_increment  # ADD BREAK POINT HERE

                    # Ensure trust value doesn't exceed 1.0
                    trust_value = min(trust_value, 1.0)

                    trust_recommendations[self.name][other_cav.name] = trust_value

        else:
            # No FOV overlap, recommend trust to other CAV
            if self.name not in trust_recommendations:
                trust_recommendations[self.name] = {}
            trust_recommendations[self.name][other_cav.name] = self.trust_scores[other_cav.name]

        # Update self.trust_scores based on trust_recommendations
        for cav_name, recommended_trust in trust_recommendations[self.name].items():
            if cav_name != self.name:  # Exclude the original self
                self.trust_scores[cav_name] = recommended_trust

        return


def main():
    root_connection = r'F:\Matt\PhD_Research\AI_Trust_Framework\nuScenes_by_Motional\samples\Sub_Sample_1'
    n_Agents = 4  # Number of CAVs that will connect together

    # Initialize trust values for connected agents
    trust_scores_init, detected_objects_init = create_cav_objects(n_Agents)

    # Syncronize connection of perspectives per agent
    cav1_stream = os.listdir(root_connection + r'\Car1')
    cav2_stream = os.listdir(root_connection + r'\Car2')
    cav3_stream = os.listdir(root_connection + r'\Car3')
    cav4_stream = os.listdir(root_connection + r'\Car4')

    # Initialize the first set of images for each CAV
    first_image_paths = [os.path.join(root_connection, f'Car{i}', 'frame_1.jpg') for i in range(1, n_Agents + 1)]

    # Set directory for initial Field of View capture for each of the 4 simulated CAVs
    #os.chdir(r'Example/')
    #image_paths = [
    #    'street_1.jpeg',
    #    'street_2.jpeg',
    #    'street_3.jpeg',
    #    'street_4.jpeg'
    #]

    # Initialize CAVs with the first image
    cavs = [
        ConnectedAutonomousVehicle(
            name=f'cav{i}',
            fov=first_image_paths[i - 1],
            trust_scores=trust_scores_init[f'cav{i}'],
            detected_objects=detected_objects_init[f'cav{i}']
        ) for i in range(1, n_Agents + 1)
    ]

    trust_scores_init = list(trust_scores_init.values())
    cav_names = [cav.name for cav in cavs]

    # Process FOVs for each CAV at the current time.
    for idx, cav in enumerate(cavs):
        print(f"Processing {cav.name}")

        #image_path = image_paths[idx]
        image_path = first_image_paths[idx]

        cav.trust_scores = tuple_to_dict(trust_scores_init, cav_names, idx)

        # Object Detection
        cav.detected_objects = detect_objects(image_path)

        # Classify Image
        labels, confidences = classify_image(image_path, model_classification)
        cav.shared_info = {'scene_label': labels, 'confidence': confidences}

    # Update each CAVs trust scores for each other based on the current shared information.
    for idx, cav in enumerate(cavs):
        for other_cav in cavs:
            if cav.name != other_cav.name:
                # Update Trust Scores with Assess Trust Function
                # ERROR HERE. THE ORIGINAL CAV TRUST VALUES ARE BEING RESET TO NOTHING
                new_trust_score = cav.assess_trust(other_cav.name)
                if new_trust_score is not None:  # Assuming assess_trust returns None if no update is needed
                    cav.trust_scores[other_cav.name] = new_trust_score
                # cav trust scores after this get reset to empty.

                cav.share_info(other_cav)

        print("")
        print(f"Trust Scores for {cav.name} are {cav.trust_scores}")
        print(f"FOV Detected Objects for {cav.name} are {cav.detected_objects}")
        print(f"FOV Scene Description for {cav.name} are {cav.shared_info}")

    # NEW processing to check and refine
    # Loop through the remaining images for each CAV
    for image_index in range(2, 19):  # Assuming there are 18 images, starting from 2 since 1 was used for initialization
        for i, cav in enumerate(cavs):
            # Construct the path for the current image
            image_path = os.path.join(root_connection, f'Car{i + 1}', f'image{image_index}.jpg')
            print(f"Processing {cav.name} with image {image_path}")

            # Update CAV's fov to the current image
            cav.fov = image_path

            # Example processing steps, modify according to your actual functions
            cav.trust_scores = tuple_to_dict(trust_scores_init, [cav.name for cav in cavs], i)
            cav.detected_objects = detect_objects(image_path)
            labels, confidences = classify_image(image_path, model_classification)
            cav.shared_info = {'scene_label': labels, 'confidence': confidences}

    # Print Final Trust Recommendations
    print("Final Trust Recommendations:")
    print(json.dumps(trust_recommendations, indent=4))


if __name__ == "__main__":
    main()
