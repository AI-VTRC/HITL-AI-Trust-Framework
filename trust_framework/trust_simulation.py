import cv2
import os
import numpy as np
import random
from PIL import Image

# Load the pre-trained ResNet101 model for scene classification
# Replace this with specific model import
from tensorflow.keras.applications.resnet_v2 import ResNet101V2, preprocess_input, decode_predictions

# Load the pre-trained object detection model (assuming it's a separate model)
# Replace this with specific object detection model import and initialization
# For example, use TensorFlow's Object Detection API or another object detection framework
object_detection_model = ResNet101V2(weights='imagenet')

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
def classify_image(image_path, object_detection_model):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to the input size expected by ResNet101
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess the image

    # Perform scene classification
    predictions = object_detection_model.predict(img)
    decoded_predictions = decode_predictions(predictions, top=1)[0]

    # Return the top scene classification result and confidence
    return decoded_predictions[0][1], decoded_predictions[0][2]


# Function to perform object detection using your specific object detection model
def detect_objects(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to the input size expected by ResNet101V2
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess the image

    # Perform object detection using ResNet101V2
    predictions = object_detection_model.predict(img)

    # Decode the object detection results and extract relevant information
    decoded_predictions = decode_predictions(predictions)

    # You should extract object labels, locations, and confidences from decoded_predictions
    # Example: [{'label': 'car', 'location': (x, y), 'confidence': 0.95}, ...]
    detected_objects = []  # Replace with your object detection results

    return detected_objects


# Class definition for CAV
class ConnectedAutonomousVehicle:
    def __init__(self, name, trust_scores, detected_objects):
        self.name = name
        self.trust_scores = trust_scores
        self.detected_objects = detected_objects
        self.image_path = None

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
        for other_cav_name, trust_score in self.trust_scores.items():
            if other_cav_name != cav_name:
                trust_score_a = updated_trust_score
                trust_score_b = trust_score
                if trust_score_a[0] < trust_threshold and trust_score_b[0] >= trust_threshold and trust_score_b[
                    0] < 1.0:
                    updated_trust_score = (
                    0.6, 0.2, 0.2)  # Set trust_score_a[0] to a higher value to trust the other CAV

        return updated_trust_score

    def share_info(self, other_cav):
        # Simulate capturing an image (Should replace this with capturing a real image)
        self.image_path = 'path_to_image_for_' + self.name  # Replace with the actual path to the image

        # Simulate scene classification
        scene_label, confidence = classify_image(self.image_path, object_detection_model)
        shared_info = {'scene_label': scene_label, 'confidence': confidence}

        # Simulate object detection (replace with your object detection logic)
        detected_objects = detect_objects(self.image_path)
        shared_info['detected_objects'] = detected_objects

        # Simulate information reception by other CAV and trust assessment
        received_info = shared_info  # In this simplified example, assume the information is received as shared
        received_scene_label = received_info['scene_label']
        received_confidence = received_info['confidence']

        # Assess trust and update trust scores
        self.trust_scores[other_cav.name] = self.assess_trust(other_cav.name)

        # Calculate overlap between FOVs (replace with your FOV logic)
        overlap = calculate_overlap(self.image_path, other_cav.image_path)

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


# Initialize trust scores for each CAV (use a dictionary)
trust_scores = {f'cav{i}': (0.33, 0.33, 0.34) for i in range(1, 5)}  # Replace with your trust scores

# Initialize objects detected by each CAV (use a dictionary)
detected_objects = {f'cav{i}': [] for i in range(1, 5)}  # Replace with your detected objects

# Initialize CAV objects
cav1 = ConnectedAutonomousVehicle(name='cav1', trust_scores=trust_scores['cav1'],
                                  detected_objects=detected_objects['cav1'])
cav2 = ConnectedAutonomousVehicle(name='cav2', trust_scores=trust_scores['cav2'],
                                  detected_objects=detected_objects['cav2'])
cav3 = ConnectedAutonomousVehicle(name='cav3', trust_scores=trust_scores['cav3'],
                                  detected_objects=detected_objects['cav3'])
cav4 = ConnectedAutonomousVehicle(name='cav4', trust_scores=trust_scores['cav4'],
                                  detected_objects=detected_objects['cav4'])

# List of CAV objects
cavs = [cav1, cav2, cav3, cav4]

# Simulate image paths for each CAV (replace with your image paths)
os.chdir(r'trust_framework/school_data/street/')
image_paths = ['street_1.jpeg',
               'street_2.jpeg',
               'street_3.jpeg',
               'street_4.jpeg']

# Process images and assess trust for each CAV
for i, cav in enumerate(cavs):
    # Share information with other CAVs
    for other_cav in cavs:
        if other_cav != cav:
            cav.share_info(other_cav)

    # Print trust scores for this CAV
    print(f"Trust scores for {cav.name}: {cav.trust_scores}")

# Print trust recommendations
print("Trust Recommendations:")
for cav_name, recommendations in trust_recommendations.items():
    print(f"{cav_name} recommends trusting:")
    for other_cav, trust_score in recommendations.items():
        print(f"{other_cav}: {trust_score}")
