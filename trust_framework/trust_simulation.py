import cv2
import numpy as np
import os
import random
import tensorflow as tf  # Assuming TensorFlow is used for ResNet101
from tensorflow.keras.applications.resnet_v2 import ResNet101V2, preprocess_input, decode_predictions
from PIL import Image

# Load the pre-trained ResNet101 model
model = ResNet101V2(weights='imagenet')


# Function to calculate overlap between two bounding boxes
def calculate_overlap(image_path1, image_path2):
    # Open the two images
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Get the bounding boxes of the images
    bbox1 = image1.getbbox()
    bbox2 = image2.getbbox()

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
    # Define your logic to check consistency between two objects (e.g., based on location and type)
    # This is a simplified example; replace it with your actual consistency checks
    location_tolerance = 10  # Tolerance for location consistency check (in pixels)

    # Check if the object labels are the same
    if obj_1['label'] == obj_2['label']:
        # Check location consistency (e.g., within a certain tolerance)
        x1, y1 = obj_1['location']
        x2, y2 = obj_2['location']
        if abs(x1 - x2) <= location_tolerance and abs(y1 - y2) <= location_tolerance:
            # Additional checks for consistency if needed
            # Replace this with your specific consistency checks

            # If all checks pass, consider the objects consistent
            return True

    # Objects are not consistent
    return False


# Simulate trust assessments for each CAV
def assess_trust(image_path, previous_trust_score, trust_scores, cav_name):
    # Simulate trust assessment based on the DC trust model
    # In this simplified example, we update trust based on received evidence and aij constant
    # Replace this logic with your specific trust assessment rules

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


# Function to process and classify an image
def classify_image(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to the input size expected by ResNet101
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess the image

    # Perform scene classification
    predictions = model.predict(img)
    decoded_predictions = decode_predictions(predictions, top=1)[0]

    # Return the top scene classification result and confidence
    return decoded_predictions[0][1], decoded_predictions[0][2]


# Simulate processing images over time for 4 CAVs
num_cavs = 4
os.chdir(r'trust_framework/school_data/street/')
image_paths = ['street_1.jpeg', 'street_2.jpeg', 'street_3.jpeg', 'street_4.jpeg']

# Initialize trust scores for each CAV (use a dictionary)
trust_scores = {f'cav{i}': (0.33, 0.33, 0.34) for i in range(1, num_cavs + 1)}

# Initialize objects detected by each CAV (use a dictionary)
detected_objects = {cav: [] for cav in trust_scores.keys()}

# Process images and assess trust for each CAV
index = 0
for cav_name, image_path in zip(trust_scores.keys(), image_paths):
    scene_label, confidence = classify_image(image_path)

    # Share information (scene classification) with other CAVs
    for other_cav_name in trust_scores.keys():
        if other_cav_name != cav_name:
            # Simulate information sharing
            shared_info = {'scene_label': scene_label, 'confidence': confidence}

            # Simulate information reception by other CAVs and trust assessment
            received_info = shared_info  # In this simplified example, assume the information is received as shared
            received_scene_label = received_info['scene_label']
            received_confidence = received_info['confidence']

            # Assess trust and update trust scores
            trust_scores[other_cav_name] = assess_trust(image_path, trust_scores[other_cav_name], trust_scores,
                                                        cav_name)

            # Print trust score between CAVs
            print(f"Trust score between {cav_name} and {other_cav_name}: {trust_scores[other_cav_name]}")

            # Check if the trust is high enough to record objects
            if trust_scores[cav_name][0] > 0.5 and trust_scores[other_cav_name][0] > 0.5:
                # Check if there is an image to calculate overlap with
                if index < len(image_paths):
                    # Calculate overlap between FOVs (replace with your FOV logic)
                    overlap = calculate_overlap(image_path, image_paths[index])  # Corrected the overlap calculation

                    # Check consistency and record objects based on overlap
                    if overlap > 0.0:
                        # Check consistency of objects detected by both CAVs
                        objects_detected_by_current_cav = received_info.get('detected_objects', [])

                        # Print detected objects with confidence
                        if objects_detected_by_current_cav:
                            print(f"Detected objects with confidence between {cav_name} and {other_cav_name}:")
                            for obj in objects_detected_by_current_cav:
                                print(f'Object: {obj["label"]}, Confidence: {obj["confidence"]}')

                        # Record consistent objects detected by the current CAV
                        detected_objects[cav_name] += objects_detected_by_current_cav

                    else:
                        # No FOV overlap, record all objects detected by the other CAV
                        detected_objects[cav_name] += received_info.get('detected_objects', [])

            # Increment the index
            index += 1

################################################################  Objects still not being printed out
# Print detected objects for each CAV
for cav_name, objects in detected_objects.items():
    print(f'Objects detected by CAV {cav_name}:')
    for obj in objects:
        print(f'Object: {obj["label"]}, Confidence: {obj["confidence"]}')
