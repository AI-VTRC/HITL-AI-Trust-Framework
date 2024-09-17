import time
import os
import json
import torch
from itertools import product
from ultralytics import YOLO
from datetime import datetime
import cv2

from hitl_user import User
from hitl_cavs import ConnectedAutonomousVehicle
from hitl_utils import classify_image, detect_objects, create_cav_objects, tuple_to_dict, format_to_bullets, get_image_count

# Define possible values for trust configuration
trust_levels = ['Cautious', 'Moderate', 'Trusting']
requires_trust_history_options = [True, False]
trust_frames_required_options = [0, 3, 5, 10]  # Example trust frame values

def run_experience(folder, user_configurations):
    """Main Driver"""
    # Load the pre-trained multi-label classification model and freeze parameters
    model_classification = torch.hub.load(
        "facebookresearch/WSL-Images", "resnext101_32x48d_wsl"
    )
    model_classification.eval()

    # Load the pre-trained object detection model
    model_object_detection = YOLO("yolov8n.pt")

    # Define the trust propagation and trust fusion data structures
    trust_recommendations = {}

    root_connection = os.path.abspath("assets/data") + os.sep + folder
    num_cars = 4

    # Determine the number of images by checking the first car's directory
    example_car_folder = os.path.join(root_connection, "Car1")
    num_images = get_image_count(example_car_folder)

    # Initialize trust values for connected agents
    trust_scores_init, detected_objects_init = create_cav_objects(num_cars)

    # Initialize Users from configurations
    users = [User(user_id=config['id'], name=config['name'],
                  trust_level=config['trust_level'],
                  requires_trust_history=config['requires_trust_history'],
                  trust_frames_required=config['trust_frames_required'])
             for config in user_configurations]

    # Initialize CAVs and link each with corresponding User
    image_paths = [os.path.join(root_connection, f"Car{i}", "frame_1.jpg") for i in range(1, len(users) + 1)]
    cavs = []

    # Initialize CAVs with corresponding User settings
    for i, user in enumerate(users):
        cavs.append(ConnectedAutonomousVehicle(
            name=f"cav{i + 1}",
            detected_objects=detected_objects_init[f"cav{i + 1}"],
            trust_scores=trust_scores_init[f"cav{i + 1}"],
            user=user  # Directly pass the user object if it contains all necessary settings and trackers
        ))

    trust_scores_init = list(trust_scores_init.values())
    cav_names = [cav.name for cav in cavs]

    # Initialize logging dictionary with human trust level logging
    log_data = {}

    # Initialize trust score tracking and log for detected objects
    for cav in cavs:
        log_data[cav.name] = {
            "human_trust_level": cav.user.trust_level if cav.user else None,  # Log human trust level if available
            "trust_scores": {other_cav.name: [] for other_cav in cavs if other_cav.name != cav.name},
            "detected_objects": []
        }

    # Create results directory
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"results/{folder}/{current_datetime}"
    os.makedirs(results_dir, exist_ok=True)

    # Process FOVs for each CAV at the current time
    for idx, cav in enumerate(cavs):
        print(f"Processing FOVs for {cav.name}")

        image_path = image_paths[idx]
        cav.trust_scores = tuple_to_dict(trust_scores_init, cav_names, idx)

        # Object Detection
        cav.detected_objects, img_with_boxes = detect_objects(image_path, model_object_detection)

        # Save the image with bounding boxes and labels
        img_output_path = os.path.join(results_dir, f"Car{idx + 1}_frame_1_with_boxes.jpg")
        cv2.imwrite(img_output_path, img_with_boxes)

        # Classify Image
        labels, confidences = classify_image(image_path, model_classification)
        cav.shared_info = {"scene_label": labels, "confidence": confidences}

        # Log initial detected objects
        log_data[cav.name]["detected_objects"].append(
            {
                "frame": 1,
                "objects": cav.detected_objects,
                "scene_label": labels,
                "confidence": confidences
            }
        )

    # Loop through the remaining images for each CAV
    for image_index in range(2, num_images + 1):  # Use num_images for accurate count
        for i, cav in enumerate(cavs):
            # Construct the path for the current image
            image_path = os.path.join(root_connection, f"Car{i + 1}", f"frame_{image_index}.jpg")
            print(f"Processing {cav.name} with image {image_path}")

            # Update CAV's fov to the current image
            cav.fov = image_path
            cav.detected_objects, img_with_boxes = detect_objects(image_path, model_object_detection)

            # Save the image with bounding boxes and labels
            img_output_path = os.path.join(results_dir, f"Car{i + 1}_frame_{image_index}_with_boxes.jpg")
            cv2.imwrite(img_output_path, img_with_boxes)

            labels, confidences = classify_image(image_path, model_classification)
            cav.shared_info = {"scene_label": labels, "confidence": confidences}

            # Update and log trust scores
            for other_cav in cavs:
                if cav.name != other_cav.name:
                    user = users[cavs.index(cav)]
                    cav.share_info(other_cav, user)
                    new_trust_score = cav.assess_trust(other_cav.name, user.name)
                    if new_trust_score is not None:
                        cav.trust_scores[other_cav.name] = new_trust_score
                        log_data[cav.name]["trust_scores"][other_cav.name].append(new_trust_score)

            # Log detected objects for each frame
            log_data[cav.name]["detected_objects"].append(
                {
                    "frame": image_index,
                    "objects": cav.detected_objects,
                    "scene_label": labels,
                    "confidence": confidences
                }
            )

        print(f"Trust Scores after processing image {image_index}:")
        print(json.dumps({cav.name: cav.trust_scores for cav in cavs}, indent=4))

    # Save the log data to a file
    print("________________________________")
    print("Final Trust Recommendations:")
    filename = os.path.join(results_dir, f"{folder}_{current_datetime}_trust_log.json")
    with open(filename, "w") as file:
        json.dump(log_data, file, indent=4)

    print(json.dumps(trust_recommendations, indent=4))
    formatted_data = format_to_bullets(trust_recommendations)
    result_filename = os.path.join(results_dir, f"{folder}_{current_datetime}.txt")
    with open(result_filename, "w") as file:
        file.write(formatted_data)


def main():
    counter = 0
    
    # Generate all possible combinations of trust settings
    for trust_level, requires_trust_history, trust_frames_required in product(trust_levels, requires_trust_history_options, trust_frames_required_options):
        # Define the user configurations with varying trust settings
        user_configurations = [
            {'id': 1, 'name': 'User1', 'trust_level': trust_level, 'requires_trust_history': requires_trust_history, 'trust_frames_required': trust_frames_required},
            {'id': 2, 'name': 'User2', 'trust_level': trust_level, 'requires_trust_history': requires_trust_history, 'trust_frames_required': trust_frames_required},
            {'id': 3, 'name': 'User3', 'trust_level': trust_level, 'requires_trust_history': requires_trust_history, 'trust_frames_required': trust_frames_required},
            {'id': 4, 'name': 'User4', 'trust_level': trust_level, 'requires_trust_history': requires_trust_history, 'trust_frames_required': trust_frames_required}
        ]

        # Run the experience for this configuration
        folder_name = f"8_29_24_scenario_1"
        print(f"Running experience for: {folder_name}")
        run_experience(folder=folder_name, user_configurations=user_configurations)
        time.sleep(10)  # Pause between runs if necessary
        
        counter += 1
        
        # Can run all combination later
        if counter == 3:
            break


if __name__ == "__main__":
    main()
