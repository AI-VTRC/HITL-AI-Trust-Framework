import time
import os
import json
import torch
from ultralytics import YOLO
from datetime import datetime

from user import User
from cavs_HITL import ConnectedAutonomousVehicle
from utils import classify_image
from utils import detect_objects
from utils import create_cav_objects
from utils import tuple_to_dict
from utils import format_to_bullets
from utils import get_image_count

temp_dir = r'D:\HITL-AI-Trust-Framework\HITL_Trust_CV_App\temp'

# Define user configurations
user_configurations = [
    {'id': 1, 'name': 'User1', 'trust_level': 'Moderate', 'requires_trust_history': True, 'trust_frames_required': 5,
     'trust_monitor': True},
    {'id': 2, 'name': 'User2', 'trust_level': 'Cautious', 'requires_trust_history': True, 'trust_frames_required': 10},
    {'id': 3, 'name': 'User3', 'trust_level': 'Trusting', 'requires_trust_history': True, 'trust_frames_required': 3},
    {'id': 4, 'name': 'User4', 'trust_level': 'Moderate', 'requires_trust_history': False, 'trust_frames_required': 0}
]


def run_experience(folder):
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

    root_connection = r'D:\HITL-AI-Trust-Framework\src\assets\data' + os.sep + folder
    num_cars = 4

    # Assumes the number of images for each Car is the same
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

    users[0].trust_monitor = True

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

    # Process FOVs for each CAV at the current time.
    for idx, cav in enumerate(cavs):
        print(f"Processing FOVs for {cav.name}")

        image_path = image_paths[idx]

        cav.trust_scores = tuple_to_dict(trust_scores_init, cav_names, idx)

        # Object Detection
        detected_objects, saved_image_path = detect_objects(image_path, model_object_detection, temp_dir)

        # Rename the image file to match the CAV name and frame number
        new_image_name = f"{cav.name}_frame_{1}.jpg"
        new_image_path = os.path.join(temp_dir, new_image_name)
        os.rename(saved_image_path, new_image_path)

        # Update the cav's detected_objects with the new image path
        cav.detected_objects = (detected_objects, new_image_path)

        # Classify Image
        labels, confidences = classify_image(image_path, model_classification)
        cav.shared_info = {"scene_label": labels, "confidence": confidences}

    # Initialize a logging dictionary
    log_data = {}

    # Initialize trust score tracking
    for cav in cavs:
        log_data[cav.name] = {
            other_cav.name: [] for other_cav in cavs if other_cav.name != cav.name
        }

    # Loop through the remaining images for each CAV
    for image_index in range(1, num_images + 1):  # Use num_images for accurate count
        for i, cav in enumerate(cavs):
            # Construct the path for the current image
            image_path = os.path.join(
                root_connection, f"Car{i + 1}", f"frame_{image_index}.jpg"
            )
            print(f"Processing {cav.name} with image {image_path}")

            # Update CAV's fov to the current image
            cav.fov = image_path
            detected_objects, saved_image_path = detect_objects(image_path, model_object_detection, temp_dir)

            # Rename the image file to match the CAV name and frame number
            new_image_name = f"{cav.name}_frame_{image_index}.jpg"
            new_image_path = os.path.join(temp_dir, new_image_name)
            os.replace(saved_image_path, new_image_path)

            # Update the cav's detected_objects with the new image path
            cav.detected_objects = (detected_objects, new_image_path)

            labels, confidences = classify_image(image_path, model_classification)
            cav.shared_info = {"scene_label": labels, "confidence": confidences}

            # Update and log trust scores
            for other_cav in cavs:
                if cav.name != other_cav.name:
                    user = users[cavs.index(cav)]  # Ensure the correct user is associated with the cav
                    # Pass the user's name to the assess_trust method

                    cav.share_info(other_cav, user, cav.detected_objects[1], other_cav.detected_objects[1])
                    new_trust_score = cav.assess_trust(other_cav.name, user.name, cav.detected_objects[1],
                                                       other_cav.detected_objects[1])

                    if new_trust_score is not None:
                        cav.trust_scores[other_cav.name] = new_trust_score
                        if cav.name not in log_data:
                            log_data[cav.name] = {}
                        if other_cav.name not in log_data[cav.name]:
                            log_data[cav.name][other_cav.name] = []
                        log_data[cav.name][other_cav.name].append(new_trust_score)

        print(f"Trust Scores after processing image {image_index}:")
        print(json.dumps({cav.name: cav.trust_scores for cav in cavs}, indent=4))

    # Save the log data to a file or further processing
    print("________________________________")
    print("Final Trust Recommendations:")
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = "results/" + f"{folder}" + f"/{folder}_{current_datetime}.json"
    with open(filename, "w") as file:
        json.dump(log_data, file, indent=4)

    print(json.dumps(trust_recommendations, indent=4))
    formatted_data = format_to_bullets(trust_recommendations)
    result_filename = "results/" + f"{folder}" + f"/{folder}_{current_datetime}.txt"
    with open(result_filename, "w") as file:
        file.write(formatted_data)


def main():
    for i in range(1, 2):  # Simulate only for Sample 1
        run_experience(folder="Sample" + str(i))
        # break
        time.sleep(60)


if __name__ == "__main__":
    main()