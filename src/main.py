import os
import json
import torch
from ultralytics import YOLO
from datetime import datetime

from cavs import ConnectedAutonomousVehicle
from utils import classify_image
from utils import detect_objects
from utils import create_cav_objects
from utils import tuple_to_dict
from utils import format_to_bullets


def main():
    """Main Driver"""
    # Load the pre-trained multi-label classification model and freeze parameters
    model_classification = torch.hub.load(
        "facebookresearch/WSL-Images", "resnext101_32x48d_wsl"
    )
    model_classification.eval()

    # Load the pre-trained object detection model
    model_object_detection = YOLO("yolov8n.pt")

    # Define the trust threshold
    trust_threshold = 0.5

    # Define the trust propagation and trust fusion data structures
    trust_recommendations = {}

    root_connection = "../data/Sample0"
    num_cars = 4

    # Initialize trust values for connected agents
    trust_scores_init, detected_objects_init = create_cav_objects(num_cars)

    # Initialize the first set of images for each CAV
    image_paths = [
        os.path.join(root_connection, f"Car{i}", "frame_1.jpg")
        for i in range(1, num_cars + 1)
    ]
    
    # Initialize CAVs with the first image
    cavs = [
        ConnectedAutonomousVehicle(
            name=f"cav{i}",
            fov=image_paths[i - 1],
            trust_scores=trust_scores_init[f"cav{i}"],
            detected_objects=detected_objects_init[f"cav{i}"],
            trust_threshold=trust_threshold,
            trust_recommendations=trust_recommendations,
        )
        for i in range(1, num_cars + 1)
    ]
    
    trust_scores_init = list(trust_scores_init.values())
    cav_names = [cav.name for cav in cavs]

    # Process FOVs for each CAV at the current time.
    for idx, cav in enumerate(cavs):
        print(f"Processing FOVs for {cav.name}")

        image_path = image_paths[idx]

        cav.trust_scores = tuple_to_dict(trust_scores_init, cav_names, idx)

        # Object Detection
        cav.detected_objects = detect_objects(image_path, model_object_detection)

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
        print("________________________________")

    # NEW processing to check and refine
    # Loop through the remaining images for each CAV
    for image_index in range(
        2, 19
    ):  # Assuming there are 18 images, starting from 2 since 1 was used for initialization
        for i, cav in enumerate(cavs):
            # Construct the path for the current image
            image_path = os.path.join(
                root_connection, f"Car{i + 1}", f"frame_{image_index}.jpg"
            )
            print(f"Processing {cav.name} with image {image_path}")

            # Update CAV's fov to the current image
            cav.fov = image_path

            # Example processing steps, modify according to your actual functions
            cav.trust_scores = tuple_to_dict(
                trust_scores_init, [cav.name for cav in cavs], i
            )
            cav.detected_objects = detect_objects(image_path, model_object_detection)
            labels, confidences = classify_image(image_path, model_classification)
            cav.shared_info = {"scene_label": labels, "confidence": confidences}

    print("________________________________")
    print("Final Trust Recommendations:")
    result = json.dumps(trust_recommendations, indent=4)
    print(result)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"results/Sample0/{current_datetime}.txt"
    formatted_data = format_to_bullets(trust_recommendations)
    with open(filename, "w") as file:
        file.write(formatted_data)


if __name__ == "__main__":
    main()
