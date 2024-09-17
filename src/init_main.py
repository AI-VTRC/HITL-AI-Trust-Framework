import time
import os
import json
import torch
from ultralytics import YOLO
from datetime import datetime
import cv2

from init_cavs import ConnectedAutonomousVehicle
from init_utils import classify_image
from init_utils import detect_objects
from init_utils import create_cav_objects
from init_utils import tuple_to_dict
from init_utils import format_to_bullets
from init_utils import get_image_count


def run_experience(folder, trust_threshold):
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

    root_connection = "assets/data/" + folder
    num_cars = 4

    # Assumes the number of images for each Car is the same
    # Determine the number of images by checking the first car's directory
    example_car_folder = os.path.join(root_connection, "Car1")
    num_images = get_image_count(example_car_folder)

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

    # Initialize a logging dictionary
    log_data = {}

    # Initialize trust score tracking
    for cav in cavs:
        log_data[cav.name] = {
            "trust_scores": {
                other_cav.name: [] for other_cav in cavs if other_cav.name != cav.name
            },
            "detected_objects": [],
        }

    # Create results directory
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"results/{folder}/{current_datetime}"
    os.makedirs(results_dir, exist_ok=True)

    # Process FOVs for each CAV at the current time.
    for idx, cav in enumerate(cavs):
        print(f"Processing FOVs for {cav.name}")

        image_path = image_paths[idx]

        cav.trust_scores = tuple_to_dict(trust_scores_init, cav_names, idx)

        # Object Detection
        cav.detected_objects, img_with_boxes = detect_objects(
            image_path, model_object_detection
        )

        # Save the image with bounding boxes and labels
        img_output_path = os.path.join(
            results_dir, f"Car{idx + 1}_frame_1_with_boxes.jpg"
        )
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
                "confidence": confidences,
            }
        )

    # Loop through the remaining images for each CAV
    for image_index in range(2, num_images + 1):  # Use num_images for accurate count
        for i, cav in enumerate(cavs):
            # Construct the path for the current image
            image_path = os.path.join(
                root_connection, f"Car{i + 1}", f"frame_{image_index}.jpg"
            )
            print(f"Processing {cav.name} with image {image_path}")

            # Update CAV's fov to the current image
            cav.fov = image_path
            cav.detected_objects, img_with_boxes = detect_objects(
                image_path, model_object_detection
            )

            # Save the image with bounding boxes and labels
            img_output_path = os.path.join(
                results_dir, f"Car{i + 1}_frame_{image_index}_with_boxes.jpg"
            )
            cv2.imwrite(img_output_path, img_with_boxes)

            labels, confidences = classify_image(image_path, model_classification)
            cav.shared_info = {"scene_label": labels, "confidence": confidences}

            # Update and log trust scores
            for other_cav in cavs:
                if cav.name != other_cav.name:
                    cav.share_info(other_cav)
                    new_trust_score = cav.assess_trust(other_cav.name)
                    if new_trust_score is not None:
                        cav.trust_scores[other_cav.name] = new_trust_score
                        log_data[cav.name]["trust_scores"][other_cav.name].append(
                            new_trust_score
                        )

            # Log detected objects
            log_data[cav.name]["detected_objects"].append(
                {
                    "frame": image_index,
                    "objects": cav.detected_objects,
                    "scene_label": labels,
                    "confidence": confidences,
                }
            )

        print(f"Trust Scores after processing image {image_index}:")
        print(json.dumps({cav.name: cav.trust_scores for cav in cavs}, indent=4))

    # Save the log data to a file or further processing
    print("________________________________")
    print("Final Trust Recommendations:")
    filename = os.path.join(
        results_dir, f"{folder}_{current_datetime}_threshold_{trust_threshold}.json"
    )
    with open(filename, "w") as file:
        json.dump(log_data, file, indent=4)

    print(json.dumps(trust_recommendations, indent=4))
    formatted_data = format_to_bullets(trust_recommendations)
    result_filename = os.path.join(results_dir, f"{folder}_{current_datetime}.txt")
    with open(result_filename, "w") as file:
        file.write(formatted_data)


def main():
    trust_thresholds = [0.3, 0.5, 0.8]
    for trust_threshold in trust_thresholds:
        for i in range(1):
            run_experience(folder="8_29_24_scenario_" + str(i + 1), trust_threshold=trust_threshold)
            # break
            time.sleep(10)


if __name__ == "__main__":
    main()
