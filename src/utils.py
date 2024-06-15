import cv2
import json
import torch
import requests
from PIL import Image
import torchvision.transforms as transforms


def classify_image(image_path, model_classification):
    """
    Process and classify an image to predict the scene it represents using a given classification model.
    The function attempts to load class labels from a local file named 'imagenet-simple-labels.json'. If the file
    is not found, it fetches the labels from a URL.

    @Parameters:
    - image_path (str): Path to the image file to be classified.
    - model_classification (torch.nn.Module): Pre-trained ResNet model for scene classification.

    @Returns:
    - tuple (str, float): A tuple containing the top predicted class label and its corresponding confidence score.

    @Raises:
    - ValueError: If the image can't be loaded or if its dimensions are invalid.
    - ConnectionError: If there's an issue fetching the class labels from the remote URL.
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

    # Make a copy, convert image to RBG, convert numpy to PIL iamge, and add batch dimension
    original_image = img.copy()
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(original_image_rgb)
    image = preprocess_classification(image)
    image = image.unsqueeze(0)

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
def detect_objects(image_path, model_object_detection):
    """
    Perform object detection on an image using a pre-defined object detection model (e.g., YOLO).
    The function relies on a globally-defined object detection model (`model_object_detection`) for predictions. Ensure
    that this model is properly initialized and loaded before calling this function.

    @Parameters:
    - image_path (str): Path to the image file on which object detection is to be performed.

    @Returns:
    - list[dict]: A list of dictionaries, where each dictionary represents a detected object and contains:
        - 'label' (str): Name of the detected object.
        - 'confidence' (float): Confidence score of the detection.
        - 'box' (list[float]): Coordinates of the bounding box in the format [x1, y1, x2, y2].
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
    detected_objects_init = {f"cav{i + 1}": [] for i in range(num_cavs)}

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
        trust_scores_init[f"cav{i + 1}"] = tuple(scores)

    return trust_scores_init, detected_objects_init


def calculate_overlap(bboxes1, bboxes2):
    """
    Calculate the overlap between lists of bounding boxes. The overlap for each bounding box
    in bboxes1 with every bounding box in bboxes2 is computed and returned as a list.

    @Parameters:
    - bboxes1 (list of tuples): A list of bounding boxes, where each bounding box is represented
                                as a tuple in the format (x1, y1, x2, y2).
    - bboxes2 (list of tuples): A list of bounding boxes, where each bounding box is represented
                                as a tuple in the format (x1, y1, x2, y2).

    @Returns:
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

    @Parameters:
    - boxA (list): A list containing the coordinates of the first bounding box in the format [x1, y1, x2, y2],
                where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    - boxB (list): A list containing the coordinates of the second bounding box in the same format as boxA.

    @Returns:
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

    @Parameters:
    - objA: a dictionary representing an object with keys 'label' and 'box'
    - objB: a dictionary representing an object with keys 'label' and 'box'
    - iou_threshold: the threshold for the IoU to consider objects consistent

    @Returns:
    - True if objects are consistent, otherwise False.
    """
    if objA["label"] != objB["label"]:
        return False

    iou = compute_iou(objA["box"], objB["box"])

    return iou >= iou_threshold


def format_to_bullets(d, indent=0):
    """Reformat data into bullet points"""
    result = ""
    for key, value in d.items():
        result += "    " * indent + f"- {key}:\n"
        if isinstance(value, dict):
            result += format_to_bullets(value, indent + 1)
        else:
            result += "    " * (indent + 1) + f"- {value}\n"
    return result
