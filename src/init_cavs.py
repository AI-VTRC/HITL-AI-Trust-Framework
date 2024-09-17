import random
from init_utils import calculate_overlap
from init_utils import are_objects_consistent


class ConnectedAutonomousVehicle:
    """
    Represents a Connected Autonomous Vehicle (CAV) and its operations related to object detection, trust assessment,
    and information sharing with other CAVs.

    @Attributes:
    - name (str): Unique identifier for the CAV.
    - fov (str): Field of View for the CAV.
    - trust_scores (dict): Dictionary containing trust scores for other CAVs.
    - detected_objects (list): List of objects detected by the CAV.
    - shared_info (dict): Information that the CAV chooses to share with others.
    """

    def __init__(
        self,
        name,
        fov,
        trust_scores,
        trust_threshold,
        trust_recommendations,
        detected_objects=None,
    ):
        self.name = name
        self.fov = fov
        self.trust_scores = trust_scores if trust_scores else {}
        self.detected_objects = detected_objects if detected_objects else []
        self.shared_info = {}
        self.trust_threshold = trust_threshold
        self.trust_recommendations = trust_recommendations

    def assess_trust(self, cav_name):
        """
        Assess the trust score for a specific CAV based on the DC trust model.
        In this implementation, a simplified trust assessment model is used which might not reflect real-world complexities.

        @Parameters:
        - cav_name (str): The name of the CAV whose trust is being assessed.

        @Returns:
        - float: Updated trust score for the given CAV.
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
        alpha_ij = (
            positive_evidence + aij * 10
        )  # considering aij as a weight for pseudo count
        beta_ij = (
            negative_evidence + (1 - aij) * 10
        )  # considering (1-aij) as a weight for pseudo count
        gamma_ij = uncertain_evidence

        # Computing the trust value omega_ij as the expected value
        total_count = alpha_ij + beta_ij + gamma_ij
        omega_ij = alpha_ij / total_count  # Expected trust value (Trustworthiness)

        # Trust fusion with other CAVs in the system
        for other_cav_name, trust_score in self.trust_scores.items():
            if other_cav_name != cav_name:
                trust_score_a = omega_ij
                trust_score_b = trust_score
                if (
                    trust_score_a < self.trust_threshold
                    and self.trust_threshold <= trust_score_b < 1.0
                ):
                    omega_ij = 0.6  # Set to a higher value to trust the other CAV

        # Updating the trust score in the trust_scores dictionary
        self.trust_scores[cav_name] = omega_ij

        return omega_ij

    def share_info(self, other_cav):
        """
        Simulate sharing of detected objects and scene information with another CAV.
        In this implementation, a simplified trust assessment model is used which might not reflect real-world complexities.

        @Parameters:
        - other_cav (ConnectedAutonomousVehicle): The other CAV to share information with.
        """

        # Simulate information reception by other CAV and trust assessment
        received_info = other_cav.shared_info
        received_scene_label = received_info["scene_label"]
        received_confidence = received_info["confidence"]

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
                    if obj_1["label"] == obj_2["label"]:
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
            if self.name not in self.trust_recommendations:
                self.trust_recommendations[self.name] = {}

            self.trust_recommendations[self.name][other_cav.name] = self.trust_scores[
                other_cav.name
            ]  # ADD BREAK POINT HERE

            # Compare shared_info
            if self.shared_info["scene_label"] == received_scene_label:
                if received_confidence > self.shared_info["confidence"]:
                    # Increase the trust value for the other CAV based on some criteria (e.g., by 10%)
                    trust_increment = 0.10
                    trust_value = self.trust_recommendations[self.name][
                        other_cav.name
                    ]  # ADD BREAK POINT HERE
                    trust_value += trust_increment  # ADD BREAK POINT HERE

                    # Ensure trust value doesn't exceed 1.0
                    trust_value = min(trust_value, 1.0)

                    self.trust_recommendations[self.name][other_cav.name] = trust_value

        else:
            # No FOV overlap, recommend trust to other CAV
            if self.name not in self.trust_recommendations:
                self.trust_recommendations[self.name] = {}
            self.trust_recommendations[self.name][other_cav.name] = self.trust_scores[
                other_cav.name
            ]

        # Update self.trust_scores based on self.trust_recommendations
        for cav_name, recommended_trust in self.trust_recommendations[
            self.name
        ].items():
            if cav_name != self.name:  # Exclude the original self
                self.trust_scores[cav_name] = recommended_trust

        return
