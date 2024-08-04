import random
from user import User
from utils import calculate_overlap
from utils import are_objects_consistent


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
                        trust_score_a < self.trust_threshold <= trust_score_b < 1.0
                ):
                    omega_ij = 0.6  # Set to a higher value to trust the other CAV

        # Updating the trust score in the trust_scores dictionary
        self.trust_scores[cav_name] = omega_ij

        return omega_ij

    def share_info(self, other_cav, user):
        """
        Simulate sharing of detected objects and scene information with another CAV.
        Adjusts trust scores based on current and historical data consistency between CAVs.

        Parameters:
        - other_cav (ConnectedAutonomousVehicle): The other CAV to share information with.
        - user (User): The user associated with this CAV who may adjust trust scores.
        """
        received_info = other_cav.shared_info
        received_scene_label = received_info["scene_label"]
        received_confidence = received_info["confidence"]

        # Initial trust assessment
        self.trust_scores[other_cav.name] = self.assess_trust(other_cav.name)

        # Current frame overlap and consistency check
        current_overlap = calculate_overlap([obj["box"] for obj in self.detected_objects],
                                            [obj["box"] for obj in other_cav.detected_objects])
        current_consistent = False
        if any(current_overlap):
            for obj_1, obj_2 in zip(self.detected_objects, other_cav.detected_objects):
                if obj_1["label"] == obj_2["label"] and are_objects_consistent(obj_1, obj_2):
                    current_consistent = True
                    if self.process_consistency(obj_1, obj_2, user, received_confidence):
                        break  # Exit after processing the first consistent match

        if not current_consistent and hasattr(self, 'previous_detected_objects'):
            # Check historical data for consistency
            historical_overlap = calculate_overlap([obj["box"] for obj in self.previous_detected_objects],
                                                   [obj["box"] for obj in other_cav.detected_objects])
            if any(historical_overlap):
                for obj_1, obj_2 in zip(self.previous_detected_objects, other_cav.detected_objects):
                    if obj_1["label"] == obj_2["label"] and are_objects_consistent(obj_1, obj_2):
                        if self.process_consistency(obj_1, obj_2, user, received_confidence):
                            break  # Exit after processing the first historical consistent match

        # Store the current detected objects for future historical checks
        self.previous_detected_objects = self.detected_objects

        print(f"Processed trust interaction between {self.name} and {other_cav.name}")

    def process_consistency(self, obj_1, obj_2, user, received_confidence):
        """
        Process and update trust based on object consistency and user settings.

        Returns True if trust was updated, False otherwise.
        """
        if user.requires_trust_history:
            user.update_trust_history(self.name, 1)
            if len(user.trust_history[self.name]) >= user.trust_frames_required:
                increment = 0.1 + (received_confidence - 0.5) * 0.05
                self.trust_scores[other_cav.name] += max(0, increment)  # Ensure no negative increment
                user.trust_history[self.name] = []  # Reset the trust history
                return True
        else:
            self.trust_scores[other_cav.name] += 0.1
            return True
        return False
