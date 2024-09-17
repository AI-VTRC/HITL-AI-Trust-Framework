import random
from hitl_utils import calculate_overlap, are_objects_consistent


class ConnectedAutonomousVehicle:
    def __init__(self, name, detected_objects=None, trust_scores=None, user=None):
        """
        Initialize a Connected Autonomous Vehicle with properties, historical data management,
        and direct access to user-specific settings.

        Parameters:
        - name (str): Identifier for the CAV.
        - detected_objects (list, optional): Initial list of objects detected by this CAV.
        - trust_scores (dict, optional): Initial dictionary of trust scores relative to other CAVs.
        - user (User, optional): The user object containing specific settings and trust management details.
        """
        self.name = name
        self.detected_objects = detected_objects if detected_objects else []
        self.previous_detected_objects = []  # Historical data for consistency checks
        self.trust_scores = trust_scores if trust_scores else {}
        self.history_length = 15  # Length to maintain history
        self.shared_info = {}
        self.user = user  # Directly use the user object

    def update_history(self):
        """Maintain a fixed-length history of detected objects."""
        if len(self.previous_detected_objects) >= self.history_length:
            self.previous_detected_objects.pop(0)
        self.previous_detected_objects.append(self.detected_objects.copy())

    def share_info(self, other_cav, user):
        """
        Share detected objects with another CAV and adjust trust scores based on the consistency of data observed.
        Includes consideration of scene labels and detection confidence as part of the decision process.

        Parameters:
        - other_cav (ConnectedAutonomousVehicle): Another CAV to compare detected objects with.
        - user (User): The user managing this CAV, with specific trust requirements.
        """
        # Simulate information reception from other CAV and initial trust assessment
        received_info = other_cav.shared_info
        received_scene_label = received_info["scene_label"]
        received_confidence = received_info["confidence"]

        # Initial trust assessment possibly influenced by scene labels and confidence
        self.trust_scores[other_cav.name] = self.assess_trust(other_cav.name, user.name)

        # Use received scene label and confidence to filter or adjust the consistency checks
        if received_scene_label == self.shared_info["scene_label"] \
                and received_confidence > self.shared_info["confidence"]:
            current_consistency = self.check_consistency(self.detected_objects, other_cav.detected_objects)
            historical_consistency = any(
                self.check_consistency(history_frame, other_cav.detected_objects)
                for history_frame in self.previous_detected_objects
            )
            consistency_factor = (current_consistency or historical_consistency)
        else:
            consistency_factor = False

        # Update trust based on user's requirement for historical data and consistency factor
        if user.requires_trust_history:
            self.handle_user_history(user, other_cav, consistency_factor)
        else:
            if consistency_factor:
                # If data is consistent, increase trust score slightly
                self.trust_scores[other_cav.name] += 0.1
            else:
                # If data is inconsistent, decrease trust score or maintain current level
                self.trust_scores[other_cav.name] = max(self.trust_scores[other_cav.name] - 0.1, 0)  # Never drop
                # below 0

        # Update the history with the latest detected objects
        self.update_history()

    def check_consistency(self, objects_1, objects_2):
        """
        Check for overlap and then consistency between two sets of objects from different CAVs.

        Parameters:
        - objects_1 (list): Objects detected by this CAV.
        - objects_2 (list): Objects detected by another CAV.

        Returns:
        - bool: True if there is an overlap and consistent objects, False otherwise.
        """
        # First, calculate if there is any meaningful overlap between the fields of view
        bboxes1 = [obj['box'] for obj in objects_1]
        bboxes2 = [obj['box'] for obj in objects_2]
        overlaps = calculate_overlap(bboxes1, bboxes2)

        # Check if any overlaps are above a certain threshold to consider them meaningful
        if not any(overlap > 0.1 for overlap in overlaps):  # Assuming 0.1 as a threshold for significant overlap
            return False  # No significant overlap found, so no consistency check is needed

        # If there is a significant overlap, check for consistency in detected objects
        return any(obj1['label'] == obj2['label'] and are_objects_consistent(obj1, obj2)
                   for obj1 in objects_1 for obj2 in objects_2)

    def handle_user_history(self, user, other_cav, is_consistent):
        """
        Manage the user's history requirements for trust adjustments based on consistency checks.

        Parameters:
        - user (User): The user associated with this CAV.
        - other_cav (ConnectedAutonomousVehicle): The other CAV involved in the trust calculation.
        - is_consistent (bool): Indicator of whether the current or historical data was consistent.
        """
        # Update the trust history based on the current consistency check
        if is_consistent:
            user.update_trust_history(self.name, 1)  # Log a positive consistency event
            if len(user.trust_history[self.name]) >= user.trust_frames_required:
                # Increase trust if the number of consistent events meets the user's required threshold
                self.trust_scores[other_cav.name] += 0.1  # Increment the trust score
                # user.trust_history[self.name] = []  # Reset the history after updating the trust score
        else:
            # Log a negative consistency event, potentially leading to a decrease in trust
            user.update_trust_history(self.name, -1)  # Log a negative consistency event
            if len(user.trust_history[self.name]) <= -user.trust_frames_required:
                # Decrease trust if the number of inconsistent events exceeds the user's tolerance
                self.trust_scores[other_cav.name] = max(self.trust_scores[other_cav.name] - 0.1, 0)  # Decrement
                # trust score
                # user.trust_history[self.name] = []  # Reset the history after updating the trust score

    def assess_trust(self, cav_name, user):
        """
        Assess the trust score for a specific CAV based on the DC trust model.
        This implementation is simplified and should be adapted for specific operational needs.

        Parameters:
        - cav_name (str): The CAV whose trust is being assessed.

        Returns:
        - float: Updated trust score for the given CAV.
        """
        if cav_name == self.name:
            return 1.0

        # Setup to retrieve user-specific settings like trust thresholds and history requirements
        user_threshold = self.user.trust_level
        requires_trust_history = self.user.requires_trust_history

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
                        trust_score_a < user_threshold <= trust_score_b < 1.0
                ):
                    omega_ij = 0.6  # Set to a higher value to trust the other CAV

        # Updating the trust score in the trust_scores dictionary
        # Check if the user requires trust history and if the threshold of trust frames is met
        if requires_trust_history:
            if self.user.check_trust_frame_threshold(cav_name):
                # Update trust score conditionally based on user settings
                self.trust_scores[cav_name] = omega_ij
                return omega_ij
            else:
                # Don't change initialized trust value, but update user history that a trustworthy comparison was made
                self.user.update_trust_frames_tracker(cav_name)
                return self.trust_scores.get(cav_name, 0)  # Return existing trust score if threshold not met

        # If User does not require a trust history with the other cav, just update based on the DC model
        else:
            # Update trust score directly if no history is required
            self.trust_scores[cav_name] = omega_ij
            return omega_ij