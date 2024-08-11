class User:
    TRUST_LEVELS = {
        'Cautious': 0.8,
        'Moderate': 0.6,
        'Trusting': 0.3
    }

    def __init__(self, user_id, name, trust_level='Moderate', requires_trust_history=False, trust_frames_required=10):
        """
        Initialize a new user associated with CAV operations.

        Parameters:
        - user_id (int): A unique identifier for the user.
        - name (str): The name of the user.
        - trust_level (str): A string indicating the trust propensity level of the user,
                             which can be 'Cautious', 'Moderate', or 'Trusting'.
                             Defaults to 'Moderate'.
        - requires_trust_history (bool): Indicates if the user requires a history of trust
                                         assessments before fully trusting a CAV.
        - trust_frames_required (int): The number of data frames that need to appear trustworthy
                                       before the user's trust is adjusted if requires_trust_history is True.
        - trust_frames_tracker (int): The number of data frames that need to appear trustworthy. Initialized to 0.
        """
        self.user_id = user_id
        self.name = name
        self.trust_level = User.TRUST_LEVELS[trust_level]
        self.requires_trust_history = requires_trust_history
        self.trust_frames_required = trust_frames_required
        self.trust_frames_tracker = {}
        self.trust_overrides = {}  # Dictionary to store user's trust overrides by CAV id
        self.trust_history = {}    # Dictionary to store history of trust values per CAV
        self.trust_monitor = False

    def update_trust_frames_tracker(self, cav_id):
        """
        Increment the trust frame tracker for a specific CAV.

        Parameters:
        - cav_id (str): The identifier of the CAV being assessed.
        """
        if cav_id in self.trust_frames_tracker:
            self.trust_frames_tracker[cav_id] += 1
        else:
            self.trust_frames_tracker[cav_id] = 1  # Initialize tracker if not existent

    def check_trust_frame_threshold(self, cav_id):
        """
        Check if the trust frame threshold is met for a specific CAV.

        Parameters:
        - cav_id (str): The identifier of the CAV being assessed.

        Returns:
        - bool: True if the threshold is met, False otherwise.
        """
        return self.trust_frames_tracker.get(cav_id, 0) >= self.trust_frames_required

    def override_trust(self, cav_id, new_trust_value):
        """
        Allows the user to override the trust score for a specific CAV.

        Parameters:
        - cav_id (str): The identifier of the CAV whose trust score is to be overridden.
        - new_trust_value (float): The new trust value set by the user.
        """
        self.trust_overrides[cav_id] = new_trust_value

    def set_thresholds(self, cav_id, thresholds):
        """
        Sets new threshold requirements for trust modifications for a specific CAV.

        Parameters:
        - cav_id (str): The identifier of the CAV for which thresholds are being set.
        - thresholds (dict): A dictionary containing various threshold values.
        """
        if cav_id not in self.trust_overrides:
            self.trust_overrides[cav_id] = {}
        self.trust_overrides[cav_id].update({'thresholds': thresholds})

    def update_trust_history(self, cav_id, trust_value):
        """
        Updates the trust history for a specific CAV based on the new trust value received.

        Parameters:
        - cav_id (str): The identifier of the CAV for which trust history is being updated.
        - trust_value (float): The new trust value to be added to the history.
        """
        if cav_id not in self.trust_history:
            self.trust_history[cav_id] = []
        self.trust_history[cav_id].append(trust_value)

        # Check if trust history meets the required frames for adjustment
        if self.requires_trust_history and len(self.trust_history[cav_id]) >= self.trust_frames_required:
            # Process trust adjustment based on full history
            # This can be an averaging function or any other statistical analysis
            print(f"Processing trust adjustment for {cav_id} based on full history...")