import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)

# Define the treatments
trust_history_required = [True, False]   # 2 levels for the 'Trust History Required' factor
num_frames_required = list(range(0, 11))  # 11 levels for 'Number of Trust Frames Required' (0 to 10)

# Generate all possible combinations of treatments (22 combinations)
treatments = [(h, f) for h in trust_history_required for f in num_frames_required]

# Define the number of users (blocks) and replicates
user_types = ['User_0.8', 'User_0.6', 'User_0.3', 'User_M']  # 4 user types (blocks)
experiments = 5  # 5 different datasets

# List to store the assignment
assignment = []

# Random assignment of treatments to each user for each experiment
for user in user_types:
    if user == 'User_M':  # Different handling for "User_M"
        for experiment in range(1, experiments + 1):
            # Randomly select history required (True/False) and num of frames from the entire range for "User_M"
            random_history = random.choice(trust_history_required)
            random_frames = random.choice(num_frames_required)
            assignment.append({
                'Experiment (Dataset)': experiment,
                'User_Type': user,
                'Trust History Required': random_history,
                'Number of Trust Frames Required': random_frames
            })
    else:
        user_treatments = random.sample(treatments, experiments)  # Randomly sample 5 treatments for each user
        for experiment in range(1, experiments + 1):
            treatment = user_treatments.pop()  # Pop ensures no treatment repeats for the user (sampling without replacement)
            assignment.append({
                'Experiment (Dataset)': experiment,
                'User_Type': user,
                'Trust History Required': treatment[0],
                'Number of Trust Frames Required': treatment[1]
            })


# Convert the assignment to a DataFrame for better visualization
assignment_df = pd.DataFrame(assignment)

# Display the randomization assignment
print(assignment_df)

assignment_df.to_csv('randomized_design.csv', index=False)
