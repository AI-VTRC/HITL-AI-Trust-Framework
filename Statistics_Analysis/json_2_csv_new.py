import pandas as pd
import json
import os

# Path to the JSON file
file_path = r'C:\HITL-AI-Trust-Framework\Simulation_Results\Original_Algorithm\json\control\control_threshold_0.6_new.json'

# Load JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Prepare to flatten the data and load into DataFrame
rows = []

# Loop through each key in the JSON and flatten the structure
for cav_key, cav_values in data.items():
    for trust_category, cav_trust_scores in cav_values['trust_scores'].items():
        for index, score in enumerate(cav_trust_scores):
            # Creating a row for each score
            row = {
                'CAV Receiver': cav_key,
                'CAV Sender': trust_category,
                'Image Frame Index': index,
                'Trust Score': score
            }
            rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

# Saving DataFrame to a CSV file, maintaining the base file name
os.chdir(r'C:\HITL-AI-Trust-Framework\Simulation_Results\Original_Algorithm\csv')
output_file = os.path.splitext(file_path)[0] + '.csv'
df.to_csv(output_file, index=False)

print(f'DataFrame has been saved as CSV: {output_file}')
