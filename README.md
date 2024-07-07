# HITL AI Trust Framework

`Object detection and scenes classification with trust assessment in computervision for CAVs.`

Steps:
- `Initialization`: Each ConnectedAutonomousVehicle (CAV) object is initialized using the first image from each of the Car directories (Car1 to Car4). This initial step sets up the scene classification and object detection for each CAV based on the first image. The initial trust scores among the CAVs are also set up using create_cav_objects.
- `First Image Processing`: Each CAV processes its first image to classify the scene and detect objects. This information is stored within each CAV's state (e.g., detected_objects, shared_info).
- `Trust Score Assessment and Update`: After processing the first image, each CAV shares the detected objects and scene information with the other CAVs. The trust scores are then assessed and updated based on the overlap and consistency of detected objects, as well as the classification results.
- `Iterative Processing for Subsequent Images`: The process repeats for each subsequent image in each car directory. For each new image (from the second to the eighteenth):
    - Each CAV updates its field of view to the new image.
    - It processes the new image for object detection and scene classification.
    - It shares this updated information with other CAVs.
    - Trust scores are reassessed and updated based on the new shared information and any new detections.
- `Continuous Update`: With each new image, trust scores are dynamically updated. These updates reflect the latest interactions and shared data between the CAVs. The trust scores can increase if the new data aligns well across the CAVs or decrease if inconsistencies are found.
- `Trust Score Understanding (current developemnt)`: trust scores are updated iteratively based on the outcomes of each new interaction—each new image processed introduces a new opportunity for trust assessment based on the data shared and detected between the CAVs.

## Reproduction
On terminal, to get the result of the run experience
```python
conda create -n hitl python=3.8
conda activate hitl
pip install -r requirements.txt
cd src
python main.py
```

On terminal, to get the .csv reports and visualization for the results, alter the details in `src/report.py`
```sh
cd src
python report.py
```

On terminal, to perform an interactive Dash visualization for the results, alter the details in `src/report_interactive.py`
```sh
cd src
python report_interactive.py
```

## Implementation validation
In the `Towards Trustworthy Perception Information Sharing on Connected and Autonomous Vehicles`, the author use Dirichlet-Categorical (DC) model for trust assessment. Our `cavs.py` correctly implement the DDC model for trust assessment
```python
# Simulate trust assessment based on the DC trust model
positive_evidence = random.randint(0, 10)
negative_evidence = random.randint(0, 10)
uncertain_evidence = random.randint(0, 10)

# Constants (aij) representing prior opinions
aij = random.uniform(0, 1)

# Trust assessment logic
alpha_ij = positive_evidence + aij * 10
beta_ij = negative_evidence + (1 - aij) * 10
gamma_ij = uncertain_evidence

# Computing the trust value omega_ij as the expected value
total_count = alpha_ij + beta_ij + gamma_ij
omega_ij = alpha_ij / total_count  # Expected trust value (Trustworthiness)
```

Also, our trust fusion to update the trust scores of other vehicles are correct compare tot the paper
```python
for other_cav_name, trust_score in self.trust_scores.items():
    if other_cav_name != cav_name:
        trust_score_a = omega_ij
        trust_score_b = trust_score
        if trust_score_a < self.trust_threshold and self.trust_threshold <= trust_score_b < 1.0:
            omega_ij = 0.6  # Set to a higher value to trust the other CAV

```

In addtion, the intiialization, object detection and object classification is correct compare to the paper.

Futhermore, the CAVs share info and logging is correct compare to the paper.
```python
# Update and log trust scores
for other_cav in cavs:
    if cav.name != other_cav.name:
        cav.share_info(other_cav)
        new_trust_score = cav.assess_trust(other_cav.name)
        if new_trust_score is not None:
            cav.trust_scores[other_cav.name] = new_trust_score
            log_data[cav.name][other_cav.name].append(new_trust_score)

```

## Things to be improved
- Evidence generation: I am not sure if this will reflect the real-world

```python
# Simulate trust assessment based on the DC trust model
positive_evidence = random.randint(0, 10)
negative_evidence = random.randint(0, 10)
uncertain_evidence = random.randint(0, 10)

# Constants (aij) representing prior opinions
aij = random.uniform(0, 1)
```

- Trust fusion: right now the trust fusion scheme does not consider historical evidence or interactions over time.
```python
for other_cav_name, trust_score in self.trust_scores.items():
    if other_cav_name != cav_name:
        trust_score_a = omega_ij
        trust_score_b = trust_score
        if trust_score_a < self.trust_threshold and self.trust_threshold <= trust_score_b < 1.0:
            omega_ij = 0.6  # Set to a higher value to trust the other CAV
```

- Temporal in info sharing: right now, there is no temporal factor included when updating the trust score.
```python
# Trust assessment logic
alpha_ij = positive_evidence + aij * 10
beta_ij = negative_evidence + (1 - aij) * 10
gamma_ij = uncertain_evidence

# Computing the trust value omega_ij as the expected value
total_count = alpha_ij + beta_ij + gamma_ij
omega_ij = alpha_ij / total_count  # Expected trust value (Trustworthiness)
```

## Results 
Results are located in `src/results` section.

Reports are located in `src/reports` section.

Plots will be provided when run the `src/report.py`.

