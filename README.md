# HITL AI Trust Framework

## About

The Human-in-the-Loop (HITL) Trust Evaluation System is a Python-based framework for simulating and analyzing trust dynamics in distributed AI systems, particularly for Connected Autonomous Vehicles (CAVs). This repository contains two comparative frameworks:

- The **Original Baseline Trust Framework**, which performs pairwise trust assessments using algorithmic inputs only.
- The proposed **PerceptiSync** system, which integrates both Human-in-the-Loop (HITL) and Crowds-in-the-Loop (CITL) feedback, enabling dynamic trust adaptation through user-configurable parameters.

At the core of PerceptiSync is a **Dirichlet-Categorical (DC) trust model**, which maintains trust distributions over time and supports configurable behavior such as:
- Trust history requirements
- Frame-based evidence thresholds
- Real-time trust monitoring and overrides
- User-defined trust level categories (Cautious, Moderate, Trusting, Dynamic)

The framework evaluates how trust evolves as CAVs exchange object detection outputs in simulated driving environments. All experiments use visual data from the **nuScenes** dataset, and incorporate personalized trust profiles for each simulated user or vehicle.

### Key Features
- **Quantitative modeling of trust** using DC distributions:
  - Trust scores are updated using feedback evidence and detection matches
  - Trust evolution follows the equation:  
    \[
    \theta_i \sim \text{Dir}(\alpha_i), \quad T_i = \frac{\alpha_{\text{trust}}}{\sum \alpha}
    \]
- **Support for subjective trust preferences** in simulation
- **Comparison between PerceptiSync and the baseline** in terms of:
  - Responsiveness to false detections
  - Human alignment and override potential
  - Expected trust behavior over time

### Trust Pattern Expectations

PerceptiSync evaluations are informed by theoretical expectations from recent research (e.g., Cirne et al.), which suggest that well-functioning trust systems exhibit **gradual, step-wise increases in trust** under reliable conditions. These patterns are used to interpret the results and validate improvements over the baseline.

### Statistical Evaluation

The repository includes statistical scripts for:
- **Mann-Kendall Trend Test** (nonparametric test of monotonic trends)
- **Welch’s T-test**, **One-Way ANOVA**, and **Tukey HSD** to compare trust behaviors across configurations
- Analysis of **Kendall's τ** coefficient to assess the consistency and direction of trust progression

This framework supports both academic research and prototyping of HITL/CITL trust modeling methods for safety-critical CPS and Embodied AI systems.

---

## Repository Contents

- `data/`: Preprocessed nuScenes data used in the experiments.
- `data_bbx/`: Bounding box-level annotations used in trust evaluation for each frame.
- `Original_Algorithm/`: Baseline trust framework with limited configurability and no human-in-the-loop feedback.
- `HITL_Algorithm/`: Implementation of the PerceptiSync framework with support for human-in-the-loop and crowds-in-the-loop configurations.
- `Simulation_Results/`: Trust assessment outputs (e.g., time series trust scores) for each randomized experiment configuration across all test scenes.
- `Statistics_Analysis/`: Python scripts used to conduct all statistical analyses in the paper (e.g., Mann-Kendall Trend Test, Welch’s t-test, ANOVA, Tukey HSD).
- `papers/`: Supporting documents related to the published research.
- `README.md`: This file.
- `requirements.txt`: Python package dependencies needed to run experiments and analyses.

---

## Running the Code

This project requires **Python 3.8 or higher**. We recommend using a conda environment.

### Step 1: Set up the environment
```bash
conda create -n hitl python=3.8
conda activate hitl
pip install -r requirements.txt

## Getting started
On terminal, to install package
```python
conda create -n hitl python=3.8
conda activate hitl
pip install -r requirements.txt
```

On terminal, to perform an interactive Streamlit visualization for the results
```sh
cd Simulation_Results
streamlit run new_interactive_report_streamlit.py
```

### Step 2: Run either the baseline or PerceptiSync framework

To run the original baseline trust framework:

```
cd Original_Algorithm
python run_baseline_experiment.py
```

To run the PerceptiSync (HITL/CITL) framework:

```
cd HITL_Algorithm
python run_hitl_experiment.py
```
### Step 3: View interactive results (optional)

Use Streamlit to visualize simulation outputs:

```
cd Simulation_Results
streamlit run new_interactive_report_streamlit.py
```

## Reproducibility

This repository supports full reproduction of the experiments described in the paper, including:

- Experiment configuration and randomized block design

- Decay factor (λ), trust feature toggles, and trust level settings

- Visualization of trust progression across scenes

- Statistical significance testing of performance differences

All data, parameters, and code dependencies are included for transparency and reproducibility.

## Citation
If you use this framework in your work, please cite our paper:

PerceptiSync: Trustworthy Object Detection using Crowds-in-the-Loop for Cyber-Physical Systems (2025)
```
@article{perceptisync2025,
  author    = {Matthew Wilchek and Minh Nguyen and Yingjie Wang and Kurt Luther and Feras A. Batarseh},
  title     = {PerceptiSync: Trustworthy Object Detection using Crowds-in-the-Loop for Cyber-Physical Systems},
  journal   = {ACM Transactions on Cyber-Physical Systems},
  volume    = {1},
  number    = {1},
  pages     = {1--25},
  month     = {January},
  year      = {2025},
  articleno = {1},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3746644},
  url       = {https://doi.org/10.1145/3746644}
}
```


