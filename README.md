# HITL AI Trust Framework

`The Human-in-the-Loop (HITL) Trust Evaluation System for Connected Autonomous Vehicles (CAVs) is designed to evaluate the trustworthiness of interactions between CAVs in real-time using machine learning and human oversight. The system processes data, including trust scores and visual evidence, to continuously assess how vehicles perceive and interact with each other, ultimately aiming to enhance decision-making in autonomous driving scenarios.`

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