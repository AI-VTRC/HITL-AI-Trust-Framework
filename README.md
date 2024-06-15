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


## Results 
Results are located in `src/results` section.

Reports are located in `src/reports` section.

Plots will be provided when run the `src/report.py`.

