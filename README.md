# HITL AI Trust Framework

`Object detection and scenes classification with trust assessment in computervision for CAVs.`

Steps:
- Initializes trust scores and detected objects; set up cavs with initial images.
- Process the first set of images to detect objects and classify scenes. Update trust scores and share information based on the detected overlaps and consistencies between CAVs
- Go thru each pairs of images, updating detected objects and shared information.
- Output trust score for all CAVs.

## Reproduction
On terminal
```python
conda create -n hitl python=3.8
conda activate hitl
pip install -r requirements.txt
cd src
python main.py
```