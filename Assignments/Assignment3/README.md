# Traffic Signs & Vehicles Detection using YOLOv8

This project uses a YOLOv8 object detection model to detect traffic signs and vehicles from street images captured by vehicle-mounted cameras. The project includes code for training, testing, and performing inference on your dataset.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing and Inference](#testing-and-inference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)

---

## Overview

The goal of this project is to implement small object detection using YOLOv8 for traffic-related scenes. The dataset is derived from the Mapillary Traffic Sign Dataset, which includes various traffic signs (e.g., stop_sign, yield_sign) and vehicles (e.g., car, bus, truck). To better detect small and distant objects, experiments include higher resolution (1280×1280), data augmentation (rotation, shear, mosaic), and anchor box tuning.

---

## Dataset

Your dataset should follow this structure:
dataset/ ├── train/ │ ├── images/ │ └── labels/ ├── valid/ │ ├── images/ │ └── labels/ └── test/ ├── images/ └── labels/

perl
Copy code
Each `.txt` label file uses the YOLO format:
<class_id> <x_center> <y_center> <width> <height>

yaml
Copy code
All coordinates are normalized to [0,1].

---

## Prerequisites

- Python 3.8+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)  
- Roboflow (optional, if using for annotation or dataset export)
- (Optional) GPU support for faster training (e.g., NVIDIA CUDA)

---

## Installation

1. **Clone or download the repository (optional)**  
   ```bash
   git clone https://github.com/yourusername/traffic-signs-yolov8.git
   cd traffic-signs-yolov8
Create and activate a virtual environment (recommended)

 
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
Install Ultralytics and any additional dependencies


pip install ultralytics opencv-python matplotlib
Usage
Training
Use the YOLOv8 Python API to fine-tune a pre-trained model on your dataset. Below is a sample script (train.py):

python

from ultralytics import YOLO

# Load a pre-trained YOLO model (e.g., YOLOv8n)
model = YOLO("yolov8n.pt")

# Train on your dataset
# - data: path to your dataset YAML file (with train/val definitions and class names)
# - imgsz: training image size (e.g., 1280 for small object detection)
# - epochs: number of epochs to train
# - batch: training batch size
model.train(data="data.yaml", imgsz=1280, epochs=30, batch=16)

Run the script:


python train.py
Testing and Inference
To run inference on images or videos:

Single image:

results = model("sample_image.jpg", conf=0.5, stream=True)
for r in results:
    r.show()  # visualize the detection
Video:

results = model("traffic_video.mp4", conf=0.5, stream=True)
for r in results:
    # Each iteration provides detections for a frame
    # You can process or save frames here
    pass
Use the stream=True option to avoid accumulating inference results in memory for larger sources.

Configuration
data.yaml (Dataset Configuration)
Below is an example data.yaml that references train, val, test paths and lists class names:

yaml

train: ../train/images
val: ../valid/images
test: ../test/images

nc: 11
names:
  - bicycle
  - bus
  - car
  - motorcycle
  - no_entry_sign
  - pedestrian
  - stop_sign
  - pedestrian_crossing_sign
  - traffic_light
  - truck
  - yield_sign

roboflow:
  workspace: yolo-jesbe
  project: traffic_signs_detector
  version: 2
  license: CC BY 4.0
  url: https://universe.roboflow.com/yolo-jesbe/traffic_signs_detector/dataset/2
Augmentations and Hyperparameters
You can modify a YOLO hyperparameter file (like hyp.scratch.yaml) to experiment with:

Rotation (degrees: 15)
Shear (shear: 10)
Scale (scale: 0.5)
Mosaic (set mosaic: 1.0 for 100% mosaic probability)
Blur if applicable (added via custom augmentations)

Troubleshooting
Insufficient Memory / OOM Errors:
Lower imgsz if you run out of memory during training or inference.
Use stream=True for video inference to prevent accumulating detection results.
Poor Small Object Detection:
Increase imgsz to 1280 or higher.
Adjust anchor boxes or enable auto-anchor in YOLO.
Further tune augmentation parameters (scale, mosaic).
Dataset Path Issues:
Ensure train, val, and test paths in data.yaml are correct.
Check folder names and structure.