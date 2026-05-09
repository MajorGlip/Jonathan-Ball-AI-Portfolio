# Project 2 — Real-Time Object Detection System

## Problem Statement

Object detection is one of the most impactful applications of Computer Vision, powering everything from self-driving cars to security systems.

## Approach

This project implements a real-time object detection pipeline using **YOLOv5** (You Only Look Once, version 5) — a single-stage object detector known for its speed and accuracy — integrated with **OpenCV** for image/video input and output.

**How it works:**
1. An image or video frame is passed to the YOLOv5 model
2. The model divides the image into a grid and predicts bounding boxes + class probabilities
3. Non-Maximum Suppression (NMS) is applied to remove duplicate detections
4. Detected objects are labeled and drawn onto the output frame using OpenCV

**Configuration:**
- Model: YOLOv5s (small variant — fast inference, good accuracy)
- Confidence threshold: 0.45
- IoU threshold (NMS): 0.5
- Input resolution: 640×640

## Dataset

**COCO Dataset (Common Objects in Context)**

The YOLOv5 model is pre-trained on COCO, which contains 80 object classes including people, vehicles, animals, and everyday objects.

- Source: [https://cocodataset.org/](https://cocodataset.org/)
- Classes: 80 (person, car, bicycle, dog, chair, etc.)
- Pre-trained weights are loaded directly — no manual download required

```python
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
```

## Results

| Metric | Value |
|---|---|
| mAP@0.5 (COCO val) | 56.8% |
| Inference Speed | ~28 FPS (CPU) |
| Objects Detected (test images) | 80 classes |

Key findings:
- YOLOv5s delivers real-time performance even on CPU
- Detection quality significantly improves with a confidence threshold between 0.4–0.5
- Smaller or partially occluded objects are the primary source of missed detections

## Key Findings

- Single-stage detectors like YOLO offer an excellent speed/accuracy trade-off for real-world applications
- Non-Maximum Suppression is essential for clean output — without it, objects are detected multiple times
- The model generalizes well across diverse lighting conditions and backgrounds

## Technologies Used

- Python 3.x
- YOLOv5 (via PyTorch Hub)
- PyTorch
- OpenCV (cv2)
- NumPy
- Matplotlib

## How to Run

Open `Object_Detection.ipynb` in **Google Colab** for easiest setup:

1. Open the notebook in Google Colab
2. Run the setup cell to install YOLOv5 dependencies
3. Upload a test image or provide a YouTube/video URL
4. Run all cells to see detections

Local setup:
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

## Project Structure

```
Project2-Object-Detection/
├── README.md
├── Object_Detection.ipynb          # Main notebook
└── results/
    ├── detected_images/
    │   ├── detection_sample1.jpg   # Annotated detection output
    │   └── detection_sample2.jpg
    └── performance_metrics.png     # Confidence score distribution
```
