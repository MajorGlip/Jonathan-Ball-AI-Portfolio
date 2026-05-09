# Project 3 — Lane Detection with Edge Analysis

## Problem Statement

Autonomous vehicles and driver assistance systems depend on the ability to accurately detect lane markings on roads. Without reliable lane detection, vehicles cannot maintain position or make safe steering decisions. This project investigates how classical computer vision techniques can be used to detect lane lines from dashcam-style road footage.

## Approach

This project uses a pipeline of classical OpenCV techniques to identify lane markings in road images and video frames:

**Processing Pipeline:**
1. **Grayscale Conversion** — Simplifies the image to a single channel for edge detection
2. **Gaussian Blur** — Reduces noise before edge detection
3. **Canny Edge Detection** — Detects sharp changes in intensity (edges) across the image
4. **Region of Interest (ROI) Masking** — Isolates the lower portion of the image where lane lines appear
5. **Hough Line Transform** — Identifies straight lines within the edge-detected image
6. **Line Averaging & Drawing** — Averages detected line segments into two smooth lane lines (left and right) drawn onto the original image

**Key Parameters:**
- Canny thresholds: low=50, high=150
- Hough parameters: rho=2, theta=π/180, threshold=15, minLineLength=40, maxLineGap=20
- ROI: Trapezoidal mask covering the bottom 60% of the frame

## Dataset

**Custom Road Image Dataset**

A collection of dashcam-style road images and short video clips captured under various lighting and road conditions (daytime, shadows, straight roads, slight curves).

- Format: JPEG images and MP4 video clips
- Included in the `sample_images/` folder of this repository (under 50MB total)
- No external download required

## Results

| Metric | Value |
|---|---|
| Lane Detection Rate (straight roads) | ~94% |
| Lane Detection Rate (curved roads) | ~71% |
| Average Processing Speed | ~45 FPS |

Key findings:
- The pipeline performed reliably on well-marked straight roads under good lighting
- Performance degraded on sharp curves, where straight-line assumptions break down
- Shadows and faded lane markings were the most common sources of failure

## Key Findings

- Classical CV techniques can achieve strong results for lane detection without any training data
- The Region of Interest mask is critical — without it, irrelevant edges dominate the output
- This project demonstrates the value of understanding fundamentals before moving to deep learning approaches (e.g., semantic segmentation with CNNs)

## Technologies Used

- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib

## How to Run

This notebook runs locally or in **Google Colab** — no GPU required.

```bash
pip install opencv-python numpy matplotlib
```

1. Open `Lane_Detection.ipynb`
2. Ensure the `sample_images/` folder is in the same directory
3. Run all cells in order

To process a custom video:
```python
cap = cv2.VideoCapture('your_video.mp4')
```

## Project Structure

```
Project3-Lane-Detection/
├── README.md
├── Lane_Detection.ipynb            # Main notebook
├── sample_images/                  # Test images (included)
│   ├── road1.jpg
│   ├── road2.jpg
│   └── road_curve.jpg
└── results/
    ├── lane_detected_sample1.jpg   # Output with lanes drawn
    ├── lane_detected_sample2.jpg
    ├── edge_detection_stages.png   # Step-by-step pipeline visualization
    └── roi_mask_example.png        # Region of interest illustration
```
