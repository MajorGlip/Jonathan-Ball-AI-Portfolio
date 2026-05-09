# Project 1 — Pharmaceutical Pill Image Classifier

## Problem Statement

Medication errors are one of the most common and preventable causes of patient harm in healthcare settings. Misidentification of pills — especially among visually similar tablets and capsules — poses a real risk to patient safety. As a pharmacy technician, I experienced firsthand how critical accurate pill identification is.

This project explores whether a Convolutional Neural Network (CNN) can be trained to reliably classify different pill types based on their visual appearance, with the goal of reducing the potential for medication errors.

## Approach

A CNN was built and trained using TensorFlow and Keras to classify images of pharmaceutical pills into their correct categories. The model learns visual features such as shape, color, imprint patterns, and size to distinguish between pill types.

**Model Architecture:**
- Input layer: 128×128 RGB images
- 3 Convolutional layers with ReLU activation and MaxPooling
- Dropout layer (0.5) for regularization
- Fully connected Dense layers
- Softmax output layer for multi-class classification

**Training Strategy:**
- Data augmentation applied (rotation, flipping, zoom, brightness shift) to improve generalization
- 80/20 train-validation split
- Optimizer: Adam | Loss: Categorical Crossentropy
- Early stopping to prevent overfitting

## Dataset

**NIH Pill Image Recognition Challenge Dataset**

A publicly available dataset provided by the U.S. National Library of Medicine, containing thousands of labeled pill images across multiple categories.

- Source: [https://pir.nlm.nih.gov/](https://pir.nlm.nih.gov/)
- Format: JPEG images organized by pill class in labeled folders
- Loading: Images are loaded using `tensorflow.keras.preprocessing.image.ImageDataGenerator`

> ⚠️ The dataset is not included in this repository due to size. Download instructions are available at the link above.

## Results

| Metric | Value |
|---|---|
| Training Accuracy | 93.4% |
| Validation Accuracy | 88.7% |
| Test Loss | 0.31 |

Key findings:
- The model performed strongly on pills with distinct shapes and imprints
- Some misclassifications occurred between similarly sized white tablets
- Data augmentation measurably improved validation accuracy by ~6%

## Key Findings

- Computer vision can serve as a meaningful second-check tool in pharmacy environments
- Visual similarity between pills (e.g., generic vs. brand name white tablets) remains the biggest challenge
- Future improvements could include incorporating pill imprint OCR for higher accuracy

## Technologies Used

- Python 3.x
- TensorFlow 2.x / Keras
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-learn (for evaluation metrics)

## How to Run

This notebook runs best in **Google Colab** (free GPU access):

1. Open `Pill_Image_Classifier.ipynb` in Google Colab
2. Download the NIH Pill dataset and upload to your Google Drive
3. Update the `dataset_path` variable in the notebook to point to your Drive folder
4. Run all cells in order (`Runtime > Run all`)

Alternatively, run locally with Python 3.8+ and install dependencies:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

## Project Structure

```
Project1-Pill-Classifier/
├── README.md
├── Pill_Image_Classifier.ipynb     # Main notebook (run all cells)
└── results/
    ├── accuracy_plot.png           # Training vs. validation accuracy curve
    ├── loss_plot.png               # Training vs. validation loss curve
    ├── confusion_matrix.png        # Confusion matrix across pill classes
    └── sample_predictions.png     # Sample correct and incorrect predictions
```
