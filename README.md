# Luggage Safety Detection

A computer vision and machine learning project for luggage safety classification and suspicious-region segmentation.

## Overview

This project combines traditional machine learning, image preprocessing, feature extraction, and deep learning to classify luggage images as safe or unsafe. For images classified as unsafe, a segmentation model generates a mask highlighting the detected region.

## Methodology

### 1. Image Preprocessing
- Grayscale conversion and noise reduction
- Frequency-domain filtering for periodic noise removal
- Wiener filtering to improve image quality

### 2. Feature Extraction
- **Histogram of Oriented Gradients (HOG):** captures image shape and edge information
- **Local Binary Patterns (LBP):** describes local texture patterns
- **Oriented FAST and Rotated BRIEF (ORB):** extracts local visual features

The extracted features are combined and used for classification.

### 3. Classification
A trained machine learning model predicts whether each luggage image is classified as safe or unsafe.

### 4. Segmentation
For images classified as unsafe, a trained U-Net model with a MobileNetV2 backbone generates a segmentation mask to highlight the relevant image region.

## Project Structure

```text
Luggage-Safety-Detection/
├── src/
│   └── luggage_safety_detection.py
└── README.md
```

## Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-image
- SciPy
- Scikit-learn / Joblib
- TensorFlow / Keras
- Matplotlib

## Notes

The project demonstrates a combined approach to image classification and segmentation for luggage safety analysis. Predictions depend on the trained models and the input images provided.
