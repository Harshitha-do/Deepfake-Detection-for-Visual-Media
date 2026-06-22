# Robust Deep Learning Arechitecture for Identifying Manipulated Visual Media

## Overview

This project is an AI-powered Deepfake Detection and Emotion Recognition System capable of analyzing images and videos to determine whether the content is real or manipulated. The system also performs facial emotion recognition on uploaded images.

The application consists of a React frontend for user interaction and a Flask backend that serves machine learning models for prediction. Deepfake detection is performed using a fine-tuned Xception neural network trained on real and fake facial datasets, while emotion recognition is powered by DeepFace.


## Features

* Deepfake Detection for Images
* Deepfake Detection for Videos
* Facial Emotion Recognition
* React-Based User Interface
* Flask REST API Backend
* Xception Transfer Learning Model
* Frame Extraction from Videos
* Automated Dataset Preparation
* Model Evaluation and Accuracy Reporting


## Project Architecture

Frontend (React.js)
↓
Flask API Backend
↓
Xception Deepfake Detection Model
↓
Prediction Results

Additional Module:
DeepFace Emotion Recognition


## Dataset

The project uses:

* Celeb-DF Dataset
* Real and Fake Face Dataset
* Extracted video frames for training

Dataset Preparation Pipeline:

1. Extract video frames
2. Create train, validation, and test datasets
3. Train Xception model
4. Evaluate model performance


## Technologies Used

### Frontend

* React.js
* Axios
* JavaScript

### Backend

* Flask
* Flask-CORS
* TensorFlow
* OpenCV
* NumPy
* DeepFace

### Machine Learning

* Xception CNN
* Transfer Learning
* ImageDataGenerator
* SGD Optimizer



## Installation

## Training Process

### Step 1: Extract Frames

```bash
py extract_all_frames.py
```

### Step 2: Prepare Dataset

```bash
py prepare_frame_dataset.py
```

### Step 3: Train Model

```bash
py train_combined.py
```

### Step 4: Evaluate Model

```bash
py evaluate_video_results.py
```

### Step 5: Run Backend

```bash
py app.py
```

### Step 6: Run Frontend

```bash
npm install
npm start
```

---

## Model

Base Model: Xception

Training Images:

* 18,500+ Training Frames
* 3,900+ Validation Frames

Input Size:

* 224 × 224

Optimizer:

* SGD

Loss Function:

* Binary Crossentropy

---

## Results

Training Accuracy: ~80%

Validation Accuracy: ~80%

The model successfully detects manipulated facial content and provides confidence scores for predictions.


## Future Enhancements

* Real-Time Webcam Deepfake Detection
* Explainable AI Heatmaps
* Advanced Video Forensics
* Audio Deepfake Detection
* Multi-Emotion Analysis
* PDF Report Generation
* Dashboard Analytics

