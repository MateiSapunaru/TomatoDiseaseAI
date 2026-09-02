# 🍅 TomatoDiseaseAI

### Deep Learning-Based Tomato Leaf Disease Classification with Explainable AI and Real-World Robustness Evaluation

---

## Overview

**TomatoDiseaseAI** is a computer vision project for automatic **tomato leaf disease classification** using deep learning.

- The system is designed to classify tomato leaf images into multiple disease categories using a **transfer learning approach based on ResNet18**, while also providing:
- **interactive inference (single-frame webcam capture)**
- **confidence scores**
- **Grad-CAM explainability**
- **webcam support**
- **image upload inference**
- **interactive evaluation dashboard**
- **FastAPI inference backend**
- **Streamlit frontend**

Unlike many plant disease classification projects that focus exclusively on benchmark dataset performance, this project explicitly investigates:

> **real-world robustness and model generalization on unseen external data**

The project explores the performance gap between clean dataset images and realistic field-like imagery, followed by a **domain adaptation stage** to improve robustness.

---

## Features

### 🌿 Tomato Disease Classification

Multi-class image classification for tomato leaf diseases.

The model predicts:

- disease class
- confidence score
- top-3 predictions

---

### 🔥 Explainable AI with Grad-CAM

The system includes **Grad-CAM explainability** to visualize:

> **which image regions influenced the model prediction**

This improves:

- transparency
- interpretability
- debugging
- trustworthiness

The generated heatmap highlights the regions of interest used by the neural network during inference.

---

### 📷 Interactive Inference (Upload & Webcam)

The application supports single-frame interactive inference via upload or webcam capture:

### Upload Image Inference

Users can upload an image and receive:

- prediction
- confidence score
- Grad-CAM visualization
- top-3 probabilities

### Webcam Inference

Users can perform single-frame inference using a webcam (Streamlit's `camera_input`). Note: this captures individual frames for interactive testing, not a continuous live video stream.

This allows quick testing under more realistic conditions.

---

### 📊 Evaluation Dashboard

The project includes a built-in evaluation interface displaying:

- confusion matrices
- ROC curves
- per-class F1 scores
- precision
- recall
- classification reports
- evaluation summaries

---

### 🌍 Real-World Robustness Evaluation

A major objective of this project is:

> **evaluating how well the model generalizes outside benchmark datasets**

The system was tested on:

### Standard dataset splits

- train
- validation
- test

### External unseen dataset

A completely separate **real-world tomato leaf dataset** was used to evaluate robustness.

This revealed an important computer vision challenge:

> **domain shift**

The project then introduces a **domain adaptation stage** to improve performance on real-world imagery.

---

## Supported Classes

The final model classifies **10 tomato leaf categories** (canonical mapping in `artifacts/class_to_idx.json`):

| Class |
|--------|
| Bacterial_spot |
| Early_blight |
| Late_blight |
| Leaf_Mold |
| Septoria_leaf_spot |
| Spider_mites Two-spotted_spider_mite |
| Target_Spot |
| Tomato_Yellow_Leaf_Curl_Virus |
| Tomato_mosaic_virus |
| healthy |

Note: some older evaluation reports in `artifacts/metrics/` reference an additional class `powdery_mildew` (11th class) from a previous run — the authoritative class mapping used at inference time is `artifacts/class_to_idx.json`.

---

## Model Architecture

The project uses:

### ResNet18 + Transfer Learning

A pretrained **ResNet18** backbone was selected because of:

- computational efficiency
- fast inference
- good feature extraction
- lightweight deployment suitability

The classifier head was modified for tomato disease classification.

---

### Training Strategy

The training pipeline consists of two stages.

---

### Stage 1 — Base Training

Initial model training performed on the primary tomato disease dataset.

Techniques used:

- transfer learning
- data augmentation
- weighted loss
- validation monitoring

---

### Stage 2 — Domain Adaptation

Fine-tuning on **external real-world unseen images**.

The goal of this stage is to improve:

- robustness
- generalization
- field-like performance

This stage was introduced after observing performance degradation on realistic imagery.

---

## Data Augmentation

To improve robustness, several augmentation techniques are applied during training.

### Geometric Augmentations

- random rotation
- random horizontal flip
- random resized crop
- perspective distortion

### Photometric Augmentations

- brightness variation
- contrast variation
- saturation variation
- hue shifts

### Blur Augmentation

- Gaussian blur

These augmentations simulate:

- camera movement
- imperfect framing
- variable lighting
- image noise
- perspective changes

---

# Example Application Interface

---

## Main Inference Interface

The application allows users to:

- upload images
- run disease detection
- visualize Grad-CAM
- inspect prediction confidence


<img width="1600" height="802" alt="image" src="https://github.com/user-attachments/assets/f70d6fe7-4241-4ff6-bef7-bf75baa4589a" />



---

## Prediction Results

The prediction interface displays:

- predicted class
- confidence score
- top-3 predictions


<img width="1600" height="798" alt="image" src="https://github.com/user-attachments/assets/af7ef170-6f02-4ac1-9478-bfb0fbbc3280" />



---

## Grad-CAM Explainability

Grad-CAM highlights the image regions used by the model.


<img width="973" height="1040" alt="image" src="https://github.com/user-attachments/assets/7eaebb49-d1af-4edc-b6b6-21e439efc717" />





# Evaluation Results

---

## Evaluation Dashboard

The Streamlit dashboard includes visualization of model performance.


<img width="1600" height="778" alt="image" src="https://github.com/user-attachments/assets/affcc14d-8953-4d68-8ee8-5c9f08b7418f" />



---

## Confusion Matrix

Visual representation of classification performance.


<img width="1108" height="980" alt="image" src="https://github.com/user-attachments/assets/67dab765-fdff-46db-9c22-2ba1c7fe9a2f" />



---

## ROC Curves

One-vs-rest ROC curves for each disease category.


<img width="762" height="724" alt="image" src="https://github.com/user-attachments/assets/b4854e16-0484-4ba9-807e-ceeebbb2d4a6" />



---

## Per-Class Metrics

Performance breakdown per disease class.

Metrics include:

- F1 Score
- Precision
- Recall


<img width="751" height="487" alt="image" src="https://github.com/user-attachments/assets/d57287bc-baf6-4fd1-8ab3-4e730323959d" />



---

## Project Structure

```text
TomatoDiseaseAI/
│
├── artifacts/
│   ├── metrics/
│   │   ├── val/
│   │   ├── test/
│   │   ├── real_world/
│   │   └── real_world_test/
│   │
│   ├── training_history.json
│   ├── real_world_adaptation_history.json
│   ├── class_to_idx.json
│   ├── tomato_resnet18.pth
│   └── tomato_resnet18_adapted.pth
│
├── dataset/
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── real_world_unseen_data/
│   ├── real_world_train/
│   └── real_world_test/
│
├── src/
│   ├── api_main.py
│   ├── streamlit_app.py
│   ├── train.py
│   ├── train_adaptation.py
│   ├── evaluate.py
│   ├── gradcam_utils.py
│   ├── model_utils.py
│   ├── create_real_world_split.py
│   └── config.py
│
├── requirements.txt
├── README.md
└── .gitignore
```



# Installation

## 1. Clone the repository

```bash
git clone https://github.com/MateiSapunaru/TomatoDiseaseAI.git

cd TomatoDiseaseAI
```

---

### Note on the dataset

`dataset/` is not tracked in this repository (it's excluded via `.gitignore` since it contains 30k+ images). To train or evaluate from scratch, populate `dataset/train`, `dataset/valid`, `dataset/test`, `dataset/real_world_train`, and `dataset/real_world_test` with class-subfolder image data matching the classes in `artifacts/class_to_idx.json`.

If you only want to run inference, this isn't needed — `artifacts/tomato_resnet18.pth` and `artifacts/tomato_resnet18_adapted.pth` are already included, so the FastAPI backend and Streamlit app work right after cloning.

---

## 2. Create virtual environment

```bash
python -m venv venv
```

---

## 3. Activate virtual environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / MacOS

```bash
source venv/bin/activate
```

---

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

---

## Step 1 — Start FastAPI Backend

```bash
uvicorn src.api_main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Step 2 — Launch Streamlit App

```bash
streamlit run src/streamlit_app.py
```

---

# API Endpoints

---

## Health Check

```http
GET /health
```

Returns:

- available classes
- inference device
- model status

---

## Prediction Endpoint

```http
POST /predict
```

Returns:

- predicted class
- confidence score
- top-3 predictions

---

## Prediction + Grad-CAM

```http
POST /predict-gradcam
```

Returns:

- prediction
- confidence score
- Grad-CAM visualization
- top-3 predictions

---

# Evaluation Outputs

The evaluation pipeline automatically generates:

### Confusion Matrix

Measures class-level classification performance.

---

### ROC Curves

One-vs-rest ROC curves.

---

### F1 Score per Class

Balanced performance metric.

---

### Precision & Recall

Class-specific performance analysis.

---

### Classification Report

Detailed summary of:

- precision
- recall
- F1 score
- support

---

# Technologies Used

### Machine Learning & Deep Learning

- Python
- PyTorch
- TorchVision
- NumPy
- Scikit-Learn

### Explainable AI

- Grad-CAM

### Backend

- FastAPI
- Uvicorn

### Frontend

- Streamlit

### Computer Vision

- OpenCV
- Pillow

### Visualization

- Matplotlib

---

# Key Contributions

This project extends beyond a conventional tomato disease classifier by introducing:

### Real-Time Inference

Interactive prediction using:

- image upload
- webcam inference

### Explainability

Grad-CAM visualization for model interpretability.

### Real-World Robustness Evaluation

Evaluation on completely unseen external data.

### Domain Adaptation

Fine-tuning to improve real-world performance.

### Interactive Evaluation Interface

Integrated visualization of evaluation metrics and plots.

---

# Future Improvements

Potential future improvements include:

- video stream inference
- model quantization
- edge deployment optimization
- larger real-world datasets
- improved domain adaptation strategies
- multi-leaf scene understanding
- object detection-based disease localization

---

## Author

**Matei Săpunaru**

Machine Learning • Computer Vision • Artificial Intelligence
