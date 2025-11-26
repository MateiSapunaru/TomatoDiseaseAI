# Tomato Leaf Disease Classifier

This project is a small end-to-end machine learning system that detects tomato leaf diseases from images. It includes a trained deep learning model, evaluation tools, and a simple web interface where users can upload a photo and view the model’s prediction.

The goal was to build something practical, lightweight, and deployable — not just a notebook model.

---

## What the Project Does

* Trains a **ResNet18** model (fine-tuned from ImageNet) to classify tomato leaf diseases into **11 categories**.
* Automatically generates evaluation metrics:

  * Confusion matrix
  * ROC curves
  * Per-class precision, recall, and F1 scores
  * Training loss and accuracy curves
* Saves all generated plots and reports locally for analysis.
* Provides a simple **FastAPI web app** where users can upload an image and get the predicted disease and confidence score.

---

## Why I Built It

I wanted a clean project that shows I can:

* Build and structure a deep learning project from training → evaluation → deployment
* Work with PyTorch and image preprocessing
* Create a usable API for real-world inference
* Connect ML models to a minimal but functional web interface

The goal was to keep things simple but complete — something that actually works end-to-end.

---

## Project Structure

```
src/
    train.py            # training script
    model_utils.py      # model, transforms, prediction utilities
    evaluate_metrics.py # generates visualizations + reports
    api_main.py         # FastAPI inference server + small HTML page
    config.py

artifacts/
    metrics/            # saved plots and evaluation outputs
```

The training script saves the best model and the class index mapping.
The API loads them once at startup and uses them for predictions.

---

## How to Run

### 1. Train the Model

```
python -m src.train
```

### 2. Generate Metrics and Visualizations

```
python -m src.evaluate_metrics
```

### 3. Start the Web App

```
uvicorn src.api_main:app --reload
```

Then open:

```
http://127.0.0.1:8000
```

Upload an image → get the predicted disease.

---

If you need a more formal or shorter version for a CV, I can generate that too.
