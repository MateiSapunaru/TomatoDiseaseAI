import base64
import io
from pathlib import Path

import requests
import streamlit as st
from PIL import Image


API_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"


st.set_page_config(
    page_title="Tomato Disease AI",
    page_icon="🍅",
    layout="wide",
)


def call_predict_api(image: Image.Image, use_gradcam: bool = False):
    endpoint = "/predict-gradcam" if use_gradcam else "/predict"

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    files = {"file": ("image.png", buffer, "image/png")}

    response = requests.post(API_URL + endpoint, files=files, timeout=60)
    response.raise_for_status()

    return response.json()


def show_predictions(result):
    st.subheader("Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted class", result["predicted_class"])

    with col2:
        st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")

    st.subheader("Top 3 predictions")

    for prediction in result["top_3_predictions"]:
        confidence = prediction["confidence"]
        st.write(f"**{prediction['class_name']}** — {confidence * 100:.2f}%")
        st.progress(confidence)


def show_gradcam(result):
    if "gradcam_image_base64" not in result:
        return

    image_bytes = base64.b64decode(result["gradcam_image_base64"])
    gradcam_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    st.subheader("Grad-CAM Explainability")
    st.image(
        gradcam_image,
        caption="Highlighted regions used by the model for prediction",
        use_container_width=True,
    )


def inference_page():
    st.title("🍅 Tomato Leaf Disease AI")
    st.write(
        "Upload an image or use the webcam to classify tomato leaf diseases. "
        "The backend uses FastAPI for inference and Grad-CAM explainability."
    )

    with st.sidebar:
        st.header("Settings")

        input_mode = st.radio(
            "Input mode",
            ["Upload image", "Webcam"],
        )

        use_gradcam = st.checkbox("Generate Grad-CAM", value=True)

        st.divider()

        if st.button("Check API health"):
            try:
                response = requests.get(API_URL + "/health", timeout=10)
                response.raise_for_status()
                st.success("API is running.")
                st.json(response.json())
            except Exception as error:
                st.error(f"API error: {error}")

    image = None

    if input_mode == "Upload image":
        uploaded_file = st.file_uploader(
            "Choose a tomato leaf image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

    else:
        camera_image = st.camera_input("Take a picture")

        if camera_image is not None:
            image = Image.open(camera_image).convert("RGB")

    if image is None:
        st.info("Upload an image or take a webcam photo to start.")
        return

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Input image")
        st.image(image, use_container_width=True)

    with right_col:
        if st.button("Run inference", type="primary"):
            with st.spinner("Running model inference..."):
                try:
                    result = call_predict_api(
                        image=image,
                        use_gradcam=use_gradcam,
                    )

                    show_predictions(result)
                    show_gradcam(result)

                except Exception as error:
                    st.error(f"Inference failed: {error}")


def read_text_file(path: Path):
    if not path.exists():
        return None

    return path.read_text(encoding="utf-8")


def show_metric_summary(folder: Path):
    summary_path = folder / "metrics_summary.txt"
    content = read_text_file(summary_path)

    if content is None:
        st.warning("metrics_summary.txt not found.")
        return

    metrics = {}

    for line in content.splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)

        try:
            metrics[key.strip()] = float(value.strip())
        except ValueError:
            metrics[key.strip()] = value.strip()

    if not metrics:
        st.text(content)
        return

    cols = st.columns(len(metrics))

    for col, (key, value) in zip(cols, metrics.items()):
        with col:
            if isinstance(value, float):
                st.metric(key, f"{value:.4f}")
            else:
                st.metric(key, value)


def show_report(folder: Path):
    report_path = folder / "classification_report.txt"
    content = read_text_file(report_path)

    if content is None:
        st.warning("classification_report.txt not found.")
        return

    st.subheader("Classification Report")
    st.code(content, language="text")


def show_plot(folder: Path, filename: str, title: str):
    path = folder / filename

    if not path.exists():
        st.warning(f"{filename} not found.")
        return

    st.subheader(title)
    st.image(str(path), use_container_width=True)


def evaluation_page():
    st.title("📊 Evaluation Results")
    st.write(
        "This page displays the saved evaluation outputs generated by the evaluation scripts."
    )

    available_folders = [
        folder
        for folder in METRICS_DIR.iterdir()
        if folder.is_dir()
    ]

    if not available_folders:
        st.warning("No evaluation folders found in artifacts/metrics.")
        return

    folder_names = sorted([folder.name for folder in available_folders])

    selected_folder_name = st.selectbox(
        "Select evaluation folder",
        folder_names,
    )

    selected_folder = METRICS_DIR / selected_folder_name

    nested_folders = [
        folder
        for folder in selected_folder.iterdir()
        if folder.is_dir()
    ]

    if nested_folders:
        nested_names = sorted([folder.name for folder in nested_folders])

        selected_nested_name = st.selectbox(
            "Select model/result type",
            nested_names,
        )

        selected_folder = selected_folder / selected_nested_name

    st.info(f"Showing results from: `{selected_folder}`")

    show_metric_summary(selected_folder)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        show_plot(
            selected_folder,
            "confusion_matrix.png",
            "Confusion Matrix",
        )

        show_plot(
            selected_folder,
            "precision_per_class.png",
            "Per-Class Precision",
        )

    with col2:
        show_plot(
            selected_folder,
            "f1_per_class.png",
            "Per-Class F1 Scores",
        )

        show_plot(
            selected_folder,
            "recall_per_class.png",
            "Per-Class Recall",
        )

    show_plot(
        selected_folder,
        "roc_curves.png",
        "ROC Curves",
    )

    show_report(selected_folder)


def main():
    page = st.sidebar.radio(
        "Navigation",
        ["Inference", "Evaluation Results"],
    )

    if page == "Inference":
        inference_page()
    else:
        evaluation_page()


if __name__ == "__main__":
    main()