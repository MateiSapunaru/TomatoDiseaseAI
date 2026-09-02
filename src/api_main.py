from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

from src import config
from src.model_utils import (
    load_trained_model,
    get_transforms,
    predict_top_k,
)
from src.gradcam_utils import (
    GradCAM,
    create_gradcam_overlay,
    pil_to_base64,
)


MODEL_TYPE = "adapted"

if MODEL_TYPE == "adapted" and config.ADAPTATION_MODEL_PATH.exists():
    model, idx_to_class = load_trained_model(config.ADAPTATION_MODEL_PATH)
else:
    model, idx_to_class = load_trained_model(config.MODEL_PATH)

class_names = [
    idx_to_class[i]
    for i in range(len(idx_to_class))
]

app = FastAPI(
    title="Tomato Leaf Disease AI",
    description="FastAPI backend for tomato leaf disease inference and Grad-CAM explainability.",
    version="1.0.0",
)


def read_image_from_upload(file_bytes):
    image = Image.open(io.BytesIO(file_bytes))
    image = image.convert("RGB")
    return image


def predict_image(image: Image.Image, top_k: int = 3):
    return predict_top_k(model, image, idx_to_class, top_k=top_k)


def generate_gradcam(image: Image.Image, class_index: int):
    transform = get_transforms(train=False)

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(config.DEVICE)

    gradcam = GradCAM(
        model=model,
        target_layer=model.layer4[-1].conv2,
    )

    cam = gradcam.generate(
        input_tensor=tensor,
        class_index=class_index,
    )

    gradcam.close()

    overlay = create_gradcam_overlay(image, cam)
    return overlay


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(config.DEVICE),
        "model_type": MODEL_TYPE,
        "classes": class_names,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        image = read_image_from_upload(file_bytes)

        predictions = predict_image(image, top_k=3)

        return {
            "predicted_class": predictions[0]["class_name"],
            "confidence": predictions[0]["confidence"],
            "top_3_predictions": predictions,
        }

    except Exception as error:
        return JSONResponse(
            status_code=400,
            content={
                "error": str(error),
            },
        )


@app.post("/predict-gradcam")
async def predict_gradcam(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        image = read_image_from_upload(file_bytes)

        predictions = predict_image(image, top_k=3)
        predicted_class = predictions[0]["class_name"]

        class_index = class_names.index(predicted_class)

        overlay = generate_gradcam(
            image=image,
            class_index=class_index,
        )

        encoded_overlay = pil_to_base64(overlay)

        return {
            "predicted_class": predicted_class,
            "confidence": predictions[0]["confidence"],
            "top_3_predictions": predictions,
            "gradcam_image_base64": encoded_overlay,
        }

    except Exception as error:
        return JSONResponse(
            status_code=400,
            content={
                "error": str(error),
            },
        )