# src/api_main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import io

from src.model_utils import load_trained_model, predict_pil_image

# Load model once at startup
model, idx_to_class = load_trained_model()

app = FastAPI(
    title="Tomato Leaf Disease Demo",
    description="Upload an image of a tomato leaf and get a disease prediction.",
    version="0.1.0",
)


@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Simple HTML upload form.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tomato Leaf Disease Classifier</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 600px;
                margin: 40px auto;
                text-align: center;
            }
            .container {
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            input[type="file"] {
                margin: 20px 0;
            }
            button {
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                background-color: #2e7d32;
                color: white;
                font-size: 16px;
                cursor: pointer;
            }
            button:hover {
                background-color: #1b5e20;
            }
            .result {
                margin-top: 20px;
                font-size: 18px;
                font-weight: bold;
            }
            .prob {
                font-size: 14px;
                color: #555;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Tomato Leaf Disease Classifier</h1>
            <p>Upload an image of a tomato leaf and the model will predict the disease.</p>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required />
                <br/>
                <button type="submit">Predict</button>
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    """
    Receive an uploaded image, run it through the model,
    and return an HTML page with the prediction.
    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        predicted_class, confidence = predict_pil_image(model, image, idx_to_class)

        # Simple HTML result page
        result_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 600px;
                    margin: 40px auto;
                    text-align: center;
                }}
                .container {{
                    border: 1px solid #ddd;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                .result {{
                    margin-top: 20px;
                    font-size: 20px;
                    font-weight: bold;
                }}
                .prob {{
                    font-size: 14px;
                    color: #555;
                }}
                a {{
                    display: inline-block;
                    margin-top: 20px;
                    text-decoration: none;
                    color: #1976d2;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Prediction Result</h1>
                <p class="result">Predicted class: {predicted_class}</p>
                <p class="prob">Confidence: {confidence:.4f}</p>
                <a href="/">&#8592; Try another image</a>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=result_html)

    except Exception as e:
        error_html = f"""
        <html>
        <body>
            <h1>Error</h1>
            <p>Could not process the image: {str(e)}</p>
            <a href="/">&#8592; Back</a>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=400)
