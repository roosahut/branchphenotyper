from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
import uvicorn
from typing import List
from utils.preprocessing import preprocess_image

app = FastAPI()

origins = [
    "http://localhost:5173",
    "localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_WEEPING_PATH = "../../models/model_weeping.keras"
model = tf.keras.models.load_model(MODEL_WEEPING_PATH)

@app.post('/api/upload')
async def upload_images(images: List[UploadFile] = File(...)):
    results = []

    for image in images:
        try:
            contents = await image.read()
            img = Image.open(io.BytesIO(contents))

            processed_img = preprocess_image(img)

            prediction = model.predict(processed_img)
            if prediction.ndim == 2 and prediction.shape[1] == 1:
                predicted_value = prediction[0][0]
            elif prediction.ndim == 1:
                predicted_value = prediction[0]
            else:
                predicted_value = prediction

            results.append({
                "filename": image.filename,
                "message": "File processed successfully!",
                "prediction": float(predicted_value)
            })

        except Exception as e:
            results.append({"filename": image.filename, "error": str(e)})

    return results


def start():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()