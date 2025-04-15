from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import uvicorn
from typing import List
from utils.preprocessing import preprocess_image
from utils.models import load_models, predict_with_models
from fastapi.staticfiles import StaticFiles

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

models = load_models()

@app.post('/api/upload')
async def upload_images(images: List[UploadFile] = File(...)):
    results = []

    for image in images:
        try:
            contents = await image.read()
            img = Image.open(io.BytesIO(contents))
            processed_img = preprocess_image(img)
            predictions = predict_with_models(models, processed_img)
            results.append({
                "filename": image.filename,
                "predictions": predictions
            })

        except Exception as e:
            results.append({"filename": image.filename, "error": str(e)})

    return results

app.mount('/', StaticFiles(directory='../frontend/dist/',
          html=True), name='root')

def start():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()