from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uvicorn
from typing import List

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

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post('/api/upload')
async def upload_images(images: List[UploadFile] = File(...)):
    uploaded_images = []

    for image in images:
        file_path = os.path.join(UPLOAD_DIR, image.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            uploaded_images.append({"message": "File uploaded successfully!", "filename": image.filename})
        except Exception as e:
            uploaded_images.append({"error": str(e), "filename": image.filename})

    return uploaded_images
    


def start():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()