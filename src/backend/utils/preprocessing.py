import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

def preprocess_image(image: Image.Image, img_dim: int = 224) -> np.ndarray:
    image = image.convert("L")
    image = image.resize((img_dim, img_dim))

    img_array = img_to_array(image) / 255.0

    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array