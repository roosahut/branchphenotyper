import os
import tensorflow as tf

MODEL_DIR = "../../models"
features = [
    "weeping", "antigravitropic", "main_trunks", "canopy_breadth",
    "primary_branches", "branch_density", "orientation"
]

def load_models():
    models = {}
    for feature in features:
        model_path = os.path.join(MODEL_DIR, f"model_{feature}.keras")
        models[feature] = tf.keras.models.load_model(model_path)
    return models

def predict_with_models(models, processed_img):
    predictions = {}
    for feature, model in models.items():
        prediction = model.predict(processed_img)
        predictions[feature] = float(prediction[0][0])
    return predictions