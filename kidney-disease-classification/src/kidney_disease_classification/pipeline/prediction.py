import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path


class PredictionPipeline:
    def __init__(self, image_path: str):
        self.image_path = image_path

        model_path = Path(__file__).resolve().parents[3] / "model" / "kidney_model.h5"
        self.model = tf.keras.models.load_model(model_path)

    def predict(self):
        img = image.load_img(self.image_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        probs = self.model.predict(img)[0]

        normal_prob = float(probs[0] * 100)
        tumor_prob = float(probs[1] * 100)

        prediction = "Tumor" if tumor_prob > normal_prob else "Normal"
        confidence = max(normal_prob, tumor_prob)

        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "normal_prob": round(normal_prob, 2),
            "tumor_prob": round(tumor_prob, 2),
        }
