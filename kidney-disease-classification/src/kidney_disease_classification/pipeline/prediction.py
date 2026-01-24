import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path


class PredictionPipeline:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.model_path = Path("artifacts/training/model.h5")

    def predict(self):
        # Load trained model
        model = tf.keras.models.load_model(self.model_path)

        # Preprocess image
        img = image.load_img(self.image_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        probs = model.predict(img_array)[0]

        normal_prob = float(probs[0]) * 100
        tumor_prob = float(probs[1]) * 100

        # Decision
        if tumor_prob > normal_prob:
            prediction = "Tumor"
            confidence = tumor_prob
        else:
            prediction = "Normal"
            confidence = normal_prob

        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "normal_prob": round(normal_prob, 2),
            "tumor_prob": round(tumor_prob, 2)
        }
