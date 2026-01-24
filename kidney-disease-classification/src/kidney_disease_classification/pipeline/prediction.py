import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image


class PredictionPipeline:
    def __init__(self, filename: str):
        self.filename = filename

    def predict(self):
        # Load trained model
        model_path = os.path.join("artifacts", "training", "model.h5")
        model = tf.keras.models.load_model(model_path)

        # Load & preprocess image
        img = image.load_img(self.filename, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # 🔴 VERY IMPORTANT
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)[0]

        normal_prob = float(preds[0])
        tumor_prob = float(preds[1])

        confidence = max(normal_prob, tumor_prob) * 100

        # Decision logic
        if confidence < 70:
            prediction = "Uncertain ⚠️"
        elif tumor_prob > normal_prob:
            prediction = "Tumor"
        else:
            prediction = "Normal"

        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "normal_prob": round(normal_prob * 100, 2),
            "tumor_prob": round(tumor_prob * 100, 2)
        }
