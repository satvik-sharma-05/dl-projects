import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.tensorflow

from kidney_disease_classification.entity.config_entity import EvaluationConfig
from kidney_disease_classification.utils.common import save_json
from kidney_disease_classification import logger


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:2],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    def load_model(self):
        return tf.keras.models.load_model(self.config.path_of_model)

    def evaluate(self):
        logger.info("Starting model evaluation")

        self.model = self.load_model()
        self._valid_generator()

        self.score = self.model.evaluate(self.valid_generator, verbose=1)

        scores = {
            "loss": float(self.score[0]),
            "accuracy": float(self.score[1])
        }

        save_json(Path("scores.json"), scores)

    def log_into_mlflow(self):
        logger.info("Logging into MLflow")

        mlflow.set_tracking_uri(self.config.mlflow_uri)

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {
                    "loss": self.score[0],
                    "accuracy": self.score[1]
                }
            )

            # ✅ Windows-safe TensorFlow logging
            mlflow.tensorflow.log_model(
                self.model,
                artifact_path="model"
            )
