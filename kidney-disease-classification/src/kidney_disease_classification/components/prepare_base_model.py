import tensorflow as tf
from pathlib import Path
from kidney_disease_classification.entity.config_entity import PrepareBaseModelConfig
from kidney_disease_classification import logger


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None

    def get_base_model(self):
        logger.info("Loading VGG16 base model")

        self.model = tf.keras.applications.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(self.config.base_model_path, self.model)
        logger.info(f"Base model saved at {self.config.base_model_path}")

    @staticmethod
    def _prepare_full_model(
        model: tf.keras.Model,
        classes: int,
        freeze_all: bool,
        freeze_till: int,
        learning_rate: float
    ) -> tf.keras.Model:

        logger.info("Preparing full model")

        if freeze_all:
            for layer in model.layers:
                layer.trainable = False

        elif freeze_till and freeze_till > 0:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        x = tf.keras.layers.Flatten()(model.output)
        output = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(x)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=output
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=learning_rate
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        logger.info("Updating base model with custom classifier head")

        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(self.config.updated_base_model_path, self.full_model)
        logger.info(
            f"Updated model saved at {self.config.updated_base_model_path}"
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
