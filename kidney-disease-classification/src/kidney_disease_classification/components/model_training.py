import tensorflow as tf
from pathlib import Path
from kidney_disease_classification.entity.config_entity import TrainingConfig
from kidney_disease_classification import logger


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        logger.info("Loading updated base model")
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        logger.info("Preparing training & validation generators")

        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:2],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_gen.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_gen = valid_gen

        self.train_generator = train_gen.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        self.steps_per_epoch = max(
            1, self.train_generator.samples // self.train_generator.batch_size
        )
        self.validation_steps = max(
            1, self.valid_generator.samples // self.valid_generator.batch_size
        )

    def train(self):
        logger.info("Starting model training")

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps
        )

        self.save_model(self.config.trained_model_path)

    def save_model(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving trained model at: {path}")
        self.model.save(path)
