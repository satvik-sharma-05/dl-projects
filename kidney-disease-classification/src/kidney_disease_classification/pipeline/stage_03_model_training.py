from kidney_disease_classification.config.configuration import ConfigurationManager
from kidney_disease_classification.components.model_training import Training
from kidney_disease_classification import logger

STAGE_NAME = "Model Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()

        trainer = Training(config=training_config)
        trainer.get_base_model()
        trainer.train_valid_generator()
        trainer.train()


if __name__ == "__main__":
    try:
        logger.info("*******************")
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")

        pipeline = ModelTrainingPipeline()
        pipeline.main()

        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e
