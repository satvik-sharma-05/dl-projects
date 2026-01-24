from kidney_disease_classification.config.configuration import ConfigurationManager
from kidney_disease_classification.components.model_evaluation import Evaluation
from kidney_disease_classification import logger

STAGE_NAME = "Model Evaluation with MLflow"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()

        evaluation = Evaluation(eval_config)
        evaluation.evaluate()
        evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info("*******************")
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")

        pipeline = ModelEvaluationPipeline()
        pipeline.main()

        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e
