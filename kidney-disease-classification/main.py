from kidney_disease_classification import logger

from kidney_disease_classification.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline
)
from kidney_disease_classification.pipeline.stage_02_prepare_base_model import (
    PrepareBaseModelTrainingPipeline
)



def run_stage(stage_name: str, pipeline_class):
    """
    Utility function to run a pipeline stage safely
    """
    try:
        logger.info("*******************")
        logger.info(f">>>>>> Stage: {stage_name} started <<<<<<")

        pipeline = pipeline_class()
        pipeline.main()

        logger.info(f">>>>>> Stage: {stage_name} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == "__main__":

    # ============================
    # Stage 01: Data Ingestion
    # ============================
    run_stage(
        stage_name="Data Ingestion",
        pipeline_class=DataIngestionTrainingPipeline
    )

    # ============================
    # Stage 02: Prepare Base Model
    # ============================
    run_stage(
        stage_name="Prepare Base Model",
        pipeline_class=PrepareBaseModelTrainingPipeline
    )

  
