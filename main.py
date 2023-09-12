from text_summarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from text_summarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from text_summarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from text_summarizer.loggings import logger

def run_pipeline(stage_name, pipeline):
    try:
        logger.info(f">>>>>> {stage_name} started <<<<<<")
        pipeline_instance = pipeline()
        pipeline_instance.main()
        logger.info(f">>>>>> {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == "__main__":
    stages = [
        ("Data Ingestion Stage", DataIngestionTrainingPipeline),
        ("Data Validation Stage", DataValidationTrainingPipeline),
        ("Data Transformation Stage", DataTransformationTrainingPipeline)
    ]

    for stage_name, pipeline_class in stages:
        run_pipeline(stage_name, pipeline_class)
