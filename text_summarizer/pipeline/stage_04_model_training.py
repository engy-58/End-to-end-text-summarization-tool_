from text_summarizer.config.configuration import ConfigurationManager
from text_summarizer.conponents.model_training import ModelTrainer


class ModelTrainingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()