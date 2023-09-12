import os
import urllib.request as request
from text_summarizer.loggings import logger
from text_summarizer.utils.common import get_size
from pathlib import Path
from text_summarizer.entity import DataIngestionConfig
from datasets import load_dataset


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def load_dataset(self):
        dataset = load_dataset(self.config.dataset_name)

        train_data = dataset['train']
        validation_data = dataset['validation']
        test_data = dataset['test']

        dataset.save_to_disk(self.config.unzip_dir)