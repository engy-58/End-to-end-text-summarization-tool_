from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path
    unzip_dir: Path
    dataset_name: str

@dataclass(frozen=True)
class DataValidationConfig:
  root_dir: Path
  STATUS_FILE: str
  ALL_REQUIRED_FILES: list

@dataclass(frozen=True)
class DataTransformationConfig: 
  root_dir: Path
  data_path: Path
  tokenizer_name: Path

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    weight_decay: float
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_steps: float
    gradient_accumulation_steps: int