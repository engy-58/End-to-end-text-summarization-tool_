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