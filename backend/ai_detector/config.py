# config.py
"""
Configuration settings for AI Detector
"""

import torch
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class DataConfig:
    """Data configuration"""
    data_dir: str = "dataset_output_20x"
    max_samples_per_class: int = 100_000
    test_size: float = 0.1
    val_size: float = 0.1
    min_text_length: int = 20
    seed: int = 42


@dataclass
class ModelConfig:
    """Model configuration"""
    # Choose one:
    # - "distilbert-base-uncased"     (fast, good)
    # - "roberta-base"                (better, slower)
    # - "microsoft/deberta-v3-small"  (best, slowest)
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    num_labels: int = 2
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration"""
    output_dir: str = "trained_model"
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100
    
    # Early stopping
    early_stopping_patience: int = 3
    
    # System
    fp16: bool = True  # Mixed precision (GPU only)
    num_workers: int = 0  # DataLoader workers (0 for Windows)
    
    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    """Main configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        # Create output directory
        os.makedirs(self.training.output_dir, exist_ok=True)
    
    def display(self):
        """Print configuration"""
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)
        print(f"\n[Data]")
        print(f"  Data directory: {self.data.data_dir}")
        print(f"  Max samples/class: {self.data.max_samples_per_class:,}")
        
        print(f"\n[Model]")
        print(f"  Model: {self.model.model_name}")
        print(f"  Max length: {self.model.max_length}")
        
        print(f"\n[Training]")
        print(f"  Device: {self.training.device}")
        print(f"  Batch size: {self.training.batch_size}")
        print(f"  Epochs: {self.training.num_epochs}")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  FP16: {self.training.fp16 and self.training.device == 'cuda'}")
        print(f"  Output: {self.training.output_dir}")
        print("="*60 + "\n")


# Default configuration instance
def get_config(**kwargs) -> Config:
    """Get configuration with optional overrides"""
    config = Config()
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config.data, key):
            setattr(config.data, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
    
    return config