from dataclasses import dataclass, field
from typing import List, Optional
import torch

@dataclass
class TrainingConfig:
    """
    Configuration for the Text Humanizer Training Pipeline
    """
    
    # Model Configuration
    model_name: str = "t5-small"  # Using T5 for distinct seq2seq nature
    max_source_length: int = 512
    max_target_length: int = 512
    
    # LoRA (PEFT) Configuration
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q", "v"]) # T5 specific modules
    
    # Training Hyperparameters
    output_dir: str = "models/humanizer_v1"
    learning_rate: float = 3e-4
    batch_size: int = 8
    num_epochs: int = 3
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: str = "fp16" if torch.cuda.is_available() else "no"
    seed: int = 42

    # Data
    train_file: str = "data/processed/train.json"
    val_file: str = "data/processed/val.json"
