"""
ManavAI Data Module
===================
This module contains all data collection, generation, and processing utilities.

Modules:
    - collect_human_text: Collect human-written text from various sources
    - generate_ai_text: Generate AI text using open-source models
    - preprocess: Clean and preprocess text data
    - dataset: PyTorch dataset classes
    - validate: Data validation utilities
    - build_dataset: Main script to build the complete dataset
"""

from .collect_human_text import HumanTextCollector
from .generate_ai_text import AITextGenerator
from .preprocess import TextPreprocessor
from .dataset import ManavAIDataset, create_dataloaders
from .validate import DataValidator
from .build_dataset import DatasetBuilder

__all__ = [
    "HumanTextCollector",
    "AITextGenerator", 
    "TextPreprocessor",
    "ManavAIDataset",
    "create_dataloaders",
    "DataValidator",
    "DatasetBuilder",
]