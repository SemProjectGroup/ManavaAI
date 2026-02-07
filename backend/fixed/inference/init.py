"""
Inference module for ManavAI.
Provides easy-to-use interfaces for AI detection and humanization.
"""

from .detector_inference import AIDetector
from .humanizer_inference import TextHumanizer

__all__ = ['AIDetector', 'TextHumanizer']