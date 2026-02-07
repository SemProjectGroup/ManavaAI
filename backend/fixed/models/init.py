"""
Models module for ManavAI.
Contains the AI Detector and Text Humanizer model architectures.
"""

from .detector import AIDetectorModel
from .humanizer import HumanizerModel

__all__ = ['AIDetectorModel', 'HumanizerModel']