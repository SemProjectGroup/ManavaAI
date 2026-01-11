# model.py
"""
AI Detector Model Definition
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig
)

from config import Config


class AIDetectorModel(nn.Module):
    """AI Text Detection Model"""
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.config = config
        
        # Load pretrained model
        model_config = AutoConfig.from_pretrained(
            config.model.model_name,
            num_labels=config.model.num_labels,
            hidden_dropout_prob=config.model.dropout,
            attention_probs_dropout_prob=config.model.dropout
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model.model_name,
            config=model_config
        )
        
        # Label mappings
        self.id2label = {0: "Human", 1: "AI"}
        self.label2id = {"Human": 0, "AI": 1}
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits
        }
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get prediction probabilities"""
        
        self.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            probs = F.softmax(outputs.logits, dim=-1)
        
        return probs


class AIDetector:
    """
    High-level AI Detector interface
    Handles model loading, saving, and inference
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize detector
        
        Args:
            config: Configuration (for training)
            model_path: Path to saved model (for inference)
        """
        self.config = config
        self.device = config.training.device if config else "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_path and os.path.exists(model_path):
            self._load_from_path(model_path)
        elif config:
            self._initialize_new(config)
        else:
            raise ValueError("Must provide either config or model_path")
    
    def _initialize_new(self, config: Config):
        """Initialize new model from config"""
        print(f"\nInitializing model: {config.model.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        self.model = AIDetectorModel(config)
        self.model.to(self.device)
        
        self.max_length = config.model.max_length
        self.id2label = {0: "Human", 1: "AI"}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def _load_from_path(self, model_path: str):
        """Load saved model"""
        print(f"\nLoading model from: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load config
        config_path = os.path.join(model_path, 'detector_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            self.max_length = saved_config.get('max_length', 256)
            self.id2label = saved_config.get('id2label', {0: "Human", 1: "AI"})
        else:
            self.max_length = 256
            self.id2label = {0: "Human", 1: "AI"}
        
        print("Model loaded successfully!")
    
    def save(self, path: str):
        """Save model and tokenizer"""
        os.makedirs(path, exist_ok=True)
        
        # Save model
        if hasattr(self.model, 'model'):
            self.model.model.save_pretrained(path)
        else:
            self.model.save_pretrained(path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save custom config
        config_data = {
            'max_length': self.max_length,
            'id2label': self.id2label,
            'model_name': self.config.model.model_name if self.config else 'unknown',
            'created': datetime.now().isoformat()
        }
        
        with open(os.path.join(path, 'detector_config.json'), 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Model saved to: {path}")
    
    def predict(self, text: str) -> Dict:
        """
        Predict if text is AI-generated
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with prediction results
        """
        self.model.eval()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            if hasattr(self.model, 'model'):
                outputs = self.model.model(**inputs)
            else:
                outputs = self.model(**inputs)
            
            probs = F.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()
        
        human_prob = probs[0][0].item()
        ai_prob = probs[0][1].item()
        
        return {
            'prediction': self.id2label[prediction],
            'is_ai': prediction == 1,
            'confidence': confidence,
            'ai_probability': ai_prob,
            'human_probability': human_prob,
            'ai_percentage': round(ai_prob * 100, 2)
        }
    
    def predict_batch(self, texts: list, batch_size: int = 32) -> list:
        """Predict for multiple texts"""
        self.model.eval()
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                if hasattr(self.model, 'model'):
                    outputs = self.model.model(**inputs)
                else:
                    outputs = self.model(**inputs)
                
                probs = F.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1)
            
            for j in range(len(batch)):
                pred = predictions[j].item()
                results.append({
                    'prediction': self.id2label[pred],
                    'is_ai': pred == 1,
                    'confidence': probs[j][pred].item(),
                    'ai_probability': probs[j][1].item(),
                    'human_probability': probs[j][0].item(),
                    'ai_percentage': round(probs[j][1].item() * 100, 2)
                })
        
        return results
    
    def get_model(self) -> nn.Module:
        """Get the underlying PyTorch model"""
        return self.model
    
    def train_mode(self):
        """Set model to training mode"""
        self.model.train()
    
    def eval_mode(self):
        """Set model to evaluation mode"""
        self.model.eval()