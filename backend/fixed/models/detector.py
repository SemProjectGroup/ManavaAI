"""
AI Content Detector Model.
Fine-tuned DeBERTa/RoBERTa for binary classification (AI vs Human text).
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class AIDetectorModel(nn.Module):
    """
    AI Content Detector based on DeBERTa/RoBERTa.
    
    This model fine-tunes a pre-trained transformer for binary classification
    to detect whether text is AI-generated or human-written.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        num_labels: int = 1,
        dropout_rate: float = 0.1,
        use_linguistic_features: bool = False,
        num_linguistic_features: int = 10
    ):
        """
        Initialize the AI Detector model.
        
        Args:
            model_name: Pre-trained model name from HuggingFace
            num_labels: Number of output labels (1 for binary with BCE)
            dropout_rate: Dropout probability
            use_linguistic_features: Whether to use additional linguistic features
            num_linguistic_features: Number of linguistic features if used
        """
        super(AIDetectorModel, self).__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_linguistic_features = use_linguistic_features
        
        # Load pre-trained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Get hidden size from config
        self.hidden_size = self.config.hidden_size
        
        # Classification head
        classifier_input_size = self.hidden_size
        if use_linguistic_features:
            classifier_input_size += num_linguistic_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_input_size, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_labels)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized AIDetectorModel with {model_name}")
        logger.info(f"Hidden size: {self.hidden_size}, Use linguistic features: {use_linguistic_features}")
    
    def _init_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        linguistic_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            linguistic_features: Optional linguistic features [batch_size, num_features]
        
        Returns:
            Logits [batch_size, num_labels]
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Concatenate linguistic features if provided
        if self.use_linguistic_features and linguistic_features is not None:
            cls_output = torch.cat([cls_output, linguistic_features], dim=1)
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        linguistic_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            linguistic_features: Optional linguistic features
        
        Returns:
            Probabilities [batch_size, 1]
        """
        logits = self.forward(input_ids, attention_mask, linguistic_features)
        return torch.sigmoid(logits)
    
    def freeze_transformer(self):
        """Freeze transformer parameters (for feature extraction only)."""
        for param in self.transformer.parameters():
            param.requires_grad = False
        logger.info("Transformer parameters frozen")
    
    def unfreeze_transformer(self):
        """Unfreeze transformer parameters for fine-tuning."""
        for param in self.transformer.parameters():
            param.requires_grad = True
        logger.info("Transformer parameters unfrozen")
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }


class AIDetectorWithAttention(nn.Module):
    """
    AI Detector with attention pooling over all tokens.
    May provide better performance for longer texts.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        num_labels: int = 1,
        dropout_rate: float = 0.1
    ):
        super(AIDetectorWithAttention, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        self.hidden_size = self.config.hidden_size
        
        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_labels)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Compute attention weights
        attention_scores = self.attention(hidden_states).squeeze(-1)  # [batch, seq_len]
        attention_scores = attention_scores.masked_fill(
            attention_mask == 0, float('-inf')
        )
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch, seq_len]
        
        # Weighted sum of hidden states
        pooled = torch.bmm(
            attention_weights.unsqueeze(1), 
            hidden_states
        ).squeeze(1)  # [batch, hidden]
        
        return self.classifier(pooled)


class EnsembleDetector(nn.Module):
    """
    Ensemble of multiple detector models for improved accuracy.
    """
    
    def __init__(
        self,
        model_names: list = None,
        dropout_rate: float = 0.1
    ):
        super(EnsembleDetector, self).__init__()
        
        if model_names is None:
            model_names = [
                "microsoft/deberta-v3-small",
                "roberta-base"
            ]
        
        self.models = nn.ModuleList([
            AIDetectorModel(name, dropout_rate=dropout_rate)
            for name in model_names
        ])
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(
            torch.ones(len(model_names)) / len(model_names)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = []
        for model in self.models:
            out = model(input_ids, attention_mask)
            outputs.append(out)
        
        # Stack and weight
        stacked = torch.stack(outputs, dim=0)  # [num_models, batch, 1]
        weights = torch.softmax(self.ensemble_weights, dim=0)
        
        # Weighted average
        weighted_output = (stacked * weights.view(-1, 1, 1)).sum(dim=0)
        
        return weighted_output