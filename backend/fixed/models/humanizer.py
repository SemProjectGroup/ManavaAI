"""
Text Humanizer Model.
Fine-tuned T5/BART for text paraphrasing that reduces AI detection scores.
"""

import torch
import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration,
    T5Config,
    BartForConditionalGeneration,
    BartConfig,
    AutoTokenizer
)
from typing import Optional, Dict, List, Union
import logging

logger = logging.getLogger(__name__)


class HumanizerModel(nn.Module):
    """
    Text Humanizer based on T5 or BART.
    
    This model paraphrases AI-generated text to make it sound more human-like
    while preserving the original meaning.
    """
    
    def __init__(
        self,
        model_name: str = "t5-small",
        max_length: int = 256,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the Humanizer model.
        
        Args:
            model_name: Pre-trained model name from HuggingFace
            max_length: Maximum sequence length
            device: Device to use (cuda/cpu)
        """
        super(HumanizerModel, self).__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine model type and load
        if 't5' in model_name.lower():
            self.model_type = 't5'
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        elif 'bart' in model_name.lower():
            self.model_type = 'bart'
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
        else:
            # Default to T5
            self.model_type = 't5'
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Style tokens (for multi-style humanization)
        self.style_tokens = {
            'casual': '<casual>',
            'formal': '<formal>',
            'academic': '<academic>',
            'creative': '<creative>',
            'simple': '<simple>'
        }
        
        # Intensity settings
        self.intensity_settings = {
            'light': {
                'num_beams': 3,
                'temperature': 0.7,
                'top_p': 0.9,
                'repetition_penalty': 1.1
            },
            'medium': {
                'num_beams': 4,
                'temperature': 0.8,
                'top_p': 0.85,
                'repetition_penalty': 1.2
            },
            'aggressive': {
                'num_beams': 5,
                'temperature': 0.9,
                'top_p': 0.8,
                'repetition_penalty': 1.3
            }
        }
        
        logger.info(f"Initialized HumanizerModel with {model_name}")
        logger.info(f"Model type: {self.model_type}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass for training.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target token IDs for training
        
        Returns:
            Model outputs including loss if labels provided
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate humanized text.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Generation parameters
        
        Returns:
            Generated token IDs
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def humanize(
        self,
        text: str,
        style: str = 'casual',
        intensity: str = 'medium',
        max_length: Optional[int] = None
    ) -> str:
        """
        Humanize a single text.
        
        Args:
            text: Input text to humanize
            style: Writing style (casual, formal, academic, creative, simple)
            intensity: Transformation intensity (light, medium, aggressive)
            max_length: Maximum output length
        
        Returns:
            Humanized text
        """
        self.eval()
        
        # Get intensity settings
        settings = self.intensity_settings.get(intensity, self.intensity_settings['medium'])
        
        # Prepare input with task prefix
        if self.model_type == 't5':
            input_text = f"humanize {style}: {text}"
        else:
            input_text = f"Paraphrase in {style} style: {text}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        ).to(self.device)
        
        # Generate
        max_len = max_length or min(len(text.split()) * 2, self.max_length)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_len,
                num_beams=settings['num_beams'],
                temperature=settings['temperature'],
                top_p=settings['top_p'],
                repetition_penalty=settings['repetition_penalty'],
                do_sample=True,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0
            )
        
        # Decode
        humanized = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return humanized
    
    def humanize_batch(
        self,
        texts: List[str],
        style: str = 'casual',
        intensity: str = 'medium',
        batch_size: int = 8
    ) -> List[str]:
        """
        Humanize multiple texts.
        
        Args:
            texts: List of input texts
            style: Writing style
            intensity: Transformation intensity
            batch_size: Batch size for processing
        
        Returns:
            List of humanized texts
        """
        self.eval()
        results = []
        
        settings = self.intensity_settings.get(intensity, self.intensity_settings['medium'])
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Prepare inputs
            if self.model_type == 't5':
                input_texts = [f"humanize {style}: {t}" for t in batch_texts]
            else:
                input_texts = [f"Paraphrase in {style} style: {t}" for t in batch_texts]
            
            # Tokenize
            inputs = self.tokenizer(
                input_texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=self.max_length,
                    num_beams=settings['num_beams'],
                    temperature=settings['temperature'],
                    do_sample=True,
                    early_stopping=True
                )
            
            # Decode
            for output in outputs:
                humanized = self.tokenizer.decode(output, skip_special_tokens=True)
                results.append(humanized)
        
        return results
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """
        Load a fine-tuned model from path.
        
        Args:
            path: Path to the saved model
            **kwargs: Additional arguments
        
        Returns:
            Loaded HumanizerModel
        """
        import os
        import json
        
        # Load config if exists
        config_path = os.path.join(path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            model_name = config.get('model_name', 't5-small')
        else:
            model_name = 't5-small'
        
        # Create instance
        instance = cls(model_name=model_name, **kwargs)
        
        # Load fine-tuned weights
        if 't5' in model_name.lower():
            instance.model = T5ForConditionalGeneration.from_pretrained(path)
        else:
            instance.model = BartForConditionalGeneration.from_pretrained(path)
        
        # Load tokenizer
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        
        return instance
    
    def save_pretrained(self, path: str):
        """
        Save model to path.
        
        Args:
            path: Path to save the model
        """
        import os
        import json
        
        os.makedirs(path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save config
        config = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'max_length': self.max_length
        }
        with open(os.path.join(path, 'humanizer_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {path}")


class ParaphraseModel(nn.Module):
    """
    Alternative paraphrase model using sequence-to-sequence architecture.
    Can be used as a lighter alternative to T5/BART.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super(ParaphraseModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(hidden_size, vocab_size)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        src_embed = self.pos_encoding(self.embedding(src))
        tgt_embed = self.pos_encoding(self.embedding(tgt))
        
        memory = self.encoder(src_embed, src_key_padding_mask=src_mask)
        output = self.decoder(tgt_embed, memory, tgt_key_padding_mask=tgt_mask)
        
        return self.output_projection(output)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)