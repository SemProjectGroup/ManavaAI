"""
AI Content Detector Inference Module.
Provides a simple interface for detecting AI-generated content.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIDetector:
    """
    AI Content Detector for identifying AI-generated text.
    
    Usage:
        detector = AIDetector()
        result = detector.detect("Your text here")
        print(f"AI Probability: {result['ai_probability']:.2%}")
    """
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Config] = None):
        """
        Initialize the AI Detector.
        
        Args:
            model_path: Path to the trained model directory
            config: Configuration object
        """
        self.config = config or Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or self.config.MODEL_SAVE_DIR
        
        # Load model and tokenizer
        self._load_model()
        
        logger.info(f"AI Detector initialized on {self.device}")
    
    def _load_model(self):
        """Load the trained model and tokenizer."""
        from transformers import AutoTokenizer, AutoModel
        from models.detector import AIDetectorModel
        
        # Find model directory
        possible_paths = [
            self.model_path,
            os.path.join(self.config.BASE_DIR, "model_files", "saved"),
            os.path.join(self.config.BASE_DIR, "models", "saved"),
        ]
        
        model_dir = None
        for p in possible_paths:
            if p and os.path.exists(p):
                model_dir = p
                break
        
        if model_dir is None:
            model_dir = self.model_path
            os.makedirs(model_dir, exist_ok=True)
        
        # Load config
        config_path = os.path.join(model_dir, 'detector_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            model_name = model_config.get('model_name', self.config.DETECTOR_MODEL_NAME)
            dropout_rate = model_config.get('dropout_rate', self.config.DROPOUT_RATE)
        else:
            model_name = self.config.DETECTOR_MODEL_NAME
            dropout_rate = self.config.DROPOUT_RATE
        
        # Load tokenizer
        tokenizer_path = os.path.join(model_dir, 'detector_tokenizer')
        if os.path.exists(tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info(f"Loaded tokenizer from {tokenizer_path}")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Loaded default tokenizer from {model_name}")
        
        # Load model
        self.model = AIDetectorModel(
            model_name=model_name,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        model_file = os.path.join(model_dir, 'detector_model.pt')
        if os.path.exists(model_file):
            self.model.load_state_dict(
                torch.load(model_file, map_location=self.device)
            )
            logger.info(f"Loaded trained model from {model_file}")
        else:
            logger.warning("No trained model found. Using base model (results may be less accurate).")
        
        self.model.eval()
    
    def detect(
        self, 
        text: str, 
        include_analysis: bool = True,
        include_sentences: bool = False
    ) -> Dict:
        """
        Detect if text is AI-generated.
        
        Args:
            text: Text to analyze
            include_analysis: Include linguistic analysis
            include_sentences: Include per-sentence analysis
        
        Returns:
            Dictionary with detection results
        """
        # Clean text
        cleaned_text = text.strip()
        
        if len(cleaned_text.split()) < 10:
            return {
                'ai_probability': 0.5,
                'human_probability': 0.5,
                'verdict': 'uncertain',
                'confidence': 'low',
                'message': 'Text is too short for reliable detection (minimum 10 words)',
                'word_count': len(cleaned_text.split()),
                'char_count': len(cleaned_text)
            }
        
        # Tokenize
        encoding = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=self.config.MAX_LENGTH,
            return_tensors='pt'
        ).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(
                encoding['input_ids'],
                encoding['attention_mask']
            )
            ai_probability = torch.sigmoid(logits).item()
        
        human_probability = 1 - ai_probability
        
        # Determine verdict and confidence
        if ai_probability >= 0.85:
            verdict = 'ai_generated'
            confidence = 'high'
        elif ai_probability >= 0.65:
            verdict = 'likely_ai'
            confidence = 'medium'
        elif ai_probability >= 0.35:
            verdict = 'mixed'
            confidence = 'low'
        elif ai_probability >= 0.15:
            verdict = 'likely_human'
            confidence = 'medium'
        else:
            verdict = 'human_written'
            confidence = 'high'
        
        result = {
            'ai_probability': ai_probability,
            'human_probability': human_probability,
            'ai_percentage': round(ai_probability * 100, 1),
            'human_percentage': round(human_probability * 100, 1),
            'verdict': verdict,
            'confidence': confidence,
            'word_count': len(cleaned_text.split()),
            'char_count': len(cleaned_text)
        }
        
        # Add linguistic analysis
        if include_analysis:
            result['linguistic_analysis'] = self._analyze_linguistics(cleaned_text)
        
        # Add per-sentence analysis
        if include_sentences:
            result['sentence_analysis'] = self._analyze_sentences(cleaned_text)
        
        return result
    
    def _analyze_linguistics(self, text: str) -> Dict:
        """Basic linguistic analysis."""
        words = text.lower().split()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s for s in sentences if len(s.strip()) > 0]
        
        # Basic metrics
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Vocabulary richness (type-token ratio)
        unique_words = set(words)
        vocabulary_richness = len(unique_words) / max(len(words), 1)
        
        # Sentence length variance (burstiness)
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) > 1:
            burstiness = np.std(sentence_lengths) / max(np.mean(sentence_lengths), 1)
        else:
            burstiness = 0
        
        # Simple perplexity estimate based on word frequency
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        
        entropy = 0
        for count in word_freq.values():
            prob = count / len(words)
            if prob > 0:
                entropy -= prob * np.log2(prob)
        perplexity = 2 ** entropy if entropy > 0 else 1
        
        return {
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'vocabulary_richness': round(vocabulary_richness, 4),
            'burstiness': round(burstiness, 4),
            'perplexity': round(perplexity, 2),
            'sentence_count': len(sentences),
            'unique_words': len(unique_words)
        }
    
    def _analyze_sentences(self, text: str) -> List[Dict]:
        """Analyze each sentence individually."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        results = []
        for sentence in sentences[:10]:  # Limit to 10 sentences
            encoding = self.tokenizer(
                sentence,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                logits = self.model(
                    encoding['input_ids'],
                    encoding['attention_mask']
                )
                prob = torch.sigmoid(logits).item()
            
            results.append({
                'sentence': sentence,
                'ai_probability': prob,
                'label': 'ai' if prob > 0.5 else 'human'
            })
        
        return results
    
    def detect_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        """Detect AI content in multiple texts."""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.config.MAX_LENGTH,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                logits = self.model(
                    encodings['input_ids'],
                    encodings['attention_mask']
                )
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            if isinstance(probs, np.floating):
                probs = [float(probs)]
            else:
                probs = probs.tolist()
            
            for text, prob in zip(batch_texts, probs):
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'ai_probability': float(prob),
                    'verdict': 'ai_generated' if prob > 0.5 else 'human_written'
                })
        
        return results
    
    def get_detailed_report(self, text: str) -> str:
        """Generate a detailed human-readable report."""
        result = self.detect(text, include_analysis=True, include_sentences=True)
        
        report = []
        report.append("=" * 60)
        report.append("AI CONTENT DETECTION REPORT")
        report.append("=" * 60)
        report.append("")
        report.append(f"📊 OVERALL SCORE")
        report.append(f"   AI Probability:    {result['ai_percentage']}%")
        report.append(f"   Human Probability: {result['human_percentage']}%")
        report.append(f"   Verdict:           {result['verdict'].replace('_', ' ').title()}")
        report.append(f"   Confidence:        {result['confidence'].title()}")
        report.append("")
        report.append(f"📝 TEXT STATISTICS")
        report.append(f"   Word Count:  {result['word_count']}")
        report.append(f"   Char Count:  {result['char_count']}")
        report.append("")
        
        if 'linguistic_analysis' in result:
            analysis = result['linguistic_analysis']
            report.append("🔍 LINGUISTIC ANALYSIS")
            report.append(f"   Perplexity:         {analysis.get('perplexity', 'N/A')}")
            report.append(f"   Burstiness:         {analysis.get('burstiness', 'N/A')}")
            report.append(f"   Vocabulary Richness: {analysis.get('vocabulary_richness', 'N/A')}")
            report.append(f"   Avg Sentence Length: {analysis.get('avg_sentence_length', 'N/A')}")
            report.append("")
        
        if 'sentence_analysis' in result and result['sentence_analysis']:
            report.append("📑 SENTENCE-LEVEL ANALYSIS")
            ai_sentences = sum(1 for s in result['sentence_analysis'] if s['label'] == 'ai')
            total_sentences = len(result['sentence_analysis'])
            report.append(f"   AI Sentences:    {ai_sentences}/{total_sentences}")
            report.append(f"   Human Sentences: {total_sentences - ai_sentences}/{total_sentences}")
            report.append("")
            
            sorted_sentences = sorted(
                result['sentence_analysis'], 
                key=lambda x: x['ai_probability'], 
                reverse=True
            )
            report.append("   Most AI-like sentences:")
            for i, sent in enumerate(sorted_sentences[:3]):
                report.append(f"   {i+1}. [{sent['ai_probability']:.0%}] {sent['sentence'][:60]}...")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


# For backwards compatibility and direct testing
def main():
    """Test the detector."""
    detector = AIDetector()
    
    test_text = """
    Cats are one of the most popular domestic animals in the world. 
    They have lived alongside humans for thousands of years and are 
    known for their independence, intelligence, and graceful behavior.
    """
    
    print("Testing AI Detector...")
    result = detector.detect(test_text)
    print(f"AI Probability: {result['ai_probability']:.2%}")
    print(f"Verdict: {result['verdict']}")


if __name__ == '__main__':
    main()