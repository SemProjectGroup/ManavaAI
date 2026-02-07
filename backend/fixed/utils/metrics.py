"""
Metrics and evaluation utilities for ManavAI.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging
import json
import os

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Track and aggregate metrics during training and evaluation.
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.epoch_metrics = []
    
    def update(self, name: str, value: float):
        """Add a metric value."""
        self.metrics[name].append(value)
    
    def update_batch(self, metrics_dict: Dict[str, float]):
        """Add multiple metric values."""
        for name, value in metrics_dict.items():
            self.update(name, value)
    
    def get_average(self, name: str) -> float:
        """Get average of a metric."""
        values = self.metrics.get(name, [])
        return np.mean(values) if values else 0.0
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get averages for all metrics."""
        return {name: self.get_average(name) for name in self.metrics}
    
    def end_epoch(self) -> Dict[str, float]:
        """End current epoch and store metrics."""
        averages = self.get_all_averages()
        self.epoch_metrics.append(averages)
        self.reset()
        return averages
    
    def reset(self):
        """Reset current metrics."""
        self.metrics = defaultdict(list)
    
    def get_history(self) -> List[Dict[str, float]]:
        """Get all epoch metrics."""
        return self.epoch_metrics
    
    def save(self, filepath: str):
        """Save metrics history to file."""
        with open(filepath, 'w') as f:
            json.dump(self.epoch_metrics, f, indent=2)
    
    def load(self, filepath: str):
        """Load metrics history from file."""
        with open(filepath, 'r') as f:
            self.epoch_metrics = json.load(f)


def calculate_bleu(
    references: List[str],
    hypotheses: List[str],
    max_n: int = 4,
    weights: Optional[Tuple[float, ...]] = None
) -> float:
    """
    Calculate BLEU score between references and hypotheses.
    
    Args:
        references: List of reference texts
        hypotheses: List of generated texts
        max_n: Maximum n-gram order
        weights: Weights for each n-gram order
    
    Returns:
        BLEU score
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        if weights is None:
            weights = tuple([1.0 / max_n] * max_n)
        
        smoothing = SmoothingFunction().method1
        scores = []
        
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = ref.lower().split()
            hyp_tokens = hyp.lower().split()
            
            if len(hyp_tokens) == 0:
                scores.append(0.0)
                continue
            
            score = sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                weights=weights,
                smoothing_function=smoothing
            )
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
        
    except ImportError:
        logger.warning("NLTK not installed. Using simple BLEU approximation.")
        return _simple_bleu(references, hypotheses)


def _simple_bleu(references: List[str], hypotheses: List[str]) -> float:
    """Simple BLEU approximation without NLTK."""
    scores = []
    
    for ref, hyp in zip(references, hypotheses):
        ref_words = set(ref.lower().split())
        hyp_words = set(hyp.lower().split())
        
        if len(hyp_words) == 0:
            scores.append(0.0)
            continue
        
        overlap = len(ref_words & hyp_words)
        precision = overlap / len(hyp_words)
        scores.append(precision)
    
    return np.mean(scores) if scores else 0.0


def calculate_rouge(
    references: List[str],
    hypotheses: List[str]
) -> Dict[str, float]:
    """
    Calculate ROUGE scores between references and hypotheses.
    
    Args:
        references: List of reference texts
        hypotheses: List of generated texts
    
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for ref, hyp in zip(references, hypotheses):
            scores = scorer.score(ref, hyp)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores) if rouge1_scores else 0.0,
            'rouge2': np.mean(rouge2_scores) if rouge2_scores else 0.0,
            'rougeL': np.mean(rougeL_scores) if rougeL_scores else 0.0
        }
        
    except ImportError:
        logger.warning("rouge-score not installed. Using simple ROUGE approximation.")
        return _simple_rouge(references, hypotheses)


def _simple_rouge(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Simple ROUGE approximation."""
    rouge1_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        ref_words = set(ref.lower().split())
        hyp_words = set(hyp.lower().split())
        
        if len(ref_words) == 0 or len(hyp_words) == 0:
            rouge1_scores.append(0.0)
            continue
        
        overlap = len(ref_words & hyp_words)
        precision = overlap / len(hyp_words)
        recall = overlap / len(ref_words)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        rouge1_scores.append(f1)
    
    avg_score = np.mean(rouge1_scores) if rouge1_scores else 0.0
    
    return {
        'rouge1': avg_score,
        'rouge2': avg_score * 0.8,  # Approximation
        'rougeL': avg_score * 0.9   # Approximation
    }


def calculate_perplexity(
    model,
    tokenizer,
    texts: List[str],
    device: str = 'cpu',
    max_length: int = 512
) -> float:
    """
    Calculate perplexity of texts using a language model.
    
    Args:
        model: Language model (GPT-2 or similar)
        tokenizer: Tokenizer for the model
        texts: List of texts
        device: Device to use
        max_length: Maximum sequence length
    
    Returns:
        Average perplexity
    """
    import torch
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            input_ids = encodings['input_ids']
            
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            
            total_loss += loss * input_ids.size(1)
            total_tokens += input_ids.size(1)
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)
    
    return perplexity


def calculate_diversity(texts: List[str]) -> Dict[str, float]:
    """
    Calculate diversity metrics for generated texts.
    
    Args:
        texts: List of generated texts
    
    Returns:
        Dictionary with diversity metrics
    """
    all_unigrams = []
    all_bigrams = []
    all_trigrams = []
    
    for text in texts:
        words = text.lower().split()
        all_unigrams.extend(words)
        all_bigrams.extend(zip(words[:-1], words[1:]))
        all_trigrams.extend(zip(words[:-2], words[1:-1], words[2:]))
    
    # Calculate distinct-n metrics
    distinct_1 = len(set(all_unigrams)) / max(len(all_unigrams), 1)
    distinct_2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    distinct_3 = len(set(all_trigrams)) / max(len(all_trigrams), 1)
    
    return {
        'distinct_1': distinct_1,
        'distinct_2': distinct_2,
        'distinct_3': distinct_3,
        'vocab_size': len(set(all_unigrams)),
        'total_tokens': len(all_unigrams)
    }


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two texts using word overlap.
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity score (0-1)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if len(words1) == 0 or len(words2) == 0:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union  # Jaccard similarity


def main():
    """Test metrics functions."""
    references = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence."
    ]
    
    hypotheses = [
        "A fast brown fox leaps over a lazy dog.",
        "Machine learning is part of AI technology."
    ]
    
    print("BLEU Score:", calculate_bleu(references, hypotheses))
    print("ROUGE Scores:", calculate_rouge(references, hypotheses))
    print("Diversity:", calculate_diversity(hypotheses))
    print("Similarity:", calculate_similarity(references[0], hypotheses[0]))


if __name__ == '__main__':
    main()  