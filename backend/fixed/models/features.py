"""
Linguistic Feature Extraction Module
Extracts perplexity, burstiness, vocabulary metrics, and other linguistic features
"""

import math
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class PerplexityCalculator:
    """
    Calculates perplexity using GPT-2 as the language model.
    Lower perplexity often indicates AI-generated text.
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Set pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    @torch.no_grad()
    def calculate_perplexity(self, text: str, stride: int = 512) -> float:
        """
        Calculate perplexity of text using sliding window approach.
        
        Args:
            text: Input text
            stride: Stride for sliding window
            
        Returns:
            Perplexity score
        """
        if not text.strip():
            return 0.0
            
        encodings = self.tokenizer(text, return_tensors="pt")
        max_length = self.model.config.n_positions
        
        nlls = []
        prev_end_loc = 0
        
        for begin_loc in range(0, encodings.input_ids.size(1), stride):
            end_loc = min(begin_loc + max_length, encodings.input_ids.size(1))
            trg_len = end_loc - prev_end_loc
            
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            outputs = self.model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
            
            nlls.append(neg_log_likelihood)
            
            prev_end_loc = end_loc
            if end_loc == encodings.input_ids.size(1):
                break
                
        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()
    
    @torch.no_grad()
    def calculate_sentence_perplexities(self, text: str) -> List[Tuple[str, float]]:
        """
        Calculate perplexity for each sentence in the text.
        
        Args:
            text: Input text
            
        Returns:
            List of (sentence, perplexity) tuples
        """
        sentences = sent_tokenize(text)
        results = []
        
        for sentence in sentences:
            if sentence.strip():
                ppl = self.calculate_perplexity(sentence)
                results.append((sentence, ppl))
                
        return results


class LinguisticFeatureExtractor:
    """
    Extracts various linguistic features from text that help distinguish
    AI-generated content from human-written content.
    """
    
    def __init__(self, perplexity_calculator: Optional[PerplexityCalculator] = None):
        self.perplexity_calculator = perplexity_calculator
        self.stop_words = set(stopwords.words('english'))
        
    def extract_all_features(self, text: str) -> Dict[str, float]:
        """
        Extract all linguistic features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        
        # Basic text statistics
        features.update(self._extract_basic_stats(text))
        
        # Vocabulary metrics
        features.update(self._extract_vocabulary_metrics(text))
        
        # Sentence-level features
        features.update(self._extract_sentence_features(text))
        
        # Burstiness features
        features.update(self._extract_burstiness_features(text))
        
        # POS tag features
        features.update(self._extract_pos_features(text))
        
        # Readability features
        features.update(self._extract_readability_features(text))
        
        # Perplexity (if calculator available)
        if self.perplexity_calculator:
            features['perplexity'] = self.perplexity_calculator.calculate_perplexity(text)
        
        return features
    
    def _extract_basic_stats(self, text: str) -> Dict[str, float]:
        """Extract basic text statistics."""
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        chars = len(text)
        
        return {
            'char_count': chars,
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
        }
    
    def _extract_vocabulary_metrics(self, text: str) -> Dict[str, float]:
        """Extract vocabulary diversity metrics."""
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha()]
        
        if not words:
            return {
                'type_token_ratio': 0,
                'hapax_ratio': 0,
                'unique_word_ratio': 0,
                'content_word_ratio': 0,
            }
        
        word_freq = Counter(words)
        unique_words = len(word_freq)
        total_words = len(words)
        
        # Type-Token Ratio (TTR)
        ttr = unique_words / total_words
        
        # Hapax Legomena Ratio (words appearing only once)
        hapax = sum(1 for count in word_freq.values() if count == 1)
        hapax_ratio = hapax / total_words
        
        # Content word ratio (non-stopwords)
        content_words = [w for w in words if w not in self.stop_words]
        content_ratio = len(content_words) / total_words if total_words > 0 else 0
        
        return {
            'type_token_ratio': ttr,
            'hapax_ratio': hapax_ratio,
            'unique_word_ratio': unique_words / total_words,
            'content_word_ratio': content_ratio,
        }
    
    def _extract_sentence_features(self, text: str) -> Dict[str, float]:
        """Extract sentence-level features."""
        sentences = sent_tokenize(text)
        
        if not sentences:
            return {
                'sentence_length_variance': 0,
                'sentence_length_std': 0,
                'short_sentence_ratio': 0,
                'long_sentence_ratio': 0,
            }
        
        sentence_lengths = [len(word_tokenize(s)) for s in sentences]
        
        variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
        std = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # Short sentences (< 10 words) and long sentences (> 30 words)
        short_count = sum(1 for l in sentence_lengths if l < 10)
        long_count = sum(1 for l in sentence_lengths if l > 30)
        
        return {
            'sentence_length_variance': variance,
            'sentence_length_std': std,
            'short_sentence_ratio': short_count / len(sentences),
            'long_sentence_ratio': long_count / len(sentences),
        }
    
    def _extract_burstiness_features(self, text: str) -> Dict[str, float]:
        """
        Extract burstiness features.
        Burstiness measures the variation in text patterns.
        AI text tends to be more uniform, while human text is more bursty.
        """
        sentences = sent_tokenize(text)
        
        if len(sentences) < 2:
            return {
                'burstiness_score': 0,
                'sentence_similarity_variance': 0,
            }
        
        # Calculate sentence lengths
        lengths = [len(word_tokenize(s)) for s in sentences]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        # Burstiness = (std - mean) / (std + mean)
        # Higher burstiness indicates more human-like variation
        if (std_length + mean_length) > 0:
            burstiness = (std_length - mean_length) / (std_length + mean_length)
        else:
            burstiness = 0
        
        # Calculate vocabulary overlap between consecutive sentences
        overlaps = []
        for i in range(len(sentences) - 1):
            words1 = set(word_tokenize(sentences[i].lower()))
            words2 = set(word_tokenize(sentences[i + 1].lower()))
            
            if words1 and words2:
                overlap = len(words1 & words2) / len(words1 | words2)
                overlaps.append(overlap)
        
        similarity_variance = np.var(overlaps) if overlaps else 0
        
        return {
            'burstiness_score': burstiness,
            'sentence_similarity_variance': similarity_variance,
        }
    
    def _extract_pos_features(self, text: str) -> Dict[str, float]:
        """Extract Part-of-Speech tag features."""
        words = word_tokenize(text)
        
        if not words:
            return {
                'noun_ratio': 0,
                'verb_ratio': 0,
                'adj_ratio': 0,
                'adv_ratio': 0,
                'pronoun_ratio': 0,
            }
        
        pos_tags = nltk.pos_tag(words)
        total = len(pos_tags)
        
        # Count POS categories
        nouns = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
        verbs = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
        adjs = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
        advs = sum(1 for _, tag in pos_tags if tag.startswith('RB'))
        pronouns = sum(1 for _, tag in pos_tags if tag.startswith('PRP'))
        
        return {
            'noun_ratio': nouns / total,
            'verb_ratio': verbs / total,
            'adj_ratio': adjs / total,
            'adv_ratio': advs / total,
            'pronoun_ratio': pronouns / total,
        }
    
    def _extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract readability metrics."""
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        if not words or not sentences:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'avg_syllables_per_word': 0,
            }
        
        # Count syllables
        def count_syllables(word):
            word = word.lower()
            count = 0
            vowels = "aeiouy"
            prev_char_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_char_was_vowel:
                    count += 1
                prev_char_was_vowel = is_vowel
            
            # Handle silent 'e'
            if word.endswith('e') and count > 1:
                count -= 1
                
            return max(1, count)
        
        total_syllables = sum(count_syllables(w) for w in words if w.isalpha())
        word_count = len([w for w in words if w.isalpha()])
        sentence_count = len(sentences)
        
        if word_count == 0 or sentence_count == 0:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'avg_syllables_per_word': 0,
            }
        
        avg_syllables = total_syllables / word_count
        avg_sentence_length = word_count / sentence_count
        
        # Flesch Reading Ease
        fre = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables
        
        # Flesch-Kincaid Grade Level
        fkg = 0.39 * avg_sentence_length + 11.8 * avg_syllables - 15.59
        
        return {
            'flesch_reading_ease': max(0, min(100, fre)),
            'flesch_kincaid_grade': max(0, fkg),
            'avg_syllables_per_word': avg_syllables,
        }
    
    def get_feature_vector(self, text: str) -> np.ndarray:
        """
        Get a feature vector for the text.
        
        Args:
            text: Input text
            
        Returns:
            Numpy array of features
        """
        features = self.extract_all_features(text)
        
        # Define consistent feature order
        feature_order = [
            'char_count', 'word_count', 'sentence_count',
            'avg_word_length', 'avg_sentence_length',
            'type_token_ratio', 'hapax_ratio', 'unique_word_ratio', 'content_word_ratio',
            'sentence_length_variance', 'sentence_length_std',
            'short_sentence_ratio', 'long_sentence_ratio',
            'burstiness_score', 'sentence_similarity_variance',
            'noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio', 'pronoun_ratio',
            'flesch_reading_ease', 'flesch_kincaid_grade', 'avg_syllables_per_word',
        ]
        
        if self.perplexity_calculator:
            feature_order.append('perplexity')
        
        vector = np.array([features.get(f, 0.0) for f in feature_order], dtype=np.float32)
        return vector
    
    @property
    def num_features(self) -> int:
        """Return the number of features extracted."""
        base_features = 23
        if self.perplexity_calculator:
            return base_features + 1
        return base_features


class FeatureNormalizer:
    """Normalizes linguistic features to have zero mean and unit variance."""
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, features: np.ndarray):
        """
        Fit the normalizer on training data.
        
        Args:
            features: Array of shape (n_samples, n_features)
        """
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        self.std[self.std == 0] = 1  # Avoid division by zero
        self.fitted = True
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted parameters.
        
        Args:
            features: Array of shape (n_samples, n_features) or (n_features,)
            
        Returns:
            Normalized features
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        return (features - self.mean) / self.std
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform(features)
    
    def save(self, path: str):
        """Save normalizer parameters."""
        np.savez(path, mean=self.mean, std=self.std)
    
    def load(self, path: str):
        """Load normalizer parameters."""
        data = np.load(path)
        self.mean = data['mean']
        self.std = data['std']
        self.fitted = True