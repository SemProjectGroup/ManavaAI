"""
Text Preprocessing Module
=========================
Cleans, normalizes, and preprocesses text data for model training.
Includes language detection, quality filtering, and deduplication.
"""

import re
import json
import hashlib
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter

from tqdm import tqdm
from loguru import logger

# Try to import optional dependencies
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    
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
        
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Using basic tokenization.")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not available. Language detection disabled.")

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config


class TextPreprocessor:
    """
    Preprocesses text data for model training.
    
    Features:
        - Text cleaning and normalization
        - Language detection and filtering
        - Quality filtering
        - Deduplication
        - Train/validation/test splitting
    """
    
    def __init__(self):
        """Initialize the text preprocessor."""
        self.config = config.preprocessing
        self.seen_hashes: Set[str] = set()
        self.stats = {
            'total_processed': 0,
            'passed_cleaning': 0,
            'passed_language': 0,
            'passed_quality': 0,
            'passed_dedup': 0,
            'final_count': 0
        }
        
        # Load stopwords
        if NLTK_AVAILABLE:
            self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = set()
        
        logger.info("TextPreprocessor initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Fix unicode
        if self.config.fix_unicode:
            text = unicodedata.normalize('NFKC', text)
            text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Remove URLs
        if self.config.remove_urls:
            text = re.sub(r'http[s]?://\S+', '', text)
            text = re.sub(r'www\.\S+', '', text)
        
        # Remove emails
        if self.config.remove_emails:
            text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Remove phone numbers
        if self.config.remove_phone_numbers:
            text = re.sub(r'[\+]?[\d\-\(\)\s]{10,}', '', text)
        
        # Remove special characters (optional)
        if self.config.remove_special_characters:
            text = re.sub(r'[^\w\s\.\,\!\?\;\:\'\"\-]', '', text)
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?]){3,}', r'\1\1', text)
        text = re.sub(r'[-_=]{3,}', '', text)
        
        # Lowercase (optional)
        if self.config.lowercase:
            text = text.lower()
        
        return text
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not LANGDETECT_AVAILABLE:
            return (self.config.language, 1.0)
        
        try:
            langs = detect_langs(text)
            if langs:
                top_lang = langs[0]
                return (top_lang.lang, top_lang.prob)
        except:
            pass
        
        return ("unknown", 0.0)
    
    def calculate_text_quality(self, text: str) -> Dict[str, float]:
        """
        Calculate various text quality metrics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return {'word_count': 0, 'quality_score': 0.0}
        
        # Basic metrics
        metrics['word_count'] = word_count
        metrics['char_count'] = len(text)
        metrics['avg_word_length'] = sum(len(w) for w in words) / word_count
        
        # Vocabulary metrics
        unique_words = set(w.lower() for w in words)
        metrics['unique_word_ratio'] = len(unique_words) / word_count
        
        # Sentence metrics
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
                metrics['sentence_count'] = len(sentences)
                if sentences:
                    metrics['avg_sentence_length'] = word_count / len(sentences)
            except:
                metrics['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
                metrics['avg_sentence_length'] = word_count / max(1, metrics['sentence_count'])
        else:
            metrics['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
            metrics['avg_sentence_length'] = word_count / max(1, metrics['sentence_count'])
        
        # Repetition check
        word_freq = Counter(words)
        most_common_freq = word_freq.most_common(1)[0][1] if word_freq else 0
        metrics['max_word_frequency'] = most_common_freq / word_count
        
        # Stopword ratio
        if self.stopwords:
            stopword_count = sum(1 for w in words if w.lower() in self.stopwords)
            metrics['stopword_ratio'] = stopword_count / word_count
        
        # Readability metrics
        if TEXTSTAT_AVAILABLE:
            try:
                metrics['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
                metrics['gunning_fog'] = textstat.gunning_fog(text)
            except:
                pass
        
        # Overall quality score (0-1)
        quality_score = 1.0
        
        # Penalize very short or very long texts
        if word_count < 30:
            quality_score *= 0.7
        if word_count > 400:
            quality_score *= 0.9
        
        # Penalize low vocabulary diversity
        if metrics['unique_word_ratio'] < 0.3:
            quality_score *= 0.6
        
        # Penalize high repetition
        if metrics['max_word_frequency'] > 0.1:
            quality_score *= 0.8
        
        # Penalize extreme sentence lengths
        if 'avg_sentence_length' in metrics:
            if metrics['avg_sentence_length'] < 5 or metrics['avg_sentence_length'] > 50:
                quality_score *= 0.8
        
        metrics['quality_score'] = quality_score
        
        return metrics
    
    def is_valid_sample(self, text: str, check_language: bool = True) -> bool:
        """
        Check if a text sample meets quality criteria.
        
        Args:
            text: Input text
            check_language: Whether to check language
            
        Returns:
            True if valid, False otherwise
        """
        self.stats['total_processed'] += 1
        
        if not text or not text.strip():
            return False
        
        text = text.strip()
        
        # Length checks
        words = text.split()
        word_count = len(words)
        
        if len(text) < config.data_collection.min_text_length:
            return False
        if len(text) > config.data_collection.max_text_length:
            return False
        if word_count < config.data_collection.min_word_count:
            return False
        if word_count > config.data_collection.max_word_count:
            return False
        
        self.stats['passed_cleaning'] += 1
        
        # Language check
        if check_language and LANGDETECT_AVAILABLE:
            lang, confidence = self.detect_language(text)
            if lang != self.config.language or confidence < self.config.min_language_confidence:
                return False
        
        self.stats['passed_language'] += 1
        
        # Quality checks
        metrics = self.calculate_text_quality(text)
        
        if metrics.get('unique_word_ratio', 0) < self.config.min_unique_words_ratio:
            return False
        
        if metrics.get('sentence_count', 0) < self.config.min_sentence_count:
            return False
        
        if metrics.get('sentence_count', 0) > self.config.max_sentence_count:
            return False
        
        self.stats['passed_quality'] += 1
        
        # Deduplication
        if self.config.use_deduplication:
            text_hash = hashlib.md5(text.lower().strip().encode()).hexdigest()
            if text_hash in self.seen_hashes:
                return False
            self.seen_hashes.add(text_hash)
        
        self.stats['passed_dedup'] += 1
        self.stats['final_count'] += 1
        
        return True
    
    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        check_language: bool = True
    ) -> int:
        """
        Process a JSONL file and output cleaned samples.
        
        Args:
            input_path: Input JSONL file
            output_path: Output JSONL file
            check_language: Whether to check language
            
        Returns:
            Number of valid samples
        """
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 0
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        valid_count = 0
        
        # Count total lines for progress bar
        with open(input_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in tqdm(infile, total=total_lines, desc="Processing"):
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', '')
                    
                    # Clean text
                    cleaned_text = self.clean_text(text)
                    
                    # Validate
                    if self.is_valid_sample(cleaned_text, check_language):
                        data['text'] = cleaned_text
                        outfile.write(json.dumps(data) + '\n')
                        valid_count += 1
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.debug(f"Error processing line: {e}")
                    continue
        
        logger.info(f"Processed {total_lines} samples, {valid_count} valid")
        return valid_count
    
    def split_dataset(
        self,
        input_path: Path,
        train_path: Path,
        val_path: Path,
        test_path: Path,
        shuffle: bool = True
    ) -> Dict[str, int]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            input_path: Input JSONL file
            train_path: Output training file
            val_path: Output validation file
            test_path: Output test file
            shuffle: Whether to shuffle data
            
        Returns:
            Dictionary with split counts
        """
        import random
        
        logger.info("Splitting dataset...")
        
        # Read all samples
        samples = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    samples.append(json.loads(line.strip()))
                except:
                    continue
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(samples)
        
        # Calculate split indices
        total = len(samples)
        train_end = int(total * self.config.train_ratio)
        val_end = train_end + int(total * self.config.val_ratio)
        
        # Split
        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]
        
        # Write splits
        for path, data in [(train_path, train_samples), 
                           (val_path, val_samples), 
                           (test_path, test_samples)]:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                for sample in data:
                    f.write(json.dumps(sample) + '\n')
        
        split_counts = {
            'train': len(train_samples),
            'val': len(val_samples),
            'test': len(test_samples),
            'total': total
        }
        
        logger.info(f"Split complete: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}")
        
        return split_counts
    
    def merge_and_balance(
        self,
        human_path: Path,
        ai_path: Path,
        output_path: Path,
        balance: bool = True
    ) -> int:
        """
        Merge human and AI datasets with optional balancing.
        
        Args:
            human_path: Path to human text file
            ai_path: Path to AI text file
            output_path: Output merged file
            balance: Whether to balance classes
            
        Returns:
            Total number of samples
        """
        logger.info("Merging and balancing datasets...")
        
        # Read both files
        human_samples = []
        ai_samples = []
        
        if human_path.exists():
            with open(human_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        data['label'] = 0  # Human
                        human_samples.append(data)
                    except:
                        continue
        
        if ai_path.exists():
            with open(ai_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        data['label'] = 1  # AI
                        ai_samples.append(data)
                    except:
                        continue
        
        logger.info(f"Human samples: {len(human_samples)}, AI samples: {len(ai_samples)}")
        
        # Balance if requested
        if balance:
            import random
            min_count = min(len(human_samples), len(ai_samples))
            random.shuffle(human_samples)
            random.shuffle(ai_samples)
            human_samples = human_samples[:min_count]
            ai_samples = ai_samples[:min_count]
            logger.info(f"Balanced to {min_count} samples each")
        
        # Merge and shuffle
        all_samples = human_samples + ai_samples
        import random
        random.shuffle(all_samples)
        
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample) + '\n')
        
        return len(all_samples)
    
    def print_stats(self):
        """Print processing statistics."""
        print("\n" + "="*50)
        print("PREPROCESSING STATISTICS")
        print("="*50)
        print(f"Total processed:     {self.stats['total_processed']}")
        print(f"Passed cleaning:     {self.stats['passed_cleaning']}")
        print(f"Passed language:     {self.stats['passed_language']}")
        print(f"Passed quality:      {self.stats['passed_quality']}")
        print(f"Passed dedup:        {self.stats['passed_dedup']}")
        print(f"Final count:         {self.stats['final_count']}")
        print("="*50 + "\n")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for text preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess text data for training")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--no-language-check", action="store_true", help="Skip language detection")
    parser.add_argument("--split", action="store_true", help="Also split into train/val/test")
    
    args = parser.parse_args()
    
    preprocessor = TextPreprocessor()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    count = preprocessor.process_file(input_path, output_path, not args.no_language_check)
    
    if args.split:
        train_path = output_path.parent / "train.jsonl"
        val_path = output_path.parent / "val.jsonl"
        test_path = output_path.parent / "test.jsonl"
        preprocessor.split_dataset(output_path, train_path, val_path, test_path)
    
    preprocessor.print_stats()


if __name__ == "__main__":
    main()