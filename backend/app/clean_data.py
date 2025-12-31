"""
Data Cleaning Script for AI Text Humanizer Project
Cleans and preprocesses human and AI-generated text data for model training.
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import hashlib
from collections import defaultdict

try:
    from bs4 import BeautifulSoup
    import ftfy
    from langdetect import detect, LangDetectException
    from tqdm import tqdm
    import emoji
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install required packages: pip install beautifulsoup4 ftfy langdetect tqdm emoji")
    exit(1)


class TextCleaner:
    """Handles all text cleaning operations."""
    
    def __init__(self, min_length: int = 50, max_length: int = 10000, language: str = 'en'):
        self.min_length = min_length
        self.max_length = max_length
        self.language = language
        self.stats = defaultdict(int)
        
    def remove_html(self, text: str) -> str:
        """Remove HTML tags and extract text content."""
        soup = BeautifulSoup(text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()
            
        text = soup.get_text()
        return text
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs and email addresses."""
        # Remove URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
        
        # Remove www links
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', '', text)
        
        # Remove shortened URLs
        text = re.sub(r'\b(?:bit\.ly|t\.co|goo\.gl|tinyurl\.com)/\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        return text
    
    def remove_special_chars(self, text: str) -> str:
        """Remove emojis, excessive punctuation, and special characters."""
        # Remove emojis
        text = emoji.replace_emoji(text, replace='')
        
        # Remove emoticons
        emoticon_pattern = r'[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]'
        text = re.sub(emoticon_pattern, '', text)
        
        # Replace excessive punctuation
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        # Remove special unicode characters but keep basic punctuation
        text = re.sub(r'[^\x00-\x7F\u00C0-\u017F]+', ' ', text)
        
        # Remove arrows and symbols
        text = re.sub(r'[→←↑↓➜➔➡️⬅️⬆️⬇️]', '', text)
        text = re.sub(r'[■□●○◆◇★☆✓✗✘]', '', text)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace."""
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def normalize_text(self, text: str) -> str:
        """Normalize text encoding and special characters."""
        # Fix broken unicode
        text = ftfy.fix_text(text)
        
        # Convert smart quotes to regular quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Convert en/em dashes to hyphens
        text = text.replace('–', '-').replace('—', '-')
        
        # Convert ellipsis
        text = text.replace('…', '...')
        
        return text
    
    def remove_noise(self, text: str) -> str:
        """Remove common noise patterns."""
        # Remove [deleted], [removed], etc.
        text = re.sub(r'\[deleted\]|\[removed\]|\[redacted\]', '', text, flags=re.IGNORECASE)
        
        # Remove Reddit-specific patterns
        text = re.sub(r'Posted by u/\S+', '', text)
        text = re.sub(r'r/\S+', '', text)
        
        # Remove social media boilerplate
        patterns = [
            r'Click here to subscribe',
            r'Follow us on',
            r'Share this:',
            r'Like this:',
            r'Posted on \d+',
            r'\d+ likes?',
            r'\d+ shares?',
            r'\d+ comments?',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove repeated characters (coooool -> cool)
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Apply all cleaning operations in sequence."""
        if not text:
            return ""
        
        # Apply cleaning steps in order
        text = self.remove_html(text)
        text = self.remove_urls(text)
        text = self.remove_special_chars(text)
        text = self.normalize_text(text)
        text = self.remove_noise(text)
        text = self.normalize_whitespace(text)
        
        return text
    
    def is_valid_text(self, text: str) -> Tuple[bool, str]:
        """Check if text meets quality criteria. Returns (is_valid, reason)."""
        if not text:
            return False, "empty"
        
        # Length check
        if len(text) < self.min_length:
            return False, "too_short"
        if len(text) > self.max_length:
            return False, "too_long"
        
        # Check for excessive non-alphabetic characters
        alpha_chars = sum(c.isalpha() for c in text)
        total_chars = len(text.replace(' ', ''))
        if total_chars > 0 and alpha_chars / total_chars < 0.7:
            return False, "low_alpha_ratio"
        
        # Check for code-like content
        code_indicators = text.count('{') + text.count('}') + text.count(';') + text.count('[')
        if code_indicators > len(text) / 50:  # Too many code-like characters
            return False, "code_like"
        
        # Check for repetitive content
        words = text.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Less than 30% unique words
                return False, "repetitive"
        
        # Language detection
        try:
            detected_lang = detect(text)
            if detected_lang != self.language:
                return False, f"wrong_language_{detected_lang}"
        except LangDetectException:
            return False, "language_detection_failed"
        
        return True, "valid"


class DataCleaner:
    """Main data cleaning pipeline."""
    
    def __init__(self, args):
        self.args = args
        self.cleaner = TextCleaner(args.min_length, args.max_length, args.language)
        self.seen_hashes = set()
        self.stats = {
            'total_processed': 0,
            'total_valid': 0,
            'removed_by_reason': defaultdict(int),
            'duplicates_removed': 0
        }
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('cleaning_log.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_text_hash(self, text: str) -> str:
        """Generate hash for duplicate detection."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate."""
        text_hash = self.get_text_hash(text)
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False
    
    def read_file(self, file_path: Path) -> List[Dict]:
        """Read data from various file formats."""
        samples = []
        
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples = data
                    else:
                        samples = [data]
            
            elif file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
            
            elif file_path.suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Split by double newlines to separate documents
                    texts = content.split('\n\n')
                    for text in texts:
                        if text.strip():
                            samples.append({'text': text.strip()})
            
            elif file_path.suffix == '.csv':
                import csv
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    samples = list(reader)
        
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
        
        return samples
    
    def process_sample(self, sample: Dict, source: str, label: str) -> Optional[Dict]:
        """Process a single text sample."""
        # Extract text from various possible keys
        text = sample.get('text', sample.get('content', sample.get('body', '')))
        
        if not text:
            self.stats['removed_by_reason']['no_text'] += 1
            return None
        
        original_length = len(text)
        
        # Clean the text
        cleaned_text = self.cleaner.clean_text(text)
        
        # Validate text quality
        is_valid, reason = self.cleaner.is_valid_text(cleaned_text)
        if not is_valid:
            self.stats['removed_by_reason'][reason] += 1
            return None
        
        # Check for duplicates
        if self.args.remove_duplicates and self.is_duplicate(cleaned_text):
            self.stats['duplicates_removed'] += 1
            return None
        
        # Create output sample
        cleaned_sample = {
            'id': f"{label}_{self.stats['total_valid']:08d}",
            'text': cleaned_text,
            'label': label,
            'source': source,
            'original_length': original_length,
            'cleaned_length': len(cleaned_text),
            'timestamp': datetime.now().isoformat()
        }
        
        return cleaned_sample
    
    def process_directory(self, input_dir: Path, output_dir: Path, label: str):
        """Process all files in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all supported files
        files = list(input_dir.glob('*.json')) + \
                list(input_dir.glob('*.jsonl')) + \
                list(input_dir.glob('*.txt')) + \
                list(input_dir.glob('*.csv'))
        
        if not files:
            self.logger.warning(f"No supported files found in {input_dir}")
            return
        
        self.logger.info(f"Processing {len(files)} files from {input_dir}")
        
        # Output file
        output_file = output_dir / f"{label}_cleaned.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for file_path in tqdm(files, desc=f"Processing {label} files"):
                samples = self.read_file(file_path)
                source = file_path.stem
                
                for sample in samples:
                    self.stats['total_processed'] += 1
                    
                    cleaned_sample = self.process_sample(sample, source, label)
                    
                    if cleaned_sample:
                        out_f.write(json.dumps(cleaned_sample) + '\n')
                        self.stats['total_valid'] += 1
                        
                        if self.args.verbose and self.stats['total_valid'] % 1000 == 0:
                            self.logger.info(f"Processed {self.stats['total_valid']} valid samples")
        
        self.logger.info(f"Saved {self.stats['total_valid']} cleaned samples to {output_file}")
    
    def generate_report(self, output_dir: Path):
        """Generate cleaning summary report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'min_length': self.args.min_length,
                'max_length': self.args.max_length,
                'language': self.args.language,
                'remove_duplicates': self.args.remove_duplicates
            },
            'statistics': {
                'total_processed': self.stats['total_processed'],
                'total_valid': self.stats['total_valid'],
                'total_removed': self.stats['total_processed'] - self.stats['total_valid'],
                'removal_rate': f"{((self.stats['total_processed'] - self.stats['total_valid']) / self.stats['total_processed'] * 100):.2f}%",
                'duplicates_removed': self.stats['duplicates_removed'],
                'removed_by_reason': dict(self.stats['removed_by_reason'])
            }
        }
        
        report_path = output_dir / 'cleaning_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info("CLEANING SUMMARY")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Total processed: {report['statistics']['total_processed']}")
        self.logger.info(f"Total valid: {report['statistics']['total_valid']}")
        self.logger.info(f"Total removed: {report['statistics']['total_removed']}")
        self.logger.info(f"Removal rate: {report['statistics']['removal_rate']}")
        self.logger.info(f"Duplicates removed: {report['statistics']['duplicates_removed']}")
        self.logger.info(f"\nRemoval reasons:")
        for reason, count in report['statistics']['removed_by_reason'].items():
            self.logger.info(f"  {reason}: {count}")
        self.logger.info(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Clean and preprocess text data')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with raw data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for cleaned data')
    parser.add_argument('--label', type=str, required=True, choices=['human', 'ai'], help='Label for the data')
    parser.add_argument('--min_length', type=int, default=50, help='Minimum text length')
    parser.add_argument('--max_length', type=int, default=10000, help='Maximum text length')
    parser.add_argument('--language', type=str, default='en', help='Target language code')
    parser.add_argument('--remove_duplicates', action='store_true', help='Remove duplicate texts')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create data cleaner and process
    cleaner = DataCleaner(args)
    cleaner.process_directory(args.input_dir, args.output_dir, args.label)
    cleaner.generate_report(Path(args.output_dir))
    
    print("\n✓ Data cleaning completed successfully!")


if __name__ == '__main__':
    main()