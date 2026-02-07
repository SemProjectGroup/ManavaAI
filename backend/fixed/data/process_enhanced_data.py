"""
Process collected data for training.
Adds linguistic features and prepares train/val/test splits.
"""

import os
import sys
import json
import random
import re
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


class EnhancedDataProcessor:
    """Process and prepare data for training."""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.raw_dir = self.config.RAW_DATA_DIR
        self.output_dir = self.config.DATA_DIR
        
        # AI indicators for feature extraction
        self.ai_phrases = [
            "it is important to note", "it is worth noting", "it is essential",
            "furthermore", "moreover", "additionally", "consequently",
            "in conclusion", "to summarize", "in today's world",
            "plays a crucial role", "cannot be overstated", "research suggests",
            "studies have shown", "it is crucial", "it is vital",
        ]
        
        self.casual_words = {
            'lol', 'haha', 'omg', 'btw', 'tbh', 'idk', 'gonna', 'wanna',
            'kinda', 'yeah', 'nope', 'cool', 'awesome', 'weird', 'stuff'
        }
    
    def process(self):
        """Main processing function."""
        print("="*60)
        print("PROCESSING DATA")
        print("="*60)
        
        # Load raw data
        all_samples = []
        
        # Load human samples
        human_file = os.path.join(self.raw_dir, "human_texts_enhanced.jsonl")
        if os.path.exists(human_file):
            human_samples = self._load_jsonl(human_file)
            print(f"Loaded {len(human_samples)} human samples")
            all_samples.extend(human_samples)
        
        # Load AI samples
        ai_file = os.path.join(self.raw_dir, "ai_texts_enhanced.jsonl")
        if os.path.exists(ai_file):
            ai_samples = self._load_jsonl(ai_file)
            print(f"Loaded {len(ai_samples)} AI samples")
            all_samples.extend(ai_samples)
        
        if not all_samples:
            print("❌ No data found! Run data collection first.")
            return
        
        # Process samples
        print(f"\nProcessing {len(all_samples)} samples...")
        processed = []
        
        for sample in tqdm(all_samples, desc="Processing"):
            processed_sample = self._process_sample(sample)
            if processed_sample:
                processed.append(processed_sample)
        
        print(f"Valid samples: {len(processed)}")
        
        # Balance dataset
        processed = self._balance_dataset(processed)
        
        # Split into train/val/test
        random.shuffle(processed)
        
        n = len(processed)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        
        train_data = processed[:train_end]
        val_data = processed[train_end:val_end]
        test_data = processed[val_end:]
        
        # Save splits
        self._save_jsonl(train_data, os.path.join(self.output_dir, "train.jsonl"))
        self._save_jsonl(val_data, os.path.join(self.output_dir, "val.jsonl"))
        self._save_jsonl(test_data, os.path.join(self.output_dir, "test.jsonl"))
        
        # Print stats
        print(f"\n✅ Data processed!")
        print(f"   Train: {len(train_data)} samples")
        print(f"   Val:   {len(val_data)} samples")
        print(f"   Test:  {len(test_data)} samples")
        
        # Label distribution
        train_ai = sum(1 for s in train_data if s['label'] == 1)
        print(f"\n   Train distribution: AI={train_ai}, Human={len(train_data)-train_ai}")
    
    def _load_jsonl(self, filepath):
        """Load JSONL file."""
        samples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples
    
    def _save_jsonl(self, samples, filepath):
        """Save to JSONL file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    def _process_sample(self, sample):
        """Process a single sample and extract features."""
        text = sample.get('text', '')
        label = sample.get('label', 0)
        
        # Clean text
        text = self._clean_text(text)
        
        # Validate
        words = text.split()
        if len(words) < 50 or len(words) > 1000:
            return None
        
        # Extract features
        features = self._extract_features(text)
        
        return {
            'text': text,
            'label': label,
            'source': sample.get('source', 'unknown'),
            'features': features
        }
    
    def _clean_text(self, text):
        """Clean and normalize text."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def _extract_features(self, text):
        """Extract linguistic features."""
        text_lower = text.lower()
        words = text_lower.split()
        
        features = {}
        
        # AI phrase count
        features['ai_phrase_count'] = sum(1 for p in self.ai_phrases if p in text_lower)
        
        # Em dash count (AI uses this a lot!)
        features['em_dash_count'] = text.count('—') + text.count('--')
        
        # Contraction count
        features['contraction_count'] = sum(1 for w in words if "'" in w)
        
        # Casual word count
        features['casual_count'] = sum(1 for w in words if w in self.casual_words)
        
        # Sentence stats
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if sentences:
            lens = [len(s.split()) for s in sentences]
            features['avg_sentence_length'] = sum(lens) / len(lens)
            features['sentence_count'] = len(sentences)
        else:
            features['avg_sentence_length'] = 0
            features['sentence_count'] = 0
        
        return features
    
    def _balance_dataset(self, samples):
        """Balance the dataset between AI and human samples."""
        human = [s for s in samples if s['label'] == 0]
        ai = [s for s in samples if s['label'] == 1]
        
        print(f"\nBefore balancing: Human={len(human)}, AI={len(ai)}")
        
        # Balance to smaller class
        min_size = min(len(human), len(ai))
        
        random.shuffle(human)
        random.shuffle(ai)
        
        balanced = human[:min_size] + ai[:min_size]
        random.shuffle(balanced)
        
        print(f"After balancing: {len(balanced)} total ({min_size} each)")
        
        return balanced


def main():
    processor = EnhancedDataProcessor()
    processor.process()


if __name__ == "__main__":
    main()