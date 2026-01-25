"""
Tokenization Script for AI Text Humanizer Project
Tokenizes formatted dataset using base model tokenizer for training.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pickle

try:
    import torch
    from transformers import AutoTokenizer, GPT2Tokenizer
    from tqdm import tqdm
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install torch transformers tqdm numpy")
    exit(1)


class DataTokenizer:
    """Handles tokenization of text data for model training."""
    
    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.stats = {
            'total_samples': 0,
            'tokenized_samples': 0,
            'failed_samples': 0,
            'avg_token_length': 0,
            'max_token_length': 0,
            'min_token_length': float('inf'),
            'truncated_samples': 0,
            'padded_samples': 0
        }
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('tokenization_log.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_tokenizer(self):
        """Load tokenizer from specified model."""
        self.logger.info(f"Loading tokenizer: {self.args.model_name}")
        
        try:
            # Load tokenizer based on model name
            if 'gpt2' in self.args.model_name.lower():
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    self.args.model_name,
                    cache_dir=self.args.cache_dir
                )
                # GPT-2 doesn't have a pad token by default
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            elif 'mistral' in self.args.model_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.args.model_name,
                    cache_dir=self.args.cache_dir,
                    trust_remote_code=True
                )
                # Mistral models typically have pad token, but check
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            else:
                # Generic AutoTokenizer for other models
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.args.model_name,
                    cache_dir=self.args.cache_dir,
                    trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Set padding side
            self.tokenizer.padding_side = self.args.padding_side
            
            self.logger.info(f"Tokenizer loaded successfully")
            self.logger.info(f"Vocabulary size: {self.tokenizer.vocab_size}")
            self.logger.info(f"Pad token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
            self.logger.info(f"EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
            self.logger.info(f"BOS token: {self.tokenizer.bos_token} (ID: {self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else 'N/A'})")
            
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def load_dataset(self, file_path: Path) -> List[Dict]:
        """Load dataset from file."""
        samples = []
        
        self.logger.info(f"Loading dataset from {file_path}")
        
        try:
            if file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, desc="Loading data"):
                        if line.strip():
                            samples.append(json.loads(line))
            
            elif file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples = data
                    else:
                        samples = [data]
            
            self.logger.info(f"Loaded {len(samples)} samples")
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
        
        return samples
    
    def tokenize_sample(self, sample: Dict) -> Optional[Dict]:
        """Tokenize a single sample."""
        try:
            text = sample.get('text', '')
            if not text:
                self.stats['failed_samples'] += 1
                return None
            
            # Tokenize text
            encoded = self.tokenizer(
                text,
                max_length=self.args.max_length,
                truncation=self.args.truncation,
                padding=False,  # We'll pad in batches during training
                return_tensors=None,  # Return as lists
                add_special_tokens=self.args.add_special_tokens
            )
            
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            
            # Track statistics
            token_length = len(input_ids)
            self.stats['avg_token_length'] += token_length
            self.stats['max_token_length'] = max(self.stats['max_token_length'], token_length)
            self.stats['min_token_length'] = min(self.stats['min_token_length'], token_length)
            
            # Check if truncated
            if token_length >= self.args.max_length:
                self.stats['truncated_samples'] += 1
            
            # Create tokenized sample
            tokenized_sample = {
                'id': sample.get('id', ''),
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': sample.get('label', ''),
                'source': sample.get('source', ''),
                'token_length': token_length,
                'original_text_length': len(text),
                'metadata': sample.get('metadata', {})
            }
            
            # Add label encoding for classification (0: human, 1: ai)
            label_map = {'human': 0, 'ai': 1}
            tokenized_sample['label_id'] = label_map.get(sample.get('label', ''), -1)
            
            self.stats['tokenized_samples'] += 1
            return tokenized_sample
            
        except Exception as e:
            self.logger.warning(f"Failed to tokenize sample {sample.get('id', 'unknown')}: {e}")
            self.stats['failed_samples'] += 1
            return None
    
    def tokenize_dataset(self, samples: List[Dict]) -> List[Dict]:
        """Tokenize all samples in dataset."""
        self.logger.info(f"Tokenizing {len(samples)} samples...")
        
        tokenized_samples = []
        self.stats['total_samples'] = len(samples)
        
        for sample in tqdm(samples, desc="Tokenizing"):
            tokenized = self.tokenize_sample(sample)
            if tokenized:
                tokenized_samples.append(tokenized)
        
        # Calculate average token length
        if self.stats['tokenized_samples'] > 0:
            self.stats['avg_token_length'] /= self.stats['tokenized_samples']
        
        self.logger.info(f"Successfully tokenized {len(tokenized_samples)} samples")
        
        return tokenized_samples
    
    def save_tokenized_data(self, tokenized_samples: List[Dict], output_path: Path):
        """Save tokenized data in specified format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving tokenized data to {output_path}")
        
        if self.args.output_format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in tqdm(tokenized_samples, desc="Saving"):
                    f.write(json.dumps(sample) + '\n')
        
        elif self.args.output_format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(tokenized_samples, f, indent=2)
        
        elif self.args.output_format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(tokenized_samples, f)
        
        elif self.args.output_format == 'pt':
            # Save as PyTorch tensors (more efficient for training)
            torch_data = {
                'input_ids': [s['input_ids'] for s in tokenized_samples],
                'attention_mask': [s['attention_mask'] for s in tokenized_samples],
                'labels': [s['label_id'] for s in tokenized_samples],
                'metadata': [
                    {
                        'id': s['id'],
                        'label': s['label'],
                        'source': s['source'],
                        'token_length': s['token_length']
                    }
                    for s in tokenized_samples
                ]
            }
            torch.save(torch_data, output_path)
        
        self.logger.info(f"Saved tokenized data to {output_path}")
    
    def save_tokenizer(self, output_dir: Path):
        """Save tokenizer for later use."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving tokenizer to {output_dir}")
        self.tokenizer.save_pretrained(output_dir)
        self.logger.info("Tokenizer saved successfully")
    
    def create_vocab_file(self, output_dir: Path):
        """Create vocabulary file for reference."""
        vocab_file = output_dir / 'vocabulary.json'
        
        vocab = self.tokenizer.get_vocab()
        
        # Sort by token ID
        sorted_vocab = {k: v for k, v in sorted(vocab.items(), key=lambda x: x[1])}
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_vocab, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Vocabulary saved to {vocab_file}")
    
    def analyze_token_distribution(self, tokenized_samples: List[Dict]) -> Dict:
        """Analyze token length distribution."""
        token_lengths = [s['token_length'] for s in tokenized_samples]
        
        analysis = {
            'mean': float(np.mean(token_lengths)),
            'median': float(np.median(token_lengths)),
            'std': float(np.std(token_lengths)),
            'min': int(np.min(token_lengths)),
            'max': int(np.max(token_lengths)),
            'percentiles': {
                '25': float(np.percentile(token_lengths, 25)),
                '50': float(np.percentile(token_lengths, 50)),
                '75': float(np.percentile(token_lengths, 75)),
                '90': float(np.percentile(token_lengths, 90)),
                '95': float(np.percentile(token_lengths, 95)),
                '99': float(np.percentile(token_lengths, 99))
            },
            'distribution': {}
        }
        
        # Create distribution bins
        bins = [0, 128, 256, 512, 1024, 2048, float('inf')]
        labels = ['0-128', '128-256', '256-512', '512-1024', '1024-2048', '2048+']
        
        for i in range(len(labels)):
            count = sum(1 for length in token_lengths if bins[i] <= length < bins[i+1])
            percentage = (count / len(token_lengths) * 100) if token_lengths else 0
            analysis['distribution'][labels[i]] = {
                'count': count,
                'percentage': round(percentage, 2)
            }
        
        return analysis
    
    def generate_report(self, output_dir: Path, tokenized_samples: List[Dict]):
        """Generate tokenization summary report."""
        # Analyze token distribution
        token_analysis = self.analyze_token_distribution(tokenized_samples)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'model_name': self.args.model_name,
                'max_length': self.args.max_length,
                'truncation': self.args.truncation,
                'padding_side': self.args.padding_side,
                'add_special_tokens': self.args.add_special_tokens
            },
            'tokenizer_info': {
                'vocab_size': self.tokenizer.vocab_size,
                'pad_token': self.tokenizer.pad_token,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token': self.tokenizer.eos_token,
                'eos_token_id': self.tokenizer.eos_token_id
            },
            'statistics': {
                'total_samples': self.stats['total_samples'],
                'tokenized_samples': self.stats['tokenized_samples'],
                'failed_samples': self.stats['failed_samples'],
                'success_rate': f"{(self.stats['tokenized_samples'] / self.stats['total_samples'] * 100):.2f}%" if self.stats['total_samples'] > 0 else "0%",
                'truncated_samples': self.stats['truncated_samples'],
                'truncation_rate': f"{(self.stats['truncated_samples'] / self.stats['tokenized_samples'] * 100):.2f}%" if self.stats['tokenized_samples'] > 0 else "0%"
            },
            'token_length_stats': {
                'average': round(self.stats['avg_token_length'], 2),
                'minimum': self.stats['min_token_length'] if self.stats['min_token_length'] != float('inf') else 0,
                'maximum': self.stats['max_token_length']
            },
            'token_distribution': token_analysis
        }
        
        report_path = output_dir / 'tokenization_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self.logger.info(f"\n{'='*70}")
        self.logger.info("TOKENIZATION SUMMARY")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Model: {self.args.model_name}")
        self.logger.info(f"Vocabulary Size: {self.tokenizer.vocab_size}")
        self.logger.info(f"\nSample Statistics:")
        self.logger.info(f"  Total samples: {self.stats['total_samples']}")
        self.logger.info(f"  Successfully tokenized: {self.stats['tokenized_samples']}")
        self.logger.info(f"  Failed: {self.stats['failed_samples']}")
        self.logger.info(f"  Success rate: {report['statistics']['success_rate']}")
        self.logger.info(f"\nToken Length Statistics:")
        self.logger.info(f"  Average: {report['token_length_stats']['average']:.2f}")
        self.logger.info(f"  Minimum: {report['token_length_stats']['minimum']}")
        self.logger.info(f"  Maximum: {report['token_length_stats']['maximum']}")
        self.logger.info(f"  Median: {token_analysis['median']:.2f}")
        self.logger.info(f"\nTruncation:")
        self.logger.info(f"  Truncated samples: {self.stats['truncated_samples']}")
        self.logger.info(f"  Truncation rate: {report['statistics']['truncation_rate']}")
        self.logger.info(f"\nToken Distribution:")
        for range_name, stats in token_analysis['distribution'].items():
            self.logger.info(f"  {range_name}: {stats['count']} ({stats['percentage']}%)")
        self.logger.info(f"\nPercentiles:")
        for percentile, value in token_analysis['percentiles'].items():
            self.logger.info(f"  {percentile}th: {value:.0f} tokens")
        self.logger.info(f"\nReport saved to: {report_path}")
    
    def process(self):
        """Main tokenization pipeline."""
        self.logger.info("Starting tokenization pipeline...")
        
        # Load tokenizer
        self.load_tokenizer()
        
        # Create output directory
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each split
        splits = ['train', 'val', 'test'] if self.args.process_splits else [self.args.input_file]
        
        for split in splits:
            if self.args.process_splits:
                input_file = Path(self.args.input_dir) / f"{split}.jsonl"
                output_file = output_dir / f"{split}_tokenized.{self.args.output_format}"
            else:
                input_file = Path(self.args.input_file)
                output_file = output_dir / f"tokenized.{self.args.output_format}"
            
            # Check if input file exists
            if not input_file.exists():
                self.logger.warning(f"Input file not found: {input_file}, skipping...")
                continue
            
            self.logger.info(f"\nProcessing: {split if self.args.process_splits else 'dataset'}")
            
            # Load dataset
            samples = self.load_dataset(input_file)
            
            # Tokenize
            tokenized_samples = self.tokenize_dataset(samples)
            
            # Save tokenized data
            self.save_tokenized_data(tokenized_samples, output_file)
            
            # Generate report (only for train split or single file)
            if not self.args.process_splits or split == 'train':
                self.generate_report(output_dir, tokenized_samples)
        
        # Save tokenizer
        self.save_tokenizer(output_dir / 'tokenizer')
        
        # Create vocabulary file
        if self.args.save_vocab:
            self.create_vocab_file(output_dir / 'tokenizer')
        
        self.logger.info("\n✓ Tokenization completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Tokenize dataset for model training')
    
    # Input/Output
    parser.add_argument('--input_file', type=str,
                        help='Path to input dataset file (for single file)')
    parser.add_argument('--input_dir', type=str,
                        help='Path to directory with train/val/test splits')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for tokenized data')
    parser.add_argument('--process_splits', action='store_true',
                        help='Process train/val/test splits from input_dir')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
                                'mistralai/Mistral-7B-v0.1', 'mistralai/Mistral-7B-Instruct-v0.2'],
                        help='Model name for tokenizer')
    parser.add_argument('--cache_dir', type=str, default='./model_cache',
                        help='Cache directory for model downloads')
    
    # Tokenization parameters
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--truncation', action='store_true', default=True,
                        help='Enable truncation')
    parser.add_argument('--padding_side', type=str, default='right',
                        choices=['left', 'right'],
                        help='Padding side')
    parser.add_argument('--add_special_tokens', action='store_true', default=True,
                        help='Add special tokens (BOS, EOS)')
    
    # Output format
    parser.add_argument('--output_format', type=str, default='jsonl',
                        choices=['jsonl', 'json', 'pickle', 'pt'],
                        help='Output format')
    parser.add_argument('--save_vocab', action='store_true',
                        help='Save vocabulary file')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.process_splits and not args.input_file:
        parser.error("Either --input_file or --process_splits with --input_dir must be specified")
    
    if args.process_splits and not args.input_dir:
        parser.error("--input_dir required when --process_splits is enabled")
    
    # Create tokenizer and process
    tokenizer = DataTokenizer(args)
    tokenizer.process()


if __name__ == '__main__':
    main()