"""
Dataset Formatting Script for AI Text Humanizer Project
Converts cleaned data into structured training format and balances the dataset.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import Counter
import random

try:
    from tqdm import tqdm
except ImportError:
    print("Missing dependency: pip install tqdm")
    exit(1)


class DatasetFormatter:
    """Formats and structures cleaned data for model training."""
    
    def __init__(self, args):
        self.args = args
        self.stats = {
            'human_samples': 0,
            'ai_samples': 0,
            'total_samples': 0,
            'balanced_human': 0,
            'balanced_ai': 0,
            'avg_human_length': 0,
            'avg_ai_length': 0
        }
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('formatting_log.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_cleaned_data(self, file_path: Path, label: str) -> List[Dict]:
        """Load cleaned data from JSONL file."""
        samples = []
        
        self.logger.info(f"Loading {label} data from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, desc=f"Loading {label} data"), 1):
                    if line.strip():
                        try:
                            sample = json.loads(line)
                            samples.append(sample)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                            continue
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return []
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {e}")
            return []
        
        self.logger.info(f"Loaded {len(samples)} {label} samples")
        return samples
    
    def validate_sample(self, sample: Dict) -> bool:
        """Validate that sample has required fields."""
        required_fields = ['text', 'label']
        
        for field in required_fields:
            if field not in sample or not sample[field]:
                return False
        
        return True
    
    def format_sample(self, sample: Dict, index: int) -> Dict:
        """Format sample into standardized structure."""
        formatted = {
            'id': sample.get('id', f"sample_{index:08d}"),
            'text': sample['text'],
            'label': sample['label'],
            'source': sample.get('source', 'unknown'),
            'length': len(sample['text']),
            'word_count': len(sample['text'].split()),
            'metadata': {
                'original_length': sample.get('original_length', len(sample['text'])),
                'cleaned_length': sample.get('cleaned_length', len(sample['text'])),
                'timestamp': sample.get('timestamp', datetime.now().isoformat())
            }
        }
        
        return formatted
    
    def balance_dataset(self, human_samples: List[Dict], ai_samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Balance dataset to have equal human and AI samples."""
        num_human = len(human_samples)
        num_ai = len(ai_samples)
        
        self.logger.info(f"Original counts - Human: {num_human}, AI: {num_ai}")
        
        if not self.args.balance:
            self.logger.info("Balancing disabled, keeping all samples")
            return human_samples, ai_samples
        
        # Determine target size
        if self.args.balance_method == 'undersample':
            # Use the smaller dataset size
            target_size = min(num_human, num_ai)
            self.logger.info(f"Undersampling to {target_size} samples per class")
            
            # Randomly sample from the larger dataset
            if num_human > target_size:
                human_samples = random.sample(human_samples, target_size)
            if num_ai > target_size:
                ai_samples = random.sample(ai_samples, target_size)
        
        elif self.args.balance_method == 'oversample':
            # Use the larger dataset size
            target_size = max(num_human, num_ai)
            self.logger.info(f"Oversampling to {target_size} samples per class")
            
            # Oversample the smaller dataset (with replacement)
            if num_human < target_size:
                additional_needed = target_size - num_human
                human_samples = human_samples + random.choices(human_samples, k=additional_needed)
            if num_ai < target_size:
                additional_needed = target_size - num_ai
                ai_samples = ai_samples + random.choices(ai_samples, k=additional_needed)
        
        elif self.args.balance_method == 'custom':
            # Use custom target size
            target_size = self.args.target_size
            self.logger.info(f"Custom balancing to {target_size} samples per class")
            
            if num_human > target_size:
                human_samples = random.sample(human_samples, target_size)
            elif num_human < target_size:
                additional_needed = target_size - num_human
                human_samples = human_samples + random.choices(human_samples, k=additional_needed)
            
            if num_ai > target_size:
                ai_samples = random.sample(ai_samples, target_size)
            elif num_ai < target_size:
                additional_needed = target_size - num_ai
                ai_samples = ai_samples + random.choices(ai_samples, k=additional_needed)
        
        self.logger.info(f"Balanced counts - Human: {len(human_samples)}, AI: {len(ai_samples)}")
        
        return human_samples, ai_samples
    
    def shuffle_and_combine(self, human_samples: List[Dict], ai_samples: List[Dict]) -> List[Dict]:
        """Combine and shuffle all samples."""
        all_samples = human_samples + ai_samples
        
        if self.args.shuffle:
            random.seed(self.args.seed)
            random.shuffle(all_samples)
            self.logger.info("Dataset shuffled")
        
        return all_samples
    
    def calculate_statistics(self, samples: List[Dict]):
        """Calculate dataset statistics."""
        human_samples = [s for s in samples if s['label'] == 'human']
        ai_samples = [s for s in samples if s['label'] == 'ai']
        
        self.stats['human_samples'] = len(human_samples)
        self.stats['ai_samples'] = len(ai_samples)
        self.stats['total_samples'] = len(samples)
        
        if human_samples:
            self.stats['avg_human_length'] = sum(s['length'] for s in human_samples) / len(human_samples)
            self.stats['avg_human_words'] = sum(s['word_count'] for s in human_samples) / len(human_samples)
        
        if ai_samples:
            self.stats['avg_ai_length'] = sum(s['length'] for s in ai_samples) / len(ai_samples)
            self.stats['avg_ai_words'] = sum(s['word_count'] for s in ai_samples) / len(ai_samples)
        
        # Calculate source distribution
        source_dist = Counter(s['source'] for s in samples)
        self.stats['source_distribution'] = dict(source_dist)
        
        # Calculate length distribution
        length_ranges = {
            '0-100': 0,
            '101-500': 0,
            '501-1000': 0,
            '1001-5000': 0,
            '5001+': 0
        }
        
        for sample in samples:
            length = sample['length']
            if length <= 100:
                length_ranges['0-100'] += 1
            elif length <= 500:
                length_ranges['101-500'] += 1
            elif length <= 1000:
                length_ranges['501-1000'] += 1
            elif length <= 5000:
                length_ranges['1001-5000'] += 1
            else:
                length_ranges['5001+'] += 1
        
        self.stats['length_distribution'] = length_ranges
    
    def save_dataset(self, samples: List[Dict], output_path: Path, format_type: str):
        """Save formatted dataset in specified format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in tqdm(samples, desc="Saving dataset"):
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        elif format_type == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
        
        elif format_type == 'csv':
            import csv
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                if samples:
                    writer = csv.DictWriter(f, fieldnames=['id', 'text', 'label', 'source', 'length', 'word_count'])
                    writer.writeheader()
                    for sample in tqdm(samples, desc="Saving dataset"):
                        row = {k: v for k, v in sample.items() if k in writer.fieldnames}
                        writer.writerow(row)
        
        self.logger.info(f"Saved {len(samples)} samples to {output_path}")
    
    def create_splits(self, samples: List[Dict], output_dir: Path):
        """Create train/validation/test splits."""
        random.seed(self.args.seed)
        random.shuffle(samples)
        
        total = len(samples)
        train_size = int(total * self.args.train_ratio)
        val_size = int(total * self.args.val_ratio)
        
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        self.logger.info(f"Created splits - Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
        
        # Save splits
        output_dir = Path(output_dir)
        self.save_dataset(train_samples, output_dir / f'train.{self.args.format}', self.args.format)
        self.save_dataset(val_samples, output_dir / f'val.{self.args.format}', self.args.format)
        self.save_dataset(test_samples, output_dir / f'test.{self.args.format}', self.args.format)
        
        return train_samples, val_samples, test_samples
    
    def generate_report(self, output_dir: Path):
        """Generate formatting summary report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'balance': self.args.balance,
                'balance_method': self.args.balance_method if self.args.balance else None,
                'shuffle': self.args.shuffle,
                'seed': self.args.seed,
                'format': self.args.format,
                'create_splits': self.args.split,
                'train_ratio': self.args.train_ratio if self.args.split else None,
                'val_ratio': self.args.val_ratio if self.args.split else None,
                'test_ratio': 1 - self.args.train_ratio - self.args.val_ratio if self.args.split else None
            },
            'statistics': self.stats
        }
        
        report_path = output_dir / 'formatting_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("DATASET FORMATTING SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total samples: {self.stats['total_samples']}")
        self.logger.info(f"Human samples: {self.stats['human_samples']}")
        self.logger.info(f"AI samples: {self.stats['ai_samples']}")
        self.logger.info(f"\nAverage lengths:")
        self.logger.info(f"  Human: {self.stats.get('avg_human_length', 0):.1f} chars, {self.stats.get('avg_human_words', 0):.1f} words")
        self.logger.info(f"  AI: {self.stats.get('avg_ai_length', 0):.1f} chars, {self.stats.get('avg_ai_words', 0):.1f} words")
        self.logger.info(f"\nLength distribution:")
        for range_name, count in self.stats.get('length_distribution', {}).items():
            percentage = (count / self.stats['total_samples'] * 100) if self.stats['total_samples'] > 0 else 0
            self.logger.info(f"  {range_name}: {count} ({percentage:.1f}%)")
        self.logger.info(f"\nTop 5 sources:")
        source_dist = self.stats.get('source_distribution', {})
        for source, count in sorted(source_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
            self.logger.info(f"  {source}: {count}")
        self.logger.info(f"\nReport saved to: {report_path}")
    
    def format(self):
        """Main formatting pipeline."""
        self.logger.info("Starting dataset formatting...")
        
        # Load data
        human_samples = self.load_cleaned_data(Path(self.args.human_file), 'human')
        ai_samples = self.load_cleaned_data(Path(self.args.ai_file), 'ai')
        
        if not human_samples and not ai_samples:
            self.logger.error("No data loaded. Exiting.")
            return
        
        # Validate and format samples
        self.logger.info("Formatting samples...")
        human_formatted = []
        for idx, sample in enumerate(tqdm(human_samples, desc="Formatting human samples")):
            if self.validate_sample(sample):
                human_formatted.append(self.format_sample(sample, idx))
        
        ai_formatted = []
        for idx, sample in enumerate(tqdm(ai_samples, desc="Formatting AI samples")):
            if self.validate_sample(sample):
                ai_formatted.append(self.format_sample(sample, idx))
        
        # Balance dataset
        human_formatted, ai_formatted = self.balance_dataset(human_formatted, ai_formatted)
        
        # Combine and shuffle
        all_samples = self.shuffle_and_combine(human_formatted, ai_formatted)
        
        # Calculate statistics
        self.calculate_statistics(all_samples)
        
        # Create output directory
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save dataset or create splits
        if self.args.split:
            self.create_splits(all_samples, output_dir)
        else:
            output_file = output_dir / f'formatted_dataset.{self.args.format}'
            self.save_dataset(all_samples, output_file, self.args.format)
        
        # Generate report
        self.generate_report(output_dir)
        
        self.logger.info("\n✓ Dataset formatting completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Format cleaned data into training dataset')
    
    # Input/Output
    parser.add_argument('--human_file', type=str, required=True, 
                        help='Path to cleaned human data (JSONL)')
    parser.add_argument('--ai_file', type=str, required=True,
                        help='Path to cleaned AI data (JSONL)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for formatted dataset')
    
    # Balancing
    parser.add_argument('--balance', action='store_true',
                        help='Balance dataset (equal human and AI samples)')
    parser.add_argument('--balance_method', type=str, default='undersample',
                        choices=['undersample', 'oversample', 'custom'],
                        help='Balancing method')
    parser.add_argument('--target_size', type=int, default=10000,
                        help='Target size per class for custom balancing')
    
    # Shuffling
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='Shuffle dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Output format
    parser.add_argument('--format', type=str, default='jsonl',
                        choices=['jsonl', 'json', 'csv'],
                        help='Output format')
    
    # Splitting
    parser.add_argument('--split', action='store_true',
                        help='Create train/val/test splits')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Validate split ratios
    if args.split:
        if args.train_ratio + args.val_ratio >= 1.0:
            print("Error: train_ratio + val_ratio must be less than 1.0")
            return
    
    # Create formatter and run
    formatter = DatasetFormatter(args)
    formatter.format()


if __name__ == '__main__':
    main()