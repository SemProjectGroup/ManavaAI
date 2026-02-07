"""
Data Validation Module
======================
Validates data quality and generates reports on dataset statistics.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from dataclasses import dataclass

from tqdm import tqdm
from loguru import logger

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config


@dataclass
class DatasetStats:
    """Statistics about a dataset."""
    total_samples: int
    human_samples: int
    ai_samples: int
    avg_text_length: float
    avg_word_count: float
    min_text_length: int
    max_text_length: int
    unique_sources: List[str]
    source_distribution: Dict[str, int]
    label_distribution: Dict[int, int]


class DataValidator:
    """
    Validates data quality and generates statistics.
    
    Features:
        - Dataset statistics calculation
        - Quality checks
        - Duplicate detection
        - Distribution analysis
        - Report generation
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the data validator.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir or config.processed_data_dir
        
    def load_samples(self, path: Path, limit: Optional[int] = None) -> List[Dict]:
        """
        Load samples from a JSONL file.
        
        Args:
            path: Path to JSONL file
            limit: Maximum samples to load
            
        Returns:
            List of samples
        """
        samples = []
        
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return samples
        
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                try:
                    samples.append(json.loads(line.strip()))
                except:
                    continue
        
        return samples
    
    def calculate_stats(self, samples: List[Dict]) -> DatasetStats:
        """
        Calculate statistics for a list of samples.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            DatasetStats object
        """
        if not samples:
            return DatasetStats(
                total_samples=0,
                human_samples=0,
                ai_samples=0,
                avg_text_length=0.0,
                avg_word_count=0.0,
                min_text_length=0,
                max_text_length=0,
                unique_sources=[],
                source_distribution={},
                label_distribution={}
            )
        
        text_lengths = []
        word_counts = []
        sources = []
        labels = []
        
        for sample in samples:
            text = sample.get('text', '')
            text_lengths.append(len(text))
            word_counts.append(len(text.split()))
            sources.append(sample.get('source', 'unknown'))
            labels.append(sample.get('label', 0))
        
        source_counter = Counter(sources)
        label_counter = Counter(labels)
        
        return DatasetStats(
            total_samples=len(samples),
            human_samples=label_counter.get(0, 0),
            ai_samples=label_counter.get(1, 0),
            avg_text_length=sum(text_lengths) / len(text_lengths),
            avg_word_count=sum(word_counts) / len(word_counts),
            min_text_length=min(text_lengths),
            max_text_length=max(text_lengths),
            unique_sources=list(source_counter.keys()),
            source_distribution=dict(source_counter),
            label_distribution=dict(label_counter)
        )
    
    def validate_sample(self, sample: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a single sample.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        if 'text' not in sample:
            issues.append("Missing 'text' field")
        elif not sample['text'] or not sample['text'].strip():
            issues.append("Empty text")
        
        if 'label' not in sample:
            issues.append("Missing 'label' field")
        elif sample['label'] not in [0, 1]:
            issues.append(f"Invalid label: {sample['label']}")
        
        # Check text quality
        if 'text' in sample and sample['text']:
            text = sample['text']
            
            if len(text) < config.data_collection.min_text_length:
                issues.append(f"Text too short: {len(text)} chars")
            
            if len(text) > config.data_collection.max_text_length:
                issues.append(f"Text too long: {len(text)} chars")
            
            word_count = len(text.split())
            if word_count < config.data_collection.min_word_count:
                issues.append(f"Too few words: {word_count}")
        
        return len(issues) == 0, issues
    
    def validate_dataset(
        self,
        path: Path,
        sample_size: Optional[int] = None
    ) -> Dict:
        """
        Validate an entire dataset.
        
        Args:
            path: Path to JSONL file
            sample_size: Number of samples to validate (None for all)
            
        Returns:
            Validation report dictionary
        """
        samples = self.load_samples(path)
        
        if sample_size:
            samples = random.sample(samples, min(sample_size, len(samples)))
        
        valid_count = 0
        invalid_count = 0
        all_issues = []
        
        for sample in tqdm(samples, desc="Validating"):
            is_valid, issues = self.validate_sample(sample)
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                all_issues.extend(issues)
        
        issue_counts = Counter(all_issues)
        
        return {
            'file': str(path),
            'total_samples': len(samples),
            'valid_samples': valid_count,
            'invalid_samples': invalid_count,
            'validity_rate': valid_count / len(samples) if samples else 0,
            'common_issues': dict(issue_counts.most_common(10))
        }
    
    def check_duplicates(self, path: Path, method: str = "exact") -> Dict:
        """
        Check for duplicate samples.
        
        Args:
            path: Path to JSONL file
            method: Duplicate detection method ("exact" or "fuzzy")
            
        Returns:
            Duplicate report
        """
        samples = self.load_samples(path)
        
        if method == "exact":
            seen = set()
            duplicates = 0
            
            for sample in tqdm(samples, desc="Checking duplicates"):
                text = sample.get('text', '').strip().lower()
                if text in seen:
                    duplicates += 1
                else:
                    seen.add(text)
            
            return {
                'method': 'exact',
                'total_samples': len(samples),
                'unique_samples': len(samples) - duplicates,
                'duplicates': duplicates,
                'duplicate_rate': duplicates / len(samples) if samples else 0
            }
        
        else:
            # Fuzzy matching (hash-based approximation)
            import hashlib
            
            def get_shingles(text, k=5):
                text = text.lower().strip()
                words = text.split()
                return set(' '.join(words[i:i+k]) for i in range(len(words) - k + 1))
            
            shingle_sets = []
            for sample in tqdm(samples, desc="Computing shingles"):
                text = sample.get('text', '')
                shingle_sets.append(get_shingles(text))
            
            # Find near-duplicates
            duplicates = 0
            threshold = 0.8
            
            for i in range(len(shingle_sets)):
                for j in range(i + 1, min(i + 100, len(shingle_sets))):
                    if shingle_sets[i] and shingle_sets[j]:
                        intersection = len(shingle_sets[i] & shingle_sets[j])
                        union = len(shingle_sets[i] | shingle_sets[j])
                        if union > 0 and intersection / union >= threshold:
                            duplicates += 1
                            break
            
            return {
                'method': 'fuzzy',
                'threshold': threshold,
                'total_samples': len(samples),
                'near_duplicates': duplicates,
                'near_duplicate_rate': duplicates / len(samples) if samples else 0
            }
    
    def analyze_distribution(
        self,
        path: Path,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Analyze data distribution and generate visualizations.
        
        Args:
            path: Path to JSONL file
            output_dir: Directory to save plots
            
        Returns:
            Distribution analysis dictionary
        """
        samples = self.load_samples(path)
        stats = self.calculate_stats(samples)
        
        analysis = {
            'total_samples': stats.total_samples,
            'label_distribution': stats.label_distribution,
            'source_distribution': stats.source_distribution,
            'avg_text_length': stats.avg_text_length,
            'avg_word_count': stats.avg_word_count,
            'length_range': (stats.min_text_length, stats.max_text_length)
        }
        
        # Generate plots if available
        if PLOTTING_AVAILABLE and PANDAS_AVAILABLE and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Text length distribution
            text_lengths = [len(s.get('text', '')) for s in samples]
            
            plt.figure(figsize=(10, 6))
            plt.hist(text_lengths, bins=50, edgecolor='black', alpha=0.7)
            plt.xlabel('Text Length (characters)')
            plt.ylabel('Frequency')
            plt.title('Text Length Distribution')
            plt.savefig(output_dir / 'text_length_dist.png', dpi=150)
            plt.close()
            
            # Label distribution
            plt.figure(figsize=(8, 6))
            labels = ['Human', 'AI']
            counts = [stats.human_samples, stats.ai_samples]
            plt.bar(labels, counts, color=['blue', 'red'], alpha=0.7)
            plt.xlabel('Label')
            plt.ylabel('Count')
            plt.title('Label Distribution')
            plt.savefig(output_dir / 'label_dist.png', dpi=150)
            plt.close()
            
            # Source distribution
            if stats.source_distribution:
                plt.figure(figsize=(12, 6))
                sources = list(stats.source_distribution.keys())
                counts = list(stats.source_distribution.values())
                plt.barh(sources, counts, alpha=0.7)
                plt.xlabel('Count')
                plt.ylabel('Source')
                plt.title('Source Distribution')
                plt.tight_layout()
                plt.savefig(output_dir / 'source_dist.png', dpi=150)
                plt.close()
            
            analysis['plots_saved'] = str(output_dir)
        
        return analysis
    
    def generate_report(
        self,
        train_path: Optional[Path] = None,
        val_path: Optional[Path] = None,
        test_path: Optional[Path] = None
    ) -> Dict:
        """
        Generate a comprehensive validation report.
        
        Args:
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data
            
        Returns:
            Complete validation report
        """
        report = {'datasets': {}}
        
        paths = {
            'train': train_path or self.data_dir / "train.jsonl",
            'val': val_path or self.data_dir / "val.jsonl",
            'test': test_path or self.data_dir / "test.jsonl"
        }
        
        for name, path in paths.items():
            if path.exists():
                logger.info(f"Analyzing {name} dataset...")
                samples = self.load_samples(path)
                stats = self.calculate_stats(samples)
                validation = self.validate_dataset(path, sample_size=1000)
                duplicates = self.check_duplicates(path)
                
                report['datasets'][name] = {
                    'path': str(path),
                    'stats': {
                        'total_samples': stats.total_samples,
                        'human_samples': stats.human_samples,
                        'ai_samples': stats.ai_samples,
                        'avg_text_length': round(stats.avg_text_length, 2),
                        'avg_word_count': round(stats.avg_word_count, 2),
                        'sources': stats.unique_sources
                    },
                    'validation': validation,
                    'duplicates': duplicates
                }
        
        # Overall summary
        total_samples = sum(
            r['stats']['total_samples'] 
            for r in report['datasets'].values()
        )
        
        report['summary'] = {
            'total_samples_all_splits': total_samples,
            'datasets_found': list(report['datasets'].keys())
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Pretty print the validation report."""
        print("\n" + "="*60)
        print("DATASET VALIDATION REPORT")
        print("="*60)
        
        for name, data in report.get('datasets', {}).items():
            print(f"\n{name.upper()} DATASET")
            print("-" * 40)
            
            stats = data.get('stats', {})
            print(f"  Total samples:     {stats.get('total_samples', 0)}")
            print(f"  Human samples:     {stats.get('human_samples', 0)}")
            print(f"  AI samples:        {stats.get('ai_samples', 0)}")
            print(f"  Avg text length:   {stats.get('avg_text_length', 0)} chars")
            print(f"  Avg word count:    {stats.get('avg_word_count', 0)} words")
            
            validation = data.get('validation', {})
            print(f"  Validity rate:     {validation.get('validity_rate', 0)*100:.1f}%")
            
            duplicates = data.get('duplicates', {})
            print(f"  Duplicate rate:    {duplicates.get('duplicate_rate', 0)*100:.1f}%")
        
        print("\n" + "="*60)
        summary = report.get('summary', {})
        print(f"TOTAL SAMPLES: {summary.get('total_samples_all_splits', 0)}")
        print("="*60 + "\n")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for data validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate dataset quality")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--report", action="store_true", help="Generate full report")
    parser.add_argument("--check-duplicates", type=str, help="Check duplicates in specific file")
    parser.add_argument("--analyze", type=str, help="Analyze specific file")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else config.processed_data_dir
    validator = DataValidator(data_dir)
    
    if args.check_duplicates:
        result = validator.check_duplicates(Path(args.check_duplicates))
        print(json.dumps(result, indent=2))
    
    elif args.analyze:
        result = validator.analyze_distribution(
            Path(args.analyze), 
            output_dir=data_dir / "analysis"
        )
        print(json.dumps(result, indent=2))
    
    else:
        report = validator.generate_report()
        validator.print_report(report)
        
        # Save report
        report_path = data_dir / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()