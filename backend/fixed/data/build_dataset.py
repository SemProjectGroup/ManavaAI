"""
Dataset Builder Module
======================
Main script to build the complete training dataset.
Orchestrates collection, generation, preprocessing, and validation.
"""

import json
import time
import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime

from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config

from .collect_human_text import HumanTextCollector
from .generate_ai_text import AITextGenerator
from .preprocess import TextPreprocessor
from .validate import DataValidator


class DatasetBuilder:
    """
    Orchestrates the complete dataset building pipeline.
    
    Steps:
        1. Collect human-written text
        2. Generate AI text
        3. Preprocess all text
        4. Merge and balance
        5. Split into train/val/test
        6. Validate final dataset
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        target_samples: int = 100000
    ):
        """
        Initialize the dataset builder.
        
        Args:
            output_dir: Directory for output files
            target_samples: Target total samples (human + AI)
        """
        self.output_dir = output_dir or config.project_root / "datasets"
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        
        self.target_samples = target_samples
        self.samples_per_class = target_samples // 2
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.human_collector = HumanTextCollector(
            output_path=self.raw_dir / "human_texts.jsonl"
        )
        self.ai_generator = AITextGenerator(
            output_path=self.raw_dir / "ai_texts.jsonl"
        )
        self.preprocessor = TextPreprocessor()
        self.validator = DataValidator(self.processed_dir)
        
        logger.info(f"DatasetBuilder initialized. Target: {target_samples} samples")
        logger.info(f"Output directory: {self.output_dir}")
    
    def collect_human_text(self, num_samples: Optional[int] = None) -> int:
        """
        Step 1: Collect human-written text.
        
        Args:
            num_samples: Number of samples to collect
            
        Returns:
            Number of samples collected
        """
        target = num_samples or self.samples_per_class
        logger.info(f"Step 1: Collecting human text (target: {target})")
        
        start_time = time.time()
        count = self.human_collector.collect_all(target_samples=target)
        elapsed = time.time() - start_time
        
        logger.info(f"Human text collection complete: {count} samples in {elapsed:.1f}s")
        return count
    
    def generate_ai_text(
        self,
        num_samples: Optional[int] = None,
        quick_mode: bool = False
    ) -> int:
        """
        Step 2: Generate AI text.
        
        Args:
            num_samples: Number of samples to generate
            quick_mode: Use only small models for faster generation
            
        Returns:
            Number of samples generated
        """
        target = num_samples or self.samples_per_class
        logger.info(f"Step 2: Generating AI text (target: {target})")
        
        start_time = time.time()
        
        if quick_mode:
            count = self.ai_generator.generate_quick_dataset(num_samples=target)
        else:
            count = self.ai_generator.generate_all(target_samples=target)
        
        elapsed = time.time() - start_time
        
        logger.info(f"AI text generation complete: {count} samples in {elapsed:.1f}s")
        return count
    
    def preprocess_data(self) -> int:
        """
        Step 3: Preprocess all collected text.
        
        Returns:
            Number of valid samples after preprocessing
        """
        logger.info("Step 3: Preprocessing data")
        
        # Preprocess human text
        human_raw = self.raw_dir / "human_texts.jsonl"
        human_clean = self.processed_dir / "human_clean.jsonl"
        
        if human_raw.exists():
            human_count = self.preprocessor.process_file(
                human_raw, human_clean, check_language=True
            )
            logger.info(f"Processed human text: {human_count} valid samples")
        else:
            human_count = 0
            logger.warning("Human text file not found")
        
        # Preprocess AI text
        ai_raw = self.raw_dir / "ai_texts.jsonl"
        ai_clean = self.processed_dir / "ai_clean.jsonl"
        
        if ai_raw.exists():
            ai_count = self.preprocessor.process_file(
                ai_raw, ai_clean, check_language=True
            )
            logger.info(f"Processed AI text: {ai_count} valid samples")
        else:
            ai_count = 0
            logger.warning("AI text file not found")
        
        return human_count + ai_count
    
    def merge_and_split(self) -> dict:
        """
        Step 4: Merge datasets and split into train/val/test.
        
        Returns:
            Split counts dictionary
        """
        logger.info("Step 4: Merging and splitting data")
        
        # Merge
        human_clean = self.processed_dir / "human_clean.jsonl"
        ai_clean = self.processed_dir / "ai_clean.jsonl"
        merged_path = self.processed_dir / "merged.jsonl"
        
        total = self.preprocessor.merge_and_balance(
            human_clean, ai_clean, merged_path, balance=True
        )
        logger.info(f"Merged dataset: {total} samples")
        
        # Split
        train_path = self.processed_dir / "train.jsonl"
        val_path = self.processed_dir / "val.jsonl"
        test_path = self.processed_dir / "test.jsonl"
        
        split_counts = self.preprocessor.split_dataset(
            merged_path, train_path, val_path, test_path, shuffle=True
        )
        
        return split_counts
    
    def validate_dataset(self) -> dict:
        """
        Step 5: Validate the final dataset.
        
        Returns:
            Validation report
        """
        logger.info("Step 5: Validating dataset")
        
        report = self.validator.generate_report()
        self.validator.print_report(report)
        
        # Save report
        report_path = self.processed_dir / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def build(
        self,
        skip_collection: bool = False,
        skip_generation: bool = False,
        quick_mode: bool = False
    ) -> dict:
        """
        Run the complete dataset building pipeline.
        
        Args:
            skip_collection: Skip human text collection
            skip_generation: Skip AI text generation
            quick_mode: Use quick mode for AI generation
            
        Returns:
            Build summary dictionary
        """
        start_time = time.time()
        summary = {
            'start_time': datetime.now().isoformat(),
            'target_samples': self.target_samples,
            'output_dir': str(self.output_dir)
        }
        
        logger.info("="*60)
        logger.info("STARTING DATASET BUILD PIPELINE")
        logger.info("="*60)
        
        try:
            # Step 1: Collect human text
            if not skip_collection:
                summary['human_collected'] = self.collect_human_text()
            else:
                logger.info("Skipping human text collection")
                summary['human_collected'] = 'skipped'
            
            # Step 2: Generate AI text
            if not skip_generation:
                summary['ai_generated'] = self.generate_ai_text(quick_mode=quick_mode)
            else:
                logger.info("Skipping AI text generation")
                summary['ai_generated'] = 'skipped'
            
            # Step 3: Preprocess
            summary['preprocessed'] = self.preprocess_data()
            
            # Step 4: Merge and split
            summary['splits'] = self.merge_and_split()
            
            # Step 5: Validate
            summary['validation'] = self.validate_dataset()
            
            summary['status'] = 'success'
            
        except Exception as e:
            logger.error(f"Build failed: {e}")
            summary['status'] = 'failed'
            summary['error'] = str(e)
        
        elapsed = time.time() - start_time
        summary['elapsed_time'] = f"{elapsed:.1f}s"
        summary['end_time'] = datetime.now().isoformat()
        
        # Save summary
        summary_path = self.output_dir / "build_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("="*60)
        logger.info(f"BUILD COMPLETE in {elapsed:.1f}s")
        logger.info(f"Summary saved to: {summary_path}")
        logger.info("="*60)
        
        return summary


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for dataset building."""
    parser = argparse.ArgumentParser(
        description="Build the ManavAI training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build complete dataset (100K samples)
  python -m data.build_dataset --target 100000
  
  # Quick build for testing (5K samples)
  python -m data.build_dataset --target 5000 --quick
  
  # Only preprocess existing data
  python -m data.build_dataset --skip-collection --skip-generation
        """
    )
    
    parser.add_argument(
        "--target", type=int, default=100000,
        help="Target total samples (default: 100000)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: use only small models for faster generation"
    )
    parser.add_argument(
        "--skip-collection", action="store_true",
        help="Skip human text collection"
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip AI text generation"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else None
    
    builder = DatasetBuilder(
        output_dir=output_dir,
        target_samples=args.target
    )
    
    summary = builder.build(
        skip_collection=args.skip_collection,
        skip_generation=args.skip_generation,
        quick_mode=args.quick
    )
    
    print(f"\nBuild status: {summary.get('status', 'unknown')}")
    if 'splits' in summary:
        print(f"Final dataset splits: {summary['splits']}")


if __name__ == "__main__":
    main()