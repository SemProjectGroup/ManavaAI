"""
Train-Test Split Script for AI Text Humanizer Project
Splits cleaned dataset into train/validation/test sets with stratification
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetSplitter:
    """Handle splitting of dataset into train/validation/test sets"""
    
    def __init__(
        self,
        input_file: str,
        output_dir: str = "data/splits",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize the dataset splitter
        
        Args:
            input_file: Path to cleaned dataset (JSON or CSV)
            output_dir: Directory to save split datasets
            train_ratio: Proportion for training set (default: 0.8)
            val_ratio: Proportion for validation set (default: 0.1)
            test_ratio: Proportion for test set (default: 0.1)
            random_state: Random seed for reproducibility
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        # Validate ratios
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self) -> pd.DataFrame:
        """Load dataset from JSON or CSV file"""
        logger.info(f"Loading dataset from {self.input_file}")
        
        file_ext = Path(self.input_file).suffix.lower()
        
        if file_ext == '.json':
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif file_ext == '.csv':
            df = pd.read_csv(self.input_file)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Use .json or .csv")
        
        logger.info(f"Loaded {len(df)} samples")
        return df
    
    def validate_dataset(self, df: pd.DataFrame) -> None:
        """Validate dataset has required columns and proper format"""
        required_columns = ['text', 'label']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for null values
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            logger.warning(f"Found null values:\n{null_counts[null_counts > 0]}")
            logger.info("Dropping rows with null values...")
            df.dropna(subset=required_columns, inplace=True)
        
        # Validate labels
        unique_labels = df['label'].unique()
        expected_labels = ['human', 'ai']
        
        if not all(label in expected_labels for label in unique_labels):
            logger.warning(f"Unexpected labels found: {unique_labels}")
            logger.info("Expected labels: 'human' and 'ai'")
    
    def get_label_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get distribution of labels in dataset"""
        distribution = df['label'].value_counts().to_dict()
        return distribution
    
    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets with stratification
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting dataset...")
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_ratio,
            random_state=self.random_state,
            stratify=df['label']  # Ensure balanced labels
        )
        
        # Second split: separate validation from training
        # Adjust validation ratio relative to remaining data
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            random_state=self.random_state,
            stratify=train_val_df['label']
        )
        
        return train_df, val_df, test_df
    
    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        format: str = 'json'
    ) -> None:
        """
        Save split datasets to files
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            format: Output format ('json' or 'csv')
        """
        logger.info(f"Saving splits to {self.output_dir} in {format} format...")
        
        splits = {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
        
        for split_name, split_df in splits.items():
            output_path = os.path.join(self.output_dir, f"{split_name}.{format}")
            
            if format == 'json':
                split_df.to_json(
                    output_path,
                    orient='records',
                    indent=2,
                    force_ascii=False
                )
            elif format == 'csv':
                split_df.to_csv(output_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Saved {split_name} set: {len(split_df)} samples → {output_path}")
    
    def print_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """Print statistics about the splits"""
        total_samples = len(train_df) + len(val_df) + len(test_df)
        
        print("\n" + "="*60)
        print("DATASET SPLIT STATISTICS")
        print("="*60)
        print(f"Total Samples: {total_samples}")
        print(f"\nSplit Ratios:")
        print(f"  Train:      {len(train_df):6d} ({len(train_df)/total_samples*100:.1f}%)")
        print(f"  Validation: {len(val_df):6d} ({len(val_df)/total_samples*100:.1f}%)")
        print(f"  Test:       {len(test_df):6d} ({len(test_df)/total_samples*100:.1f}%)")
        
        print(f"\nLabel Distribution:")
        for split_name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
            dist = self.get_label_distribution(split_df)
            print(f"  {split_name}:")
            for label, count in dist.items():
                print(f"    {label}: {count:6d} ({count/len(split_df)*100:.1f}%)")
        
        print("="*60 + "\n")
    
    def run(self, output_format: str = 'json') -> None:
        """Execute the complete splitting process"""
        try:
            # Load and validate dataset
            df = self.load_dataset()
            self.validate_dataset(df)
            
            # Show initial distribution
            logger.info(f"Initial label distribution: {self.get_label_distribution(df)}")
            
            # Split dataset
            train_df, val_df, test_df = self.split_dataset(df)
            
            # Save splits
            self.save_splits(train_df, val_df, test_df, format=output_format)
            
            # Print statistics
            self.print_statistics(train_df, val_df, test_df)
            
            logger.info("Dataset splitting completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during dataset splitting: {str(e)}")
            raise


def main():
    """Main execution function"""
    
    # Configuration
    INPUT_FILE = "data/cleaned_dataset.json"  # Path to your cleaned dataset
    OUTPUT_DIR = "data/splits"
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    OUTPUT_FORMAT = 'json'  # 'json' or 'csv'
    
    # Create splitter and run
    splitter = DatasetSplitter(
        input_file=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_state=42
    )
    
    splitter.run(output_format=OUTPUT_FORMAT)


if __name__ == "__main__":
    main()