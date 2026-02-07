"""
PyTorch Dataset Module
======================
Provides dataset classes and data loading utilities for training.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from loguru import logger

try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config


class ManavAIDataset(Dataset):
    """
    PyTorch Dataset for AI detection training.
    
    Attributes:
        data: List of samples with text and labels
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        include_metadata: Whether to include metadata in output
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Optional["PreTrainedTokenizer"] = None,
        tokenizer_name: Optional[str] = None,
        max_length: int = 512,
        include_metadata: bool = False,
        lazy_loading: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to JSONL data file
            tokenizer: Pre-initialized tokenizer (optional)
            tokenizer_name: Tokenizer name to load (optional)
            max_length: Maximum sequence length
            include_metadata: Whether to include metadata
            lazy_loading: Whether to load data lazily
        """
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.include_metadata = include_metadata
        self.lazy_loading = lazy_loading
        
        # Initialize tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_name is not None and TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = None
        
        # Load data
        if lazy_loading:
            # Count lines for length
            self.data = None
            self._length = self._count_lines()
        else:
            self.data = self._load_data()
            self._length = len(self.data)
        
        logger.info(f"Dataset initialized with {self._length} samples")
    
    def _count_lines(self) -> int:
        """Count lines in the data file."""
        count = 0
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count
    
    def _load_data(self) -> List[Dict]:
        """Load all data into memory."""
        data = []
        
        if not self.data_path.exists():
            logger.warning(f"Data file not found: {self.data_path}")
            return data
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading dataset"):
                try:
                    sample = json.loads(line.strip())
                    data.append(sample)
                except json.JSONDecodeError:
                    continue
        
        return data
    
    def _get_sample(self, idx: int) -> Dict:
        """Get a sample by index."""
        if self.lazy_loading:
            # Read specific line
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i == idx:
                        return json.loads(line.strip())
            return {}
        else:
            return self.data[idx]
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample.
        
        Returns:
            Dictionary with:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - labels: Label (0=human, 1=AI)
                - text: Original text (if include_metadata)
                - metadata: Sample metadata (if include_metadata)
        """
        sample = self._get_sample(idx)
        text = sample.get('text', '')
        label = sample.get('label', 0)
        
        result = {'labels': torch.tensor(label, dtype=torch.long)}
        
        # Tokenize if tokenizer available
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            result['input_ids'] = encoding['input_ids'].squeeze(0)
            result['attention_mask'] = encoding['attention_mask'].squeeze(0)
            
            if 'token_type_ids' in encoding:
                result['token_type_ids'] = encoding['token_type_ids'].squeeze(0)
        else:
            result['text'] = text
        
        # Include metadata if requested
        if self.include_metadata:
            result['text'] = text
            result['source'] = sample.get('source', '')
            result['metadata'] = sample.get('metadata', {})
        
        return result


class HumanizerDataset(Dataset):
    """
    Dataset for humanizer model training.
    
    Uses AI-generated text as input and human text as target.
    """
    
    def __init__(
        self,
        ai_data_path: Union[str, Path],
        human_data_path: Union[str, Path],
        tokenizer: Optional["PreTrainedTokenizer"] = None,
        tokenizer_name: Optional[str] = None,
        max_input_length: int = 512,
        max_output_length: int = 512,
    ):
        """
        Initialize the humanizer dataset.
        
        Args:
            ai_data_path: Path to AI text JSONL
            human_data_path: Path to human text JSONL
            tokenizer: Pre-initialized tokenizer
            tokenizer_name: Tokenizer name to load
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
        """
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        # Initialize tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_name is not None and TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            raise ValueError("Tokenizer required for HumanizerDataset")
        
        # Load data
        self.ai_texts = self._load_texts(Path(ai_data_path))
        self.human_texts = self._load_texts(Path(human_data_path))
        
        # Balance datasets
        min_len = min(len(self.ai_texts), len(self.human_texts))
        random.shuffle(self.ai_texts)
        random.shuffle(self.human_texts)
        self.ai_texts = self.ai_texts[:min_len]
        self.human_texts = self.human_texts[:min_len]
        
        logger.info(f"HumanizerDataset initialized with {min_len} pairs")
    
    def _load_texts(self, path: Path) -> List[str]:
        """Load texts from a JSONL file."""
        texts = []
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        texts.append(data.get('text', ''))
                    except:
                        continue
        return texts
    
    def __len__(self) -> int:
        return len(self.ai_texts)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a training pair.
        
        Returns:
            Dictionary with input and target encodings.
        """
        ai_text = self.ai_texts[idx]
        human_text = self.human_texts[idx]
        
        # Add task prefix
        input_text = f"humanize: {ai_text}"
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            human_text,
            max_length=self.max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Replace padding token ID with -100 for loss calculation
        labels = target_encoding['input_ids'].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'labels': labels,
        }


def create_dataloaders(
    train_path: Union[str, Path],
    val_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
    tokenizer_name: str = "microsoft/deberta-v3-base",
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 4,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create DataLoaders for training.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data (optional)
        test_path: Path to test data (optional)
        tokenizer_name: Tokenizer to use
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Create datasets
    train_dataset = ManavAIDataset(
        train_path,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_path and Path(val_path).exists():
        val_dataset = ManavAIDataset(
            val_path,
            tokenizer=tokenizer,
            max_length=max_length
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    test_loader = None
    if test_path and Path(test_path).exists():
        test_dataset = ManavAIDataset(
            test_path,
            tokenizer=tokenizer,
            max_length=max_length
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Test dataset loading."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("--data", type=str, required=True, help="Data file path")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    
    args = parser.parse_args()
    
    # Create dataset
    dataset = ManavAIDataset(
        args.data,
        tokenizer_name=args.tokenizer,
        include_metadata=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Label: {sample['labels']}")
        if 'text' in sample:
            print(f"Text preview: {sample['text'][:200]}...")
    
    # Test DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size)
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    for key, value in batch.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")


if __name__ == "__main__":
    main()