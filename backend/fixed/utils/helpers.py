"""
General helper utilities for ManavAI.
"""

import os
import sys
import json
import random
import logging
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path
import torch
import numpy as np


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional file to log to
        format_string: Custom format string
    
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch.
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_jsonl(filepath: str) -> Generator[Dict, None, None]:
    """
    Load a JSONL file as a generator.
    
    Args:
        filepath: Path to JSONL file
    
    Yields:
        Dictionary for each line
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(data: List[Dict], filepath: str, mode: str = 'w'):
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        filepath: Path to output file
        mode: File mode ('w' for write, 'a' for append)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def chunk_text(
    text: str, 
    chunk_size: int = 500, 
    overlap: int = 50,
    by_sentences: bool = True
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to split
        chunk_size: Target chunk size (in words or sentences)
        overlap: Overlap between chunks
        by_sentences: If True, chunk by sentences; otherwise by words
    
    Returns:
        List of text chunks
    """
    import re
    
    if by_sentences:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_length + sentence_words > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep overlap sentences
                overlap_sentences = int(overlap / max(1, current_length / len(current_chunk)))
                current_chunk = current_chunk[-overlap_sentences:]
                current_length = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_words
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    else:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = words[i:i + chunk_size]
            chunks.append(' '.join(chunk))
        
        return chunks


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'total_millions': total / 1e6,
        'trainable_millions': trainable / 1e6
    }


def print_model_summary(model: torch.nn.Module, model_name: str = "Model"):
    """
    Print a summary of model architecture and parameters.
    
    Args:
        model: PyTorch model
        model_name: Name to display
    """
    params = count_parameters(model)
    
    print(f"\n{'='*50}")
    print(f"{model_name} Summary")
    print(f"{'='*50}")
    print(f"Total parameters:     {params['total']:,} ({params['total_millions']:.2f}M)")
    print(f"Trainable parameters: {params['trainable']:,} ({params['trainable_millions']:.2f}M)")
    print(f"Frozen parameters:    {params['frozen']:,}")
    print(f"{'='*50}\n")


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
    
    Returns:
        The path string
    """
    os.makedirs(path, exist_ok=True)
    return path


def file_exists(path: str) -> bool:
    """Check if file exists."""
    return os.path.isfile(path)


def dir_exists(path: str) -> bool:
    """Check if directory exists."""
    return os.path.isdir(path)


def get_file_size(path: str) -> str:
    """
    Get human-readable file size.
    
    Args:
        path: File path
    
    Returns:
        Formatted size string
    """
    size = os.path.getsize(path)
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    
    return f"{size:.2f} PB"


def progress_bar(
    iterable,
    desc: str = "",
    total: Optional[int] = None,
    disable: bool = False
):
    """
    Create a progress bar.
    
    Args:
        iterable: Iterable to wrap
        desc: Description text
        total: Total count (optional)
        disable: If True, disable the progress bar
    
    Returns:
        tqdm progress bar or plain iterable
    """
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, disable=disable)
    except ImportError:
        # Fallback if tqdm not installed
        return iterable


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(
        self, 
        patience: int = 5, 
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop training.
        
        Args:
            score: Current metric value
        
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def main():
    """Test helper functions."""
    print("Testing helper functions...")
    
    # Test set_seed
    set_seed(42)
    print(f"Random number: {random.random()}")
    
    # Test device detection
    device = get_device()
    
    # Test time formatting
    print(f"30 seconds: {format_time(30)}")
    print(f"90 seconds: {format_time(90)}")
    print(f"3700 seconds: {format_time(3700)}")
    
    # Test chunking
    test_text = "This is sentence one. This is sentence two. " * 50
    chunks = chunk_text(test_text, chunk_size=100, overlap=20)
    print(f"Created {len(chunks)} chunks from text")
    
    print("All tests passed!")


if __name__ == '__main__':
    main()
    