# data_loader.py
"""
Data loading and preprocessing utilities
"""

import os
import gc
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from config import Config


class TextClassificationDataset(Dataset):
    """PyTorch Dataset for text classification"""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DataManager:
    """Manages data loading and preprocessing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
    
    def load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer"""
        if self.tokenizer is None:
            print(f"Loading tokenizer: {self.config.model.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.model_name
            )
        return self.tokenizer
    
    def load_raw_data(self) -> Tuple[List[str], List[int]]:
        """Load raw data from parquet files"""
        data_dir = self.config.data.data_dir
        max_per_class = self.config.data.max_samples_per_class
        
        print(f"\nLoading data from: {data_dir}")
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                "Please run the data collection script first."
            )
        
        # Find parquet files
        files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
        
        if not files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")
        
        print(f"Found {len(files)} parquet files")
        
        human_texts = []
        ai_texts = []
        
        # Load files
        for filename in tqdm(files, desc="Loading files"):
            filepath = os.path.join(data_dir, filename)
            
            try:
                df = pd.read_parquet(filepath)
                
                if 'text' not in df.columns:
                    continue
                
                # Filter by minimum length
                df = df[df['text'].str.len() >= self.config.data.min_text_length]
                
                # Categorize by label
                if 'label' in df.columns:
                    human_df = df[df['label'] == 'human']
                    ai_df = df[df['label'] == 'ai']
                    human_texts.extend(human_df['text'].tolist())
                    ai_texts.extend(ai_df['text'].tolist())
                elif 'label_int' in df.columns:
                    human_df = df[df['label_int'] == 0]
                    ai_df = df[df['label_int'] == 1]
                    human_texts.extend(human_df['text'].tolist())
                    ai_texts.extend(ai_df['text'].tolist())
                elif 'human' in filename.lower():
                    human_texts.extend(df['text'].tolist())
                elif 'ai' in filename.lower():
                    ai_texts.extend(df['text'].tolist())
                
                del df
                gc.collect()
                
            except Exception as e:
                print(f"  Warning: Could not load {filename}: {e}")
                continue
        
        print(f"\nLoaded:")
        print(f"  Human texts: {len(human_texts):,}")
        print(f"  AI texts: {len(ai_texts):,}")
        
        # Balance dataset
        min_size = min(len(human_texts), len(ai_texts), max_per_class)
        print(f"\nBalancing to {min_size:,} samples per class...")
        
        np.random.seed(self.config.data.seed)
        
        if len(human_texts) > min_size:
            indices = np.random.choice(len(human_texts), min_size, replace=False)
            human_texts = [human_texts[i] for i in indices]
        
        if len(ai_texts) > min_size:
            indices = np.random.choice(len(ai_texts), min_size, replace=False)
            ai_texts = [ai_texts[i] for i in indices]
        
        # Combine
        texts = human_texts + ai_texts
        labels = [0] * len(human_texts) + [1] * len(ai_texts)  # 0=Human, 1=AI
        
        # Shuffle
        combined = list(zip(texts, labels))
        np.random.shuffle(combined)
        texts, labels = zip(*combined)
        
        print(f"Final dataset: {len(texts):,} samples")
        
        return list(texts), list(labels)
    
    def create_splits(
        self,
        texts: List[str],
        labels: List[int]
    ) -> Dict[str, Tuple[List[str], List[int]]]:
        """Create train/val/test splits"""
        
        print("\nCreating data splits...")
        
        # Test split
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels,
            test_size=self.config.data.test_size,
            random_state=self.config.data.seed,
            stratify=labels
        )
        
        # Validation split
        val_ratio = self.config.data.val_size / (1 - self.config.data.test_size)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels,
            test_size=val_ratio,
            random_state=self.config.data.seed,
            stratify=train_val_labels
        )
        
        splits = {
            'train': (train_texts, train_labels),
            'val': (val_texts, val_labels),
            'test': (test_texts, test_labels)
        }
        
        print(f"  Train: {len(train_texts):,}")
        print(f"  Val:   {len(val_texts):,}")
        print(f"  Test:  {len(test_texts):,}")
        
        return splits
    
    def create_dataloaders(
        self,
        splits: Dict[str, Tuple[List[str], List[int]]]
    ) -> Dict[str, DataLoader]:
        """Create PyTorch DataLoaders"""
        
        tokenizer = self.load_tokenizer()
        dataloaders = {}
        
        for split_name, (texts, labels) in splits.items():
            dataset = TextClassificationDataset(
                texts=texts,
                labels=labels,
                tokenizer=tokenizer,
                max_length=self.config.model.max_length
            )
            
            is_train = split_name == 'train'
            
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=self.config.training.batch_size,
                shuffle=is_train,
                num_workers=self.config.training.num_workers,
                pin_memory=self.config.training.device == 'cuda'
            )
        
        return dataloaders
    
    def prepare_data(self) -> Tuple[Dict[str, DataLoader], Dict[str, Tuple]]:
        """Complete data preparation pipeline"""
        
        # Load raw data
        texts, labels = self.load_raw_data()
        
        # Create splits
        splits = self.create_splits(texts, labels)
        
        # Clear memory
        del texts, labels
        gc.collect()
        
        # Create dataloaders
        dataloaders = self.create_dataloaders(splits)
        
        return dataloaders, splits