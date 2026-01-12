#!/usr/bin/env python
"""
==============================================================================
AI TEXT DETECTOR - ULTIMATE TRAINING SCRIPT
==============================================================================

First run:  Creates optimized cache (one-time, ~5-10 min)
Next runs:  Loads instantly from cache (~2 seconds)

Usage:
    python train.py --samples 10000 --epochs 1      # Quick test
    python train.py --samples 50000 --epochs 2      # Demo
    python train.py --samples 200000 --epochs 3     # Production
    python train.py --interactive                   # Test model
    python train.py --rebuild-cache                 # Force rebuild cache

==============================================================================
"""

import os
import sys
import gc
import json
import argparse
import warnings
import pickle
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

try:
    from colorama import init, Fore, Style
    init()
except ImportError:
    class Dummy:
        RED = GREEN = YELLOW = CYAN = RESET_ALL = ''
    Fore = Style = Dummy()


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data paths
    RAW_DATA_DIR = r"F:\data\dataset_output_20x"     # Your original data
    CACHE_DIR = r"F:\data\ai_detector_cache"         # Cache location (same drive = fast)
    
    # Model
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 256
    
    # Training defaults
    SAMPLES_PER_CLASS = 50_000
    BATCH_SIZE = 16
    EPOCHS = 2
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # Cache settings
    CACHE_SIZE = 500_000  # Cache up to 500K samples per class
    
    # Output
    OUTPUT_DIR = "trained_model"
    
    # System
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FP16 = torch.cuda.is_available()
    SEED = 42


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# SMART DATA LOADER - Uses cache if available, creates it if not
# ============================================================================

class SmartDataLoader:
    """
    Smart data loading with automatic caching.
    First run: Loads from parquet files and creates cache (~5-10 min)
    Next runs: Loads from cache instantly (~2 seconds)
    """
    
    def __init__(self, raw_dir: str, cache_dir: str, cache_size: int = 500_000):
        self.raw_dir = raw_dir
        self.cache_dir = cache_dir
        self.cache_size = cache_size
        
        self.human_cache = os.path.join(cache_dir, 'human_texts.pkl')
        self.ai_cache = os.path.join(cache_dir, 'ai_texts.pkl')
        self.info_file = os.path.join(cache_dir, 'cache_info.json')
    
    def cache_exists(self) -> bool:
        """Check if valid cache exists"""
        return (
            os.path.exists(self.human_cache) and
            os.path.exists(self.ai_cache) and
            os.path.exists(self.info_file)
        )
    
    def build_cache(self):
        """Build cache from raw parquet files (one-time operation)"""
        
        print(f"\n{Fore.YELLOW}{'='*60}")
        print("BUILDING DATA CACHE (One-time operation)")
        print(f"{'='*60}{Style.RESET_ALL}")
        print(f"  Source: {self.raw_dir}")
        print(f"  Cache:  {self.cache_dir}")
        print(f"  Target: {self.cache_size:,} samples per class")
        print(f"\n  This will take ~5-10 minutes but only runs ONCE.")
        print(f"  Future training will load in ~2 seconds!\n")
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if not os.path.exists(self.raw_dir):
            print(f"{Fore.RED}ERROR: Data directory not found: {self.raw_dir}{Style.RESET_ALL}")
            sys.exit(1)
        
        files = sorted([f for f in os.listdir(self.raw_dir) if f.endswith('.parquet')])
        print(f"  Found {len(files)} parquet files\n")
        
        human_texts = []
        ai_texts = []
        
        pbar = tqdm(files, desc="Building cache")
        
        for filename in pbar:
            # Stop early if we have enough
            if len(human_texts) >= self.cache_size and len(ai_texts) >= self.cache_size:
                break
            
            filepath = os.path.join(self.raw_dir, filename)
            
            try:
                df = pd.read_parquet(filepath)
                
                if 'text' not in df.columns:
                    continue
                
                # Filter
                df = df[df['text'].str.len() >= 20]
                df = df[df['text'].str.len() <= 10000]
                
                # Extract texts
                if 'label' in df.columns:
                    h = df[df['label'] == 'human']['text'].tolist()
                    a = df[df['label'] == 'ai']['text'].tolist()
                elif 'label_int' in df.columns:
                    h = df[df['label_int'] == 0]['text'].tolist()
                    a = df[df['label_int'] == 1]['text'].tolist()
                elif 'human' in filename.lower():
                    h = df['text'].tolist()
                    a = []
                elif 'ai' in filename.lower():
                    h = []
                    a = df['text'].tolist()
                else:
                    continue
                
                # Add only what we need
                need_h = self.cache_size - len(human_texts)
                need_a = self.cache_size - len(ai_texts)
                
                human_texts.extend(h[:need_h])
                ai_texts.extend(a[:need_a])
                
                pbar.set_postfix({'human': len(human_texts), 'ai': len(ai_texts)})
                
                del df
                gc.collect()
                
            except Exception as e:
                continue
        
        pbar.close()
        
        print(f"\n  Collected: {len(human_texts):,} human, {len(ai_texts):,} AI")
        
        # Shuffle
        np.random.seed(Config.SEED)
        np.random.shuffle(human_texts)
        np.random.shuffle(ai_texts)
        
        # Save cache
        print(f"\n  Saving cache...")
        
        with open(self.human_cache, 'wb') as f:
            pickle.dump(human_texts, f)
        
        with open(self.ai_cache, 'wb') as f:
            pickle.dump(ai_texts, f)
        
        info = {
            'human_count': len(human_texts),
            'ai_count': len(ai_texts),
            'created': datetime.now().isoformat(),
            'source': self.raw_dir
        }
        
        with open(self.info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        # Calculate sizes
        h_size = os.path.getsize(self.human_cache) / (1024**2)
        a_size = os.path.getsize(self.ai_cache) / (1024**2)
        
        print(f"\n{Fore.GREEN}  ✓ Cache created!{Style.RESET_ALL}")
        print(f"    Human: {len(human_texts):,} texts ({h_size:.1f} MB)")
        print(f"    AI:    {len(ai_texts):,} texts ({a_size:.1f} MB)")
        print(f"{'='*60}\n")
        
        return human_texts, ai_texts
    
    def load_from_cache(self, samples_per_class: int):
        """Load data from cache (instant!)"""
        
        print(f"\n{'='*60}")
        print("LOADING DATA FROM CACHE")
        print(f"{'='*60}")
        
        # Load info
        with open(self.info_file, 'r') as f:
            info = json.load(f)
        
        print(f"  Cache: {self.cache_dir}")
        print(f"  Available: {info['human_count']:,} human, {info['ai_count']:,} AI")
        print(f"  Requested: {samples_per_class:,} per class")
        
        # Load texts
        start = datetime.now()
        
        with open(self.human_cache, 'rb') as f:
            human_texts = pickle.load(f)
        
        with open(self.ai_cache, 'rb') as f:
            ai_texts = pickle.load(f)
        
        load_time = (datetime.now() - start).total_seconds()
        
        # Sample what we need
        if len(human_texts) > samples_per_class:
            human_texts = human_texts[:samples_per_class]
        if len(ai_texts) > samples_per_class:
            ai_texts = ai_texts[:samples_per_class]
        
        # Balance
        min_size = min(len(human_texts), len(ai_texts))
        human_texts = human_texts[:min_size]
        ai_texts = ai_texts[:min_size]
        
        print(f"\n  {Fore.GREEN}✓ Loaded {min_size*2:,} samples in {load_time:.1f} seconds!{Style.RESET_ALL}")
        print(f"{'='*60}")
        
        return human_texts, ai_texts
    
    def load(self, samples_per_class: int, force_rebuild: bool = False):
        """Main load function - uses cache or builds it"""
        
        if force_rebuild or not self.cache_exists():
            human_texts, ai_texts = self.build_cache()
            
            # Trim to requested size
            human_texts = human_texts[:samples_per_class]
            ai_texts = ai_texts[:samples_per_class]
        else:
            human_texts, ai_texts = self.load_from_cache(samples_per_class)
        
        # Balance
        min_size = min(len(human_texts), len(ai_texts))
        human_texts = human_texts[:min_size]
        ai_texts = ai_texts[:min_size]
        
        # Combine and shuffle
        texts = human_texts + ai_texts
        labels = [0] * len(human_texts) + [1] * len(ai_texts)
        
        combined = list(zip(texts, labels))
        np.random.seed(Config.SEED)
        np.random.shuffle(combined)
        texts, labels = zip(*combined)
        
        return list(texts), list(labels)


# ============================================================================
# DATASET
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    """Main training function"""
    
    # Update config
    Config.SAMPLES_PER_CLASS = args.samples
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    
    if args.data_dir:
        Config.RAW_DATA_DIR = args.data_dir
    if args.cache_dir:
        Config.CACHE_DIR = args.cache_dir
    if args.output_dir:
        Config.OUTPUT_DIR = args.output_dir
    if args.model:
        Config.MODEL_NAME = args.model
    
    set_seed(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Header
    print(f"\n{Fore.CYAN}{'='*60}")
    print("   AI TEXT DETECTOR - TRAINING")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    # System info
    print(f"\n[System]")
    print(f"  Device: {Config.DEVICE}")
    if Config.DEVICE == "cuda":
        print(f"  {Fore.GREEN}✓ GPU: {torch.cuda.get_device_name(0)}{Style.RESET_ALL}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"  {Fore.YELLOW}⚠ CPU mode (slow){Style.RESET_ALL}")
    
    print(f"\n[Config]")
    print(f"  Model:      {Config.MODEL_NAME}")
    print(f"  Samples:    {Config.SAMPLES_PER_CLASS:,} per class")
    print(f"  Epochs:     {Config.EPOCHS}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    
    # Load data
    loader = SmartDataLoader(
        Config.RAW_DATA_DIR,
        Config.CACHE_DIR,
        Config.CACHE_SIZE
    )
    
    texts, labels = loader.load(Config.SAMPLES_PER_CLASS, force_rebuild=args.rebuild_cache)
    
    # Split
    print(f"\n{'='*60}")
    print("PREPARING SPLITS")
    print(f"{'='*60}")
    
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=Config.SEED, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=Config.SEED, stratify=temp_labels
    )
    
    print(f"  Train: {len(train_texts):,}")
    print(f"  Val:   {len(val_texts):,}")
    print(f"  Test:  {len(test_texts):,}")
    
    del texts, labels, temp_texts, temp_labels
    gc.collect()
    
    # Load model
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME, num_labels=2)
    model.to(Config.DEVICE)
    
    print(f"  Model: {Config.MODEL_NAME}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Dataloaders
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, Config.MAX_LENGTH)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, Config.MAX_LENGTH)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, Config.MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                              num_workers=0, pin_memory=Config.DEVICE=="cuda")
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE*2, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE*2, num_workers=0)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    total_steps = len(train_loader) * Config.EPOCHS
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler() if Config.FP16 else None
    
    # Training
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")
    print(f"  Steps/epoch: {len(train_loader):,}")
    print(f"  Total steps: {total_steps:,}")
    
    best_f1 = 0.0
    start_time = datetime.now()
    
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels_batch = batch['labels'].to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            if Config.FP16:
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{total_loss/(step+1):.4f}'})
        
        # Validation
        model.eval()
        val_preds, val_true = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attention_mask = batch['attention_mask'].to(Config.DEVICE)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(batch['labels'].numpy())
        
        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds)
        
        print(f"\n  Epoch {epoch+1}: Acc={val_acc:.4f}, F1={val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_path = os.path.join(Config.OUTPUT_DIR, 'best')
            os.makedirs(best_path, exist_ok=True)
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"  {Fore.GREEN}✓ Saved best model (F1={best_f1:.4f}){Style.RESET_ALL}")
    
    train_time = datetime.now() - start_time
    
    # Save final
    final_path = os.path.join(Config.OUTPUT_DIR, 'final')
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Test
    print(f"\n{'='*60}")
    print("TESTING")
    print(f"{'='*60}")
    
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(Config.OUTPUT_DIR, 'best'))
    model.to(Config.DEVICE)
    model.eval()
    
    test_preds, test_true = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(batch['labels'].numpy())
    
    test_acc = accuracy_score(test_true, test_preds)
    test_prec = precision_score(test_true, test_preds)
    test_rec = recall_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds)
    cm = confusion_matrix(test_true, test_preds)
    
    print(f"\n  {Fore.CYAN}Results:{Style.RESET_ALL}")
    print(f"  ─────────────────────────")
    print(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"             Pred:Human  Pred:AI")
    print(f"  Human        {cm[0][0]:5d}     {cm[0][1]:5d}")
    print(f"  AI           {cm[1][0]:5d}     {cm[1][1]:5d}")
    
    # Demo
    print(f"\n{'='*60}")
    print("DEMO")
    print(f"{'='*60}")
    
    demos = [
        "I went to the store today and bought some milk.",
        "The implementation of AI represents a paradigm shift in computational methodologies.",
        "lol cant believe that happened!! telling everyone tomorrow",
        "Furthermore, the aforementioned considerations necessitate comprehensive examination.",
    ]
    
    for text in demos:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                          max_length=Config.MAX_LENGTH, padding=True).to(Config.DEVICE)
        with torch.no_grad():
            probs = F.softmax(model(**inputs).logits, dim=1)
            ai_prob = probs[0][1].item()
        
        icon = f"{Fore.RED}🤖 AI" if ai_prob > 0.5 else f"{Fore.GREEN}👤 Human"
        print(f"\n  \"{text[:50]}...\"")
        print(f"   → {icon}{Style.RESET_ALL} ({ai_prob*100:.1f}% AI)")
    
    # Summary
    print(f"\n{Fore.GREEN}{'='*60}")
    print("✓ COMPLETE!")
    print(f"{'='*60}{Style.RESET_ALL}")
    print(f"  Time:     {train_time}")
    print(f"  Best F1:  {best_f1:.4f}")
    print(f"  Test Acc: {test_acc:.1%}")
    print(f"  Model:    {Config.OUTPUT_DIR}")
    print(f"\n  Test it: python train.py --interactive")
    print(f"{'='*60}")
    
    # Save results
    with open(os.path.join(Config.OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump({
            'time': str(train_time),
            'samples': Config.SAMPLES_PER_CLASS,
            'epochs': Config.EPOCHS,
            'best_f1': best_f1,
            'test_acc': test_acc,
            'test_f1': test_f1
        }, f, indent=2)


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive(model_path=None):
    """Interactive detection"""
    
    if model_path is None:
        for path in [os.path.join(Config.OUTPUT_DIR, 'best'), 
                     os.path.join(Config.OUTPUT_DIR, 'final')]:
            if os.path.exists(path):
                model_path = path
                break
    
    if not model_path or not os.path.exists(model_path):
        print(f"{Fore.RED}No model found! Train first:{Style.RESET_ALL}")
        print("  python train.py --samples 10000 --epochs 1")
        return
    
    print(f"\nLoading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(Config.DEVICE)
    model.eval()
    print(f"{Fore.GREEN}✓ Ready!{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}{'='*60}")
    print("   AI TEXT DETECTOR")
    print(f"{'='*60}{Style.RESET_ALL}")
    print("Enter text, press Enter twice. Type 'quit' to exit.\n")
    
    while True:
        try:
            print(f"{Fore.YELLOW}Text:{Style.RESET_ALL}")
            lines = []
            while True:
                line = input()
                if line.lower() in ['quit', 'exit', 'q']:
                    print(f"\n{Fore.CYAN}Bye!{Style.RESET_ALL}")
                    return
                if line == '' and lines:
                    break
                lines.append(line)
            
            text = ' '.join(lines).strip()
            if len(text) < 10:
                print(f"{Fore.RED}Too short!{Style.RESET_ALL}\n")
                continue
            
            inputs = tokenizer(text, return_tensors='pt', truncation=True,
                              max_length=Config.MAX_LENGTH, padding=True).to(Config.DEVICE)
            
            with torch.no_grad():
                probs = F.softmax(model(**inputs).logits, dim=1)
                human_prob = probs[0][0].item()
                ai_prob = probs[0][1].item()
            
            print(f"\n{'='*50}")
            if ai_prob > 0.5:
                print(f"{Fore.RED}🤖 AI GENERATED{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}👤 HUMAN WRITTEN{Style.RESET_ALL}")
            print(f"{'='*50}")
            
            h_bar = int(human_prob * 20)
            a_bar = int(ai_prob * 20)
            print(f"\n  Human: {Fore.GREEN}{'█'*h_bar}{'░'*(20-h_bar)}{Style.RESET_ALL} {human_prob*100:.1f}%")
            print(f"  AI:    {Fore.RED}{'█'*a_bar}{'░'*(20-a_bar)}{Style.RESET_ALL} {ai_prob*100:.1f}%")
            print(f"{'='*50}\n")
            
        except KeyboardInterrupt:
            print(f"\n{Fore.CYAN}Bye!{Style.RESET_ALL}")
            break


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AI Text Detector Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --samples 10000 --epochs 1    # Quick (~5 min)
  python train.py --samples 50000 --epochs 2    # Demo (~20 min)  
  python train.py --samples 200000 --epochs 3   # Production (~1.5 hr)
  python train.py --interactive                 # Test model
  python train.py --rebuild-cache               # Rebuild data cache
        """
    )
    
    parser.add_argument('--samples', type=int, default=50000)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--cache-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--model', type=str, default=None,
                        choices=['distilbert-base-uncased', 'roberta-base', 'bert-base-uncased'])
    parser.add_argument('--interactive', '-i', action='store_true')
    parser.add_argument('--rebuild-cache', action='store_true', help='Force rebuild data cache')
    parser.add_argument('--model-path', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive(args.model_path)
    else:
        train(args)


if __name__ == "__main__":
    main()