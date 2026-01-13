"""
================================================================================
ENHANCED AI DETECTOR TRAINING
================================================================================
Best practices for maximum accuracy:
- Larger model (RoBERTa or DeBERTa)
- Longer sequences (512 tokens)
- Gradient accumulation (larger effective batch)
- Cosine learning rate schedule
- Mixed precision training
- Early stopping
- Data augmentation
================================================================================
"""

import os
import sys
import gc
import json
import pickle
import random
import argparse
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)

try:
    from colorama import init, Fore, Style
    init()
except:
    class D:
        RED = GREEN = YELLOW = CYAN = RESET_ALL = ''
    Fore = Style = D()


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    # Data
    CACHE_DIR = r"F:\data\mega_ai_dataset"
    
    # Model - Use best model
    MODEL_NAME = "roberta-base"  # or "microsoft/deberta-v3-base"
    MAX_LENGTH = 512
    
    # Training - Optimized settings
    SAMPLES_PER_CLASS = 200_000
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION = 4  # Effective batch = 32
    EPOCHS = 4
    LEARNING_RATE = 1e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # Regularization
    DROPOUT = 0.1
    LABEL_SMOOTHING = 0.1
    
    # Early stopping
    PATIENCE = 2
    
    # Output
    OUTPUT_DIR = "trained_model_enhanced"
    
    # System
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FP16 = torch.cuda.is_available()
    SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def augment_text(text: str) -> str:
    """Simple but effective text augmentation"""
    if random.random() > 0.3:  # 30% chance to augment
        return text
    
    words = text.split()
    if len(words) < 5:
        return text
    
    augmented = words.copy()
    
    # Random deletion (10% of words)
    if random.random() < 0.5:
        num_delete = max(1, int(len(augmented) * 0.1))
        for _ in range(num_delete):
            if len(augmented) > 5:
                idx = random.randint(0, len(augmented) - 1)
                augmented.pop(idx)
    
    # Random swap (2 words)
    if random.random() < 0.5 and len(augmented) > 2:
        i, j = random.sample(range(len(augmented)), 2)
        augmented[i], augmented[j] = augmented[j], augmented[i]
    
    return ' '.join(augmented)


# ============================================================================
# DATASET
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        if self.augment:
            text = augment_text(text)
        
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
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ============================================================================
# LABEL SMOOTHING LOSS
# ============================================================================

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes=2, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.classes = classes
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.classes - 1)
        
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_label = one_hot * confidence + (1 - one_hot) * smooth_value
        
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(smooth_label * log_prob).sum(dim=1).mean()
        
        return loss


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(cache_dir: str, samples_per_class: int):
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")
    
    with open(os.path.join(cache_dir, "human_texts.pkl"), "rb") as f:
        human_texts = pickle.load(f)
    with open(os.path.join(cache_dir, "ai_texts.pkl"), "rb") as f:
        ai_texts = pickle.load(f)
    
    print(f"  Available: {len(human_texts):,} human, {len(ai_texts):,} AI")
    
    # Sample
    random.seed(Config.SEED)
    if len(human_texts) > samples_per_class:
        human_texts = random.sample(human_texts, samples_per_class)
    if len(ai_texts) > samples_per_class:
        ai_texts = random.sample(ai_texts, samples_per_class)
    
    # Balance
    min_size = min(len(human_texts), len(ai_texts))
    human_texts = human_texts[:min_size]
    ai_texts = ai_texts[:min_size]
    
    # Combine
    texts = human_texts + ai_texts
    labels = [0] * len(human_texts) + [1] * len(ai_texts)
    
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    
    print(f"  {Fore.GREEN}✓ Using {len(texts):,} samples{Style.RESET_ALL}")
    
    return list(texts), list(labels)


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    # Update config
    if args.model:
        Config.MODEL_NAME = args.model
    if args.samples:
        Config.SAMPLES_PER_CLASS = args.samples
    if args.epochs:
        Config.EPOCHS = args.epochs
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.cache_dir:
        Config.CACHE_DIR = args.cache_dir
    if args.output_dir:
        Config.OUTPUT_DIR = args.output_dir
    
    set_seed(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{Fore.CYAN}{'='*60}")
    print("   ENHANCED AI DETECTOR TRAINING")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    print(f"\n[Config]")
    print(f"  Model:         {Config.MODEL_NAME}")
    print(f"  Max length:    {Config.MAX_LENGTH}")
    print(f"  Samples:       {Config.SAMPLES_PER_CLASS:,} per class")
    print(f"  Batch:         {Config.BATCH_SIZE} × {Config.GRADIENT_ACCUMULATION} = {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION}")
    print(f"  Epochs:        {Config.EPOCHS}")
    print(f"  LR:            {Config.LEARNING_RATE}")
    print(f"  Label smooth:  {Config.LABEL_SMOOTHING}")
    print(f"  Device:        {Config.DEVICE}")
    
    if Config.DEVICE == "cuda":
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
        print(f"  FP16:          {Config.FP16}")
    
    # Load data
    texts, labels = load_data(Config.CACHE_DIR, Config.SAMPLES_PER_CLASS)
    
    # Split
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=Config.SEED, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=Config.SEED, stratify=temp_labels
    )
    
    print(f"\n[Splits]")
    print(f"  Train: {len(train_texts):,}")
    print(f"  Val:   {len(val_texts):,}")
    print(f"  Test:  {len(test_texts):,}")
    
    del texts, labels
    gc.collect()
    
    # Load model
    print(f"\n{'='*60}")
    print(f"LOADING MODEL: {Config.MODEL_NAME}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=2,
        hidden_dropout_prob=Config.DROPOUT,
        attention_probs_dropout_prob=Config.DROPOUT,
    )
    model.to(Config.DEVICE)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, Config.MAX_LENGTH, augment=True)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, Config.MAX_LENGTH, augment=False)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, Config.MAX_LENGTH, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE*2, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE*2, num_workers=0)
    
    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_params = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': Config.WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_params, lr=Config.LEARNING_RATE)
    
    # Scheduler
    total_steps = (len(train_loader) // Config.GRADIENT_ACCUMULATION) * Config.EPOCHS
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Loss with label smoothing
    criterion = LabelSmoothingLoss(classes=2, smoothing=Config.LABEL_SMOOTHING)
    
    scaler = GradScaler() if Config.FP16 else None
    
    # Training
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")
    
    best_f1 = 0.0
    patience = 0
    start_time = datetime.now()
    
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels_batch = batch['labels'].to(Config.DEVICE)
            
            if Config.FP16:
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels_batch) / Config.GRADIENT_ACCUMULATION
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels_batch) / Config.GRADIENT_ACCUMULATION
                loss.backward()
            
            total_loss += loss.item() * Config.GRADIENT_ACCUMULATION
            
            if (step + 1) % Config.GRADIENT_ACCUMULATION == 0:
                if Config.FP16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
            
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
            patience = 0
            best_path = os.path.join(Config.OUTPUT_DIR, 'best')
            os.makedirs(best_path, exist_ok=True)
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"  {Fore.GREEN}✓ New best! F1={best_f1:.4f}{Style.RESET_ALL}")
        else:
            patience += 1
            if patience >= Config.PATIENCE:
                print(f"\n{Fore.YELLOW}Early stopping!{Style.RESET_ALL}")
                break
    
    train_time = datetime.now() - start_time
    
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
    test_f1 = f1_score(test_true, test_preds)
    cm = confusion_matrix(test_true, test_preds)
    
    print(f"\n{Fore.CYAN}TEST RESULTS:{Style.RESET_ALL}")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  F1 Score: {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"\n{classification_report(test_true, test_preds, target_names=['Human', 'AI'])}")
    
    print(f"\n{Fore.GREEN}{'='*60}")
    print("✓ COMPLETE!")
    print(f"{'='*60}{Style.RESET_ALL}")
    print(f"  Time:     {train_time}")
    print(f"  Best F1:  {best_f1:.4f}")
    print(f"  Test Acc: {test_acc:.2%}")
    
    # Save results
    with open(os.path.join(Config.OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump({
            'model': Config.MODEL_NAME,
            'samples': Config.SAMPLES_PER_CLASS,
            'epochs': Config.EPOCHS,
            'time': str(train_time),
            'best_f1': best_f1,
            'test_acc': test_acc,
            'test_f1': test_f1
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--cache-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    
    train(parser.parse_args())


if __name__ == "__main__":
    main()