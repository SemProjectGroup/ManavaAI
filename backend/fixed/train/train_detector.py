"""
Training script for the AI Content Detector model.
Fine-tunes DeBERTa/RoBERTa for binary classification (AI vs Human text).
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from tqdm import tqdm
from datetime import datetime
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from models.detector import AIDetectorModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DetectorDataset(Dataset):
    """Dataset for AI detection training."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }


class DetectorTrainer:
    """Trainer class for the AI Content Detector."""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.DETECTOR_MODEL_NAME)
        
        # Initialize model
        self.model = AIDetectorModel(
            model_name=self.config.DETECTOR_MODEL_NAME,
            num_labels=1,
            dropout_rate=self.config.DROPOUT_RATE
        ).to(self.device)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Best model tracking
        self.best_val_accuracy = 0.0
        
        # W&B integration (optional)
        self.use_wandb = False
        try:
            import wandb
            self.wandb = wandb
            self.use_wandb = True
        except ImportError:
            logger.info("Weights & Biases not installed. Skipping W&B logging.")
    
    def load_jsonl(self, file_path):
        """Load data from a JSONL file."""
        texts = []
        labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    texts.append(sample['text'])
                    labels.append(sample['label'])
        
        return texts, labels
    
    def load_data(self, data_dir=None):
        """Load training, validation, and test data."""
        # Try multiple possible data locations
        possible_dirs = [
            data_dir,
            os.path.join(self.config.BASE_DIR, "datasets"),
            os.path.join(self.config.BASE_DIR, "datasets", "processed"),
            self.config.PROCESSED_DATA_DIR,
            os.path.join(self.config.BASE_DIR, "data_files", "processed"),
        ]
        
        data_dir = None
        for d in possible_dirs:
            if d and os.path.exists(d):
                # Check if train.jsonl exists
                if os.path.exists(os.path.join(d, "train.jsonl")):
                    data_dir = d
                    break
                # Check for processed_dataset.jsonl
                if os.path.exists(os.path.join(d, "processed_dataset.jsonl")):
                    data_dir = d
                    break
        
        if data_dir is None:
            raise FileNotFoundError(
                "Could not find dataset. Please run data collection first.\n"
                "Expected locations:\n"
                f"  - {os.path.join(self.config.BASE_DIR, 'datasets', 'train.jsonl')}\n"
                f"  - {self.config.PROCESSED_DATA_DIR}"
            )
        
        logger.info(f"Loading data from: {data_dir}")
        
        # Check which format exists
        train_file = os.path.join(data_dir, "train.jsonl")
        single_file = os.path.join(data_dir, "processed_dataset.jsonl")
        
        if os.path.exists(train_file):
            # Load separate train/val/test files
            logger.info("Loading split dataset (train/val/test)...")
            
            train_texts, train_labels = self.load_jsonl(train_file)
            logger.info(f"Loaded {len(train_texts)} training samples")
            
            val_file = os.path.join(data_dir, "val.jsonl")
            if os.path.exists(val_file):
                val_texts, val_labels = self.load_jsonl(val_file)
                logger.info(f"Loaded {len(val_texts)} validation samples")
            else:
                val_texts, val_labels = [], []
            
            test_file = os.path.join(data_dir, "test.jsonl")
            if os.path.exists(test_file):
                test_texts, test_labels = self.load_jsonl(test_file)
                logger.info(f"Loaded {len(test_texts)} test samples")
            else:
                test_texts, test_labels = [], []
            
            return {
                'train': (train_texts, train_labels),
                'val': (val_texts, val_labels),
                'test': (test_texts, test_labels)
            }
        
        elif os.path.exists(single_file):
            # Load single file and split
            logger.info("Loading single dataset file and splitting...")
            texts, labels = self.load_jsonl(single_file)
            logger.info(f"Loaded {len(texts)} total samples")
            
            return {
                'all': (texts, labels)
            }
        
        else:
            raise FileNotFoundError(f"No dataset files found in {data_dir}")
    
    def prepare_dataloaders(self, data_dict, test_size=0.1, val_size=0.1):
        """Create dataloaders from data dictionary."""
        from sklearn.model_selection import train_test_split
        
        if 'train' in data_dict:
            # Already split
            train_texts, train_labels = data_dict['train']
            val_texts, val_labels = data_dict['val']
            test_texts, test_labels = data_dict['test']
            
            # If val or test are empty, split from train
            if len(val_texts) == 0:
                train_texts, val_texts, train_labels, val_labels = train_test_split(
                    train_texts, train_labels,
                    test_size=val_size,
                    random_state=self.config.RANDOM_SEED,
                    stratify=train_labels
                )
            
            if len(test_texts) == 0:
                train_texts, test_texts, train_labels, test_labels = train_test_split(
                    train_texts, train_labels,
                    test_size=test_size,
                    random_state=self.config.RANDOM_SEED,
                    stratify=train_labels
                )
        else:
            # Need to split
            texts, labels = data_dict['all']
            
            # First split: train+val vs test
            train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
                texts, labels,
                test_size=test_size,
                random_state=self.config.RANDOM_SEED,
                stratify=labels
            )
            
            # Second split: train vs val
            val_ratio = val_size / (1 - test_size)
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_val_texts, train_val_labels,
                test_size=val_ratio,
                random_state=self.config.RANDOM_SEED,
                stratify=train_val_labels
            )
        
        logger.info(f"Dataset sizes - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        # Log label distribution
        train_ai = sum(train_labels)
        train_human = len(train_labels) - train_ai
        logger.info(f"Training set - AI: {train_ai}, Human: {train_human}")
        
        # Create datasets
        train_dataset = DetectorDataset(
            train_texts, train_labels, self.tokenizer, self.config.MAX_LENGTH
        )
        val_dataset = DetectorDataset(
            val_texts, val_labels, self.tokenizer, self.config.MAX_LENGTH
        )
        test_dataset = DetectorDataset(
            test_texts, test_labels, self.tokenizer, self.config.MAX_LENGTH
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def setup_training(self, train_loader):
        """Setup optimizer and scheduler."""
        
        # Optimizer with weight decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.WEIGHT_DECAY
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.LEARNING_RATE,
            eps=1e-8
        )
        
        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * self.config.NUM_EPOCHS
        warmup_steps = int(total_steps * self.config.WARMUP_RATIO)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits.squeeze(), labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.sigmoid(logits.squeeze()).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
        accuracy = accuracy_score(all_labels, binary_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, binary_preds, average='binary', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits.squeeze(), labels)
                
                total_loss += loss.item()
                probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
                
                # Handle single item batch
                if isinstance(probs, np.floating):
                    probs = [float(probs)]
                else:
                    probs = probs.tolist() if hasattr(probs, 'tolist') else [probs]
                
                all_probs.extend(probs)
                all_preds.extend([1 if p > 0.5 else 0 for p in probs])
                all_labels.extend(labels.cpu().numpy().tolist())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        try:
            auc_roc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc_roc = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc
        }
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint."""
        checkpoint_dir = self.config.CHECKPOINT_DIR
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'detector_latest.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'detector_best.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"✓ Saved best model with accuracy: {metrics['accuracy']:.4f}")
        
        # Save epoch checkpoint
        epoch_path = os.path.join(checkpoint_dir, f'detector_epoch_{epoch+1}.pt')
        torch.save(checkpoint, epoch_path)
    
    def save_final_model(self):
        """Save the final trained model for inference."""
        model_dir = self.config.MODEL_SAVE_DIR
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'detector_model.pt')
        torch.save(self.model.state_dict(), model_path)
        
        # Save tokenizer
        tokenizer_dir = os.path.join(model_dir, 'detector_tokenizer')
        self.tokenizer.save_pretrained(tokenizer_dir)
        
        # Save config
        config_dict = {
            'model_name': self.config.DETECTOR_MODEL_NAME,
            'max_length': self.config.MAX_LENGTH,
            'dropout_rate': self.config.DROPOUT_RATE
        }
        with open(os.path.join(model_dir, 'detector_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"✓ Final model saved to {model_dir}")
    
    def train(self, data_path=None, init_wandb=False):
        """Main training loop."""
        
        # Initialize W&B if requested
        if init_wandb and self.use_wandb:
            self.wandb.init(
                project="manavai-detector",
                config={
                    "model": self.config.DETECTOR_MODEL_NAME,
                    "learning_rate": self.config.LEARNING_RATE,
                    "batch_size": self.config.BATCH_SIZE,
                    "epochs": self.config.NUM_EPOCHS
                }
            )
        
        # Load data
        logger.info("Loading training data...")
        data_dict = self.load_data(data_path)
        
        # Prepare dataloaders
        logger.info("Preparing dataloaders...")
        train_loader, val_loader, test_loader = self.prepare_dataloaders(data_dict)
        
        # Setup training
        self.setup_training(train_loader)
        
        # Training loop
        logger.info("="*60)
        logger.info("STARTING TRAINING")
        logger.info("="*60)
        training_history = []
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            logger.info(f"Epoch {epoch+1} - Train Loss: {train_metrics['loss']:.4f}, "
                       f"Accuracy: {train_metrics['accuracy']:.4f}")
            
            # Validate
            val_metrics = self.validate(val_loader)
            logger.info(f"Epoch {epoch+1} - Val Loss: {val_metrics['loss']:.4f}, "
                       f"Accuracy: {val_metrics['accuracy']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}, "
                       f"AUC-ROC: {val_metrics['auc_roc']:.4f}")
            
            # Track metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics
            }
            training_history.append(epoch_metrics)
            
            # Log to W&B
            if self.use_wandb and init_wandb:
                self.wandb.log({
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1'],
                    'val_auc_roc': val_metrics['auc_roc']
                })
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_metrics['accuracy']
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping check
            if val_metrics['accuracy'] >= self.config.DETECTOR_TARGET_ACCURACY:
                logger.info(f"✓ Reached target accuracy of {self.config.DETECTOR_TARGET_ACCURACY}!")
                break
        
        # Final evaluation on test set
        logger.info("="*60)
        logger.info("EVALUATING ON TEST SET")
        logger.info("="*60)
        test_metrics = self.validate(test_loader)
        logger.info(f"Test Results - Accuracy: {test_metrics['accuracy']:.4f}, "
                   f"F1: {test_metrics['f1']:.4f}, "
                   f"AUC-ROC: {test_metrics['auc_roc']:.4f}")
        
        # Save final model
        self.save_final_model()
        
        # Save training history
        history_path = os.path.join(self.config.MODEL_SAVE_DIR, 'detector_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Close W&B
        if self.use_wandb and init_wandb:
            self.wandb.finish()
        
        return training_history, test_metrics


def main():
    """Main function to run detector training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AI Detector Model')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data directory')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    args = parser.parse_args()
    
    # Load config and override if needed
    config = Config()
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    
    # Initialize trainer
    trainer = DetectorTrainer(config)
    
    # Train
    history, test_metrics = trainer.train(
        data_path=args.data_path,
        init_wandb=args.wandb
    )
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"Best Validation Accuracy: {trainer.best_val_accuracy:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()