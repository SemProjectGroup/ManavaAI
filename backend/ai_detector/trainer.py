# trainer.py
"""
Model Training Logic
"""

import os
import json
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from transformers import get_linear_schedule_with_warmup

from config import Config
from model import AIDetector


class Trainer:
    """Model Trainer"""
    
    def __init__(
        self,
        detector: AIDetector,
        train_loader,
        val_loader,
        config: Config
    ):
        self.detector = detector
        self.model = detector.get_model()
        self.config = config
        self.device = config.training.device
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * config.training.num_epochs
        warmup_steps = int(total_steps * config.training.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Mixed precision
        self.use_fp16 = config.training.fp16 and self.device == 'cuda'
        self.scaler = GradScaler() if self.use_fp16 else None
        
        # Tracking
        self.best_val_f1 = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.training.num_epochs}",
            leave=True
        )
        
        for batch in progress:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_fp16:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            current_lr = self.scheduler.get_last_lr()[0]
            progress.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Periodic evaluation
            if self.global_step % self.config.training.eval_steps == 0:
                val_metrics = self.evaluate()
                self.val_history.append({
                    'step': self.global_step,
                    **val_metrics
                })
                
                # Save best model
                if val_metrics['f1'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['f1']
                    self.patience_counter = 0
                    self._save_checkpoint('best')
                else:
                    self.patience_counter += 1
                
                self.model.train()
        
        epoch_loss = total_loss / num_batches
        self.train_history.append({
            'epoch': epoch + 1,
            'loss': epoch_loss
        })
        
        return epoch_loss
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
                
                total_loss += loss.item()
                num_batches += 1
                
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / num_batches,
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0)
        }
        
        print(f"\n  Validation - Loss: {metrics['loss']:.4f}, "
              f"Acc: {metrics['accuracy']:.4f}, "
              f"F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint"""
        path = os.path.join(self.config.training.output_dir, name)
        self.detector.save(path)
    
    def train(self) -> Dict[str, float]:
        """Full training loop"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"Total epochs: {self.config.training.num_epochs}")
        print(f"Steps per epoch: {len(self.train_loader)}")
        print(f"Total steps: {len(self.train_loader) * self.config.training.num_epochs}")
        print("="*60 + "\n")
        
        for epoch in range(self.config.training.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch + 1} - Train Loss: {train_loss:.4f}")
            
            # Evaluate
            val_metrics = self.evaluate()
            
            # Save best
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.patience_counter = 0
                self._save_checkpoint('best')
                print(f"  ✓ New best model! F1: {self.best_val_f1:.4f}")
            else:
                self.patience_counter += 1
                print(f"  No improvement. Patience: {self.patience_counter}/{self.config.training.early_stopping_patience}")
            
            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        self._save_checkpoint('final')
        
        # Save training history
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'best_f1': self.best_val_f1
        }
        
        history_path = os.path.join(self.config.training.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        print(f"Models saved to: {self.config.training.output_dir}")
        print("="*60)
        
        return {'best_f1': self.best_val_f1}


def evaluate_on_test(
    detector: AIDetector,
    test_texts: list,
    test_labels: list
) -> Dict[str, float]:
    """Final evaluation on test set"""
    
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    # Get predictions
    print(f"Running predictions on {len(test_texts):,} samples...")
    results = detector.predict_batch(test_texts, batch_size=64)
    
    predictions = [1 if r['is_ai'] else 0 for r in results]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(test_labels, predictions),
        'precision': precision_score(test_labels, predictions),
        'recall': recall_score(test_labels, predictions),
        'f1': f1_score(test_labels, predictions)
    }
    
    # Print results
    print("\n" + "-"*40)
    print("TEST RESULTS")
    print("-"*40)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1 Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    print("-"*40)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Human    AI")
    print(f"Actual Human  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       AI     {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=['Human', 'AI']))
    
    return metrics