"""
LoRA Training Pipeline for AI Text Humanizer
Trains a language model with LoRA adapters to humanize AI-generated text
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline"""
    
    # Model settings
    model_name: str = "mistralai/Mistral-7B-v0.1"  # or "gpt2-medium"
    max_length: int = 512
    
    # LoRA settings
    lora_r: int = 16  # Rank
    lora_alpha: int = 32  # Alpha scaling
    lora_dropout: float = 0.05
    target_modules: List[str] = None  # Will be set based on model
    
    # Training settings
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Validation settings
    eval_steps: int = 100  # Evaluate every N steps
    save_steps: int = 500  # Save checkpoint every N steps
    logging_steps: int = 10
    
    # Paths
    train_data_path: str = "data/splits/train.json"
    val_data_path: str = "data/splits/validation.json"
    output_dir: str = "models/humanizer"
    checkpoint_dir: str = "models/checkpoints"
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # Use fp16
    
    # Other
    seed: int = 42
    
    def __post_init__(self):
        """Set target modules based on model"""
        if self.target_modules is None:
            if "mistral" in self.model_name.lower() or "llama" in self.model_name.lower():
                self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif "gpt2" in self.model_name.lower():
                self.target_modules = ["c_attn", "c_proj"]
            else:
                self.target_modules = ["q_proj", "v_proj"]  # Default


class HumanizerDataset(Dataset):
    """Dataset for AI text humanization"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        human_only: bool = True
    ):
        """
        Initialize dataset
        
        Args:
            data_path: Path to JSON data file
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            human_only: If True, only use human-written texts for training
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter for human texts if specified
        if human_only:
            self.texts = [item['text'] for item in data if item['label'] == 'human']
            logger.info(f"Loaded {len(self.texts)} human-written texts")
        else:
            self.texts = [item['text'] for item in data]
            logger.info(f"Loaded {len(self.texts)} total texts")
        
        if len(self.texts) == 0:
            raise ValueError("No texts found in dataset!")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized item"""
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # For causal LM, labels are the same as input_ids
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # For causal LM loss
        }


class LoRATrainer:
    """Training pipeline for LoRA fine-tuning"""
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration"""
        self.config = config
        
        # Set seed for reproducibility
        self._set_seed(config.seed)
        
        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def load_model(self):
        """Load base model and apply LoRA"""
        logger.info(f"Loading base model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.mixed_precision else torch.float32,
            device_map="auto" if self.config.device == "cuda" else None,
            trust_remote_code=True
        )
        
        # Configure LoRA
        logger.info("Configuring LoRA adapters...")
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("Model loaded successfully with LoRA adapters!")
    
    def prepare_data(self):
        """Prepare training and validation data loaders"""
        logger.info("Preparing data loaders...")
        
        # Create datasets
        train_dataset = HumanizerDataset(
            self.config.train_data_path,
            self.tokenizer,
            self.config.max_length,
            human_only=True
        )
        
        val_dataset = HumanizerDataset(
            self.config.val_data_path,
            self.tokenizer,
            self.config.max_length,
            human_only=True
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.config.device == "cuda" else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.config.device == "cuda" else False
        )
        
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        logger.info("Setting up optimizer and scheduler...")
        
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Calculate total training steps
        total_steps = len(self.train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        
        # Create scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Total training steps: {total_steps}")
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute training loss"""
        # Move batch to device
        input_ids = batch['input_ids'].to(self.config.device)
        attention_mask = batch['attention_mask'].to(self.config.device)
        labels = batch['labels'].to(self.config.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs.loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and return average loss"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        logger.info("Running validation...")
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            loss = self.compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.model.train()
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, step: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint-epoch{epoch}-step{step}"
        )
        
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        
        # Save LoRA weights
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save training state
        state = {
            'epoch': epoch,
            'global_step': step,
            'val_loss': val_loss,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'training_history': self.training_history
        }
        
        torch.save(state, os.path.join(checkpoint_path, 'training_state.pt'))
        
        logger.info("Checkpoint saved!")
    
    def save_final_model(self):
        """Save final trained model"""
        logger.info(f"Saving final model to {self.config.output_dir}")
        
        # Save LoRA adapters
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save training history
        history_path = os.path.join(self.config.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save config
        config_path = os.path.join(self.config.output_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info("Final model saved!")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Training for {self.config.num_epochs} epochs")
        logger.info(f"Device: {self.config.device}")
        
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*60}")
            
            epoch_loss = 0
            num_batches = 0
            self.optimizer.zero_grad()
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Compute loss
                loss = self.compute_loss(batch)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                num_batches += 1
                
                # Update weights every N steps
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        current_lr = self.scheduler.get_last_lr()[0]
                        self.training_history['train_loss'].append(loss.item() * self.config.gradient_accumulation_steps)
                        self.training_history['learning_rate'].append(current_lr)
                        
                        progress_bar.set_postfix({
                            'loss': f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                            'lr': f"{current_lr:.2e}"
                        })
                    
                    # Validation
                    if self.global_step % self.config.eval_steps == 0:
                        val_loss = self.validate()
                        self.training_history['val_loss'].append(val_loss)
                        
                        logger.info(f"Step {self.global_step} - Validation Loss: {val_loss:.4f}")
                        
                        # Save best model
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            logger.info(f"New best validation loss: {val_loss:.4f}")
                            self.save_checkpoint(epoch, self.global_step, val_loss)
                    
                    # Save checkpoint
                    elif self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(epoch, self.global_step, self.best_val_loss)
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
            
            # Validation at end of epoch
            val_loss = self.validate()
            logger.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")
            
            # Save checkpoint at end of epoch
            self.save_checkpoint(epoch + 1, self.global_step, val_loss)
        
        # Save final model
        self.save_final_model()
        
        logger.info("\n" + "="*60)
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Final model saved to: {self.config.output_dir}")
        logger.info("="*60)
    
    def run(self):
        """Execute complete training pipeline"""
        try:
            logger.info("Initializing training pipeline...")
            
            # Load model
            self.load_model()
            
            # Prepare data
            self.prepare_data()
            
            # Setup optimizer
            self.setup_optimizer()
            
            # Train
            self.train()
            
            logger.info("Training pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise


def main():
    """Main execution function"""
    
    # Create configuration
    config = TrainingConfig(
        # Model settings
        model_name="gpt2-medium",  # Use gpt2-medium for faster training
        max_length=512,
        
        # LoRA settings
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        
        # Training settings
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4,
        gradient_accumulation_steps=4,
        
        # Validation settings
        eval_steps=100,
        save_steps=500,
        logging_steps=10,
        
        # Paths
        train_data_path="data/splits/train.json",
        val_data_path="data/splits/validation.json",
        output_dir="models/humanizer",
        checkpoint_dir="models/checkpoints",
        
        # Hardware
        device="cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision=True,
        
        # Seed
        seed=42
    )
    
    # Create trainer
    trainer = LoRATrainer(config)
    
    # Run training
    trainer.run()


if __name__ == "__main__":
    main()