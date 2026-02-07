"""
train_humanizer_v3.py
Optimized training settings for rewriting structure, not just words.
"""

import os
import sys
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    get_cosine_schedule_with_warmup
)
from tqdm import tqdm
import logging
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewriteDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Varied instructions prevent the model from ignoring the prompt
        self.prompts = [
            "Humanize this text: ",
            "Rewrite to be natural and punchy: ",
            "Remove the AI style from this: ",
            "Make this sound like a real person: ",
            "Paraphrase casually: "
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        prompt = random.choice(self.prompts)
        
        input_text = prompt + pair['source']
        target_text = pair['target']

        source_enc = self.tokenizer(
            input_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt"
        )
        target_enc = self.tokenizer(
            target_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt"
        )

        labels = target_enc.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_enc.input_ids.squeeze(),
            "attention_mask": source_enc.attention_mask.squeeze(),
            "labels": labels
        }

class HardcoreTrainer:
    def __init__(self):
        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use Flan-T5-Base (Best instruction follower for this size)
        self.model_name = "google/flan-t5-base"
        logger.info(f"Loading {self.model_name}...")
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)

        # BFloat16 for stability
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def load_data(self):
        data_dir = os.path.join(self.config.DATA_DIR, "humanizer")
        
        def load(name):
            path = os.path.join(data_dir, name)
            if not os.path.exists(path): return []
            with open(path, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f]

        train_data = load("train_pairs.jsonl")
        val_data = load("val_pairs.jsonl")
        if not train_data: raise ValueError("Run prepare_data_v4.py first!")
        return train_data, val_data

    def train(self, epochs=10, batch_size=4, lr=1e-4):
        # NOTE: Lower LR (1e-4) + More Epochs (10) = Better generalization
        
        train_data, val_data = self.load_data()
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=batch_size*2, shuffle=False, num_workers=0)

        # Dataset wrapper
        train_dataset = RewriteDataset(train_data, self.tokenizer)
        val_dataset = RewriteDataset(val_data, self.tokenizer)
        
        # Update loaders to use dataset wrapper
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps
        )
        
        scaler = torch.cuda.amp.GradScaler() if self.amp_dtype == torch.float16 else None
        accum_steps = 16 // batch_size
        
        best_loss = float('inf')

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()
            
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for i, batch in enumerate(progress):
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                    outputs = self.model(input_ids=input_ids, attention_mask=mask, labels=labels)
                    loss = outputs.loss / accum_steps

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (i + 1) % accum_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * accum_steps
                progress.set_postfix({'loss': f"{total_loss / (i+1):.4f}"})

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                        outputs = self.model(input_ids=input_ids, attention_mask=mask, labels=labels)
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                save_path = os.path.join(self.config.MODEL_SAVE_DIR, "humanizer_v2")
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                logger.info(f"🔥 New Best Model Saved (Loss: {best_loss:.4f})")

if __name__ == "__main__":
    trainer = HardcoreTrainer()
    trainer.train()