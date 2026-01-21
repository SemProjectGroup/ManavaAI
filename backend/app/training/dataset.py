import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any

class HumanizerDataset(Dataset):
    """
    Dataset for loading aligned (AI Text, Human Text) pairs.
    """
    def __init__(self, data_path: str, tokenizer: Any, max_source_length: int = 512, max_target_length: int = 512):
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def _load_data(self, path: str) -> List[Dict]:
        """Loads JSON data. Expects a list of dicts with 'ai_text' and 'human_text' keys."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"Warning: Data file not found at {path}. Returning empty list.")
            return []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        ai_text = item.get("ai_text", "")
        human_text = item.get("human_text", "")

        # Tokenize Inputs
        model_inputs = self.tokenizer(
            ai_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize Targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                human_text,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        return {
            "input_ids": model_inputs.input_ids.squeeze(),
            "attention_mask": model_inputs.attention_mask.squeeze(),
            "labels": labels.input_ids.squeeze()
        }
