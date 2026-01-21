import json
import random
import os
from sklearn.model_selection import train_test_split
from typing import List, Dict

def robotize_text(text: str) -> str:
    """
    Simulates 'AI-like' text by removing contractions, 
    making sentences more rigid, and removing emotion words.
    This is a heuristic approach to generate synthetic 'Input' for training.
    """
    # Simple rule-based transformations
    replacements = {
        "can't": "cannot",
        "won't": "will not",
        "don't": "do not",
        "I'm": "I am",
        "it's": "it is",
        "you're": "you are",
        "they're": "they are",
        "we're": "we are",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        ", ": ". ", # Break long sentences (crudely)
    }
    
    robot_text = text
    for k, v in replacements.items():
        robot_text = robot_text.replace(k, v)
        
    # Occasionally inject "As an AI language model..." or similar robotic headers
    if random.random() < 0.1:
        robot_text = "In summary, " + robot_text
        
    return robot_text

def prepare_synthetic_data(input_file: str, output_dir: str):
    """
    Reads human text, creates synthetic AI pairs, and splits into train/val.
    """
    print(f"Reading human text from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except FileNotFoundError:
        print("Input file not found.")
        return

    # Split into rough 'paragraphs' or chunks
    paragraphs = [p.strip() for p in raw_text.split('\n\n') if len(p.split()) > 10]
    
    data_pairs = []
    print(f"Generating synthetic pairs for {len(paragraphs)} paragraphs...")
    
    for p in paragraphs:
        # HUMAN is the Target, ROBOTIZED (Synthetic AI) is the Input
        ai_version = robotize_text(p)
        data_pairs.append({
            "ai_text": ai_version,
            "human_text": p
        })

    # Train/Val Split
    train_data, val_data = train_test_split(data_pairs, test_size=0.1, random_state=42)
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)
        
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2)
        
    print(f"Saved {len(train_data)} training pairs to {train_path}")
    print(f"Saved {len(val_data)} validation pairs to {val_path}")

if __name__ == "__main__":
    # Example usage:
    # prepare_synthetic_data("d:/SemProject/ManavaAI/gutenberg_dataset.txt", "d:/SemProject/ManavaAI/backend/data/processed")
    pass
