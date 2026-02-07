"""
prepare_data_v3.py
Data Prep tuned to specific Detector Features.
"""
import os
import sys
import json
import random
import re
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

class TunedDataPreparer:
    def __init__(self):
        self.config = Config()
        self.output_dir = os.path.join(self.config.DATA_DIR, "humanizer")
        os.makedirs(self.output_dir, exist_ok=True)

        # EXACT phrases from your extract_features function
        self.ai_phrases = [
            "it is important", "furthermore", "moreover", "consequently",
            "in conclusion", "therefore", "additionally", "thus",
            "it is essential", "studies have shown", "research suggests"
        ]

        self.contractions = {
            "can't": "cannot", "won't": "will not", "don't": "do not", 
            "isn't": "is not", "aren't": "are not", "it's": "it is", 
            "I'm": "I am", "you're": "you are"
        }

    def make_robotic(self, text):
        """
        Transforms human text to trigger the specific detector features.
        """
        result = text
        
        # 1. Kill contractions (Trigger Feature #3)
        for short, full in self.contractions.items():
            result = re.sub(r'\b' + re.escape(short) + r'\b', full, result, flags=re.IGNORECASE)

        # 2. Inject AI phrases (Trigger Feature #1)
        sentences = result.split('. ')
        new_sents = []
        for i, s in enumerate(sentences):
            if i == 0 and random.random() < 0.5:
                # Add phrase at start
                phrase = random.choice(self.ai_phrases)
                # Fix capitalization
                s = phrase + " that " + s[0].lower() + s[1:] if s else s
            new_sents.append(s)
        
        result = '. '.join(new_sents)

        # 3. Kill First Person (Trigger Feature #9)
        result = re.sub(r'\bI think\b', 'it is believed', result, flags=re.IGNORECASE)
        result = re.sub(r'\bI feel\b', 'it appears', result, flags=re.IGNORECASE)

        # 4. Flatten punctuation (Trigger Feature #7, #8)
        result = result.replace('!', '.').replace('?', '.')

        return result

    def prepare(self):
        print("Creating dataset tuned to detector features...")
        human_file = os.path.join(self.config.RAW_DATA_DIR, "human_texts_enhanced.jsonl")
        
        if not os.path.exists(human_file):
            print("Run data_collection.py first.")
            return

        pairs = []
        with open(human_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    data = json.loads(line)
                    target = data['text'] # The real human text
                    
                    # Generate the robotic source
                    source = self.make_robotic(target)
                    
                    if source != target:
                        pairs.append({"source": source, "target": target})
                except: continue

        # Save
        random.shuffle(pairs)
        train_len = int(len(pairs) * 0.9)
        
        self._save(pairs[:train_len], "train_pairs.jsonl")
        self._save(pairs[train_len:], "val_pairs.jsonl")
        print(f"Created {len(pairs)} pairs.")

    def _save(self, data, name):
        with open(os.path.join(self.output_dir, name), 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    p = TunedDataPreparer()
    p.prepare()