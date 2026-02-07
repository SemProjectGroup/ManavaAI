"""
prepare_data_v4.py
The "Hardcore" Data Preparer.
Aggressively transforms human text into 'Robotic Sludge' to force the model
to learn deep structural rewriting.
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

class HardcoreDataPreparer:
    def __init__(self):
        self.config = Config()
        self.output_dir = os.path.join(self.config.DATA_DIR, "humanizer")
        os.makedirs(self.output_dir, exist_ok=True)

        # Complex words AI loves to use
        self.complex_map = {
            "use": "utilize", "help": "facilitate", "show": "demonstrate",
            "make": "generate", "buy": "purchase", "eat": "consume",
            "big": "substantial", "small": "minimal", "bad": "detrimental",
            "good": "beneficial", "think": "hypothesize", "guess": "estimate",
            "know": "acknowledge", "fix": "rectify", "tell": "articulate",
            "try": "attempt", "start": "commence", "end": "conclude"
        }

        # The "AI Glue" that reduces perplexity
        self.connectors = [
            ", and consequently ", ", furthermore ", ", moreover ", 
            ", therefore ", ", thus ", ", which implies that ", 
            "; however, ", "; nevertheless, "
        ]

        self.intros = [
            "It is important to recognize that ", "In the context of modern society, ",
            "Generally speaking, ", "Research has consistently shown that ",
            "From a analytical perspective, ", "It is worth noting that "
        ]

    def destroy_structure(self, text):
        """
        Takes punchy human text and turns it into a long, boring run-on sentence.
        """
        # 1. Expand Contractions (The basics)
        text = text.replace("n't", " not").replace("'re", " are").replace("'s", " is")
        text = text.replace("'m", " am").replace("'ll", " will").replace("'ve", " have")

        # 2. Swap simple words for complex ones
        words = text.split()
        new_words = []
        for w in words:
            clean_w = w.lower().strip(".,!?")
            if clean_w in self.complex_map and random.random() < 0.6:
                # Keep capitalization if present
                replacement = self.complex_map[clean_w]
                if w[0].isupper(): replacement = replacement.capitalize()
                new_words.append(replacement)
            else:
                new_words.append(w)
        text = " ".join(new_words)

        # 3. DESTROY PUNCTUATION (The Key Step)
        # AI text has low sentence variance. We mimic this by combining short sentences.
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 1:
            combined = []
            buffer = ""
            for i, sent in enumerate(sentences):
                # Remove trailing punctuation from the previous part
                sent = sent.strip()
                if not sent: continue
                
                # Combine every 2-3 sentences into one giant block
                if i % 2 != 0: 
                    # Use a connector instead of a period
                    connector = random.choice(self.connectors)
                    # Lowercase the start of the next sentence
                    sent_lower = sent[0].lower() + sent[1:] if sent else ""
                    # Remove the dot from the buffer
                    if buffer.endswith('.'): buffer = buffer[:-1]
                    buffer += connector + sent_lower
                else:
                    if buffer: combined.append(buffer)
                    buffer = sent
            
            if buffer: combined.append(buffer)
            text = " ".join(combined)

        # 4. Add boring intro
        if random.random() < 0.7:
            text = random.choice(self.intros) + text[0].lower() + text[1:]

        return text

    def prepare(self):
        print("="*60)
        print("PREPARING HARDCORE TRAINING DATA")
        print("="*60)
        
        human_file = os.path.join(self.config.RAW_DATA_DIR, "human_texts_enhanced.jsonl")
        
        if not os.path.exists(human_file):
            print(f"❌ Missing {human_file}. Run data collection first.")
            return

        pairs = []
        print("Processing...")
        
        with open(human_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    data = json.loads(line)
                    target = data.get('text', '').strip()
                    
                    # Skip too short or too long
                    if len(target.split()) < 20 or len(target.split()) > 300: 
                        continue

                    # Create the Robotic Input
                    source = self.destroy_structure(target)
                    
                    # Only keep if they are significantly different
                    if source != target:
                        pairs.append({
                            "source": source, # The "Bad" AI text
                            "target": target  # The "Good" Human text
                        })
                except: continue

        print(f"Generated {len(pairs)} pairs.")
        
        # Save
        random.shuffle(pairs)
        split = int(len(pairs) * 0.9)
        
        self._save(pairs[:split], "train_pairs.jsonl")
        self._save(pairs[split:], "val_pairs.jsonl")

    def _save(self, data, name):
        with open(os.path.join(self.output_dir, name), 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved {name}")

if __name__ == "__main__":
    p = HardcoreDataPreparer()
    p.prepare()