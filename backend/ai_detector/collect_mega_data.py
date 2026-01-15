"""
================================================================================
MEGA AI DETECTION DATA COLLECTOR
================================================================================
Collects diverse, properly-labeled data for AI detection training.

Target: 500K+ human texts, 500K+ AI texts from multiple sources

Usage:
    python collect_mega_data.py
    python collect_mega_data.py --target 1000000  # 1M per class

================================================================================
"""

import os
import sys
import gc
import json
import pickle
import random
import argparse
import hashlib
from datetime import datetime
from typing import List, Set, Tuple
from collections import defaultdict

# Install required packages
def install(package):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

try:
    from datasets import load_dataset
except ImportError:
    install("datasets")
    from datasets import load_dataset

try:
    from tqdm import tqdm
except ImportError:
    install("tqdm")
    from tqdm import tqdm

try:
    import pandas as pd
except ImportError:
    install("pandas")
    import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    OUTPUT_DIR = r"F:\data\mega_ai_dataset"
    TARGET_PER_CLASS = 500_000  # 500K each
    MIN_WORDS = 15
    MAX_WORDS = 1000
    SEED = 42
    
    # Track sources for analysis
    SOURCES = defaultdict(int)


# ============================================================================
# DEDUPLICATION
# ============================================================================

class Deduplicator:
    """Fast deduplication using hashing"""
    
    def __init__(self):
        self.seen_hashes: Set[str] = set()
    
    def get_hash(self, text: str) -> str:
        """Get hash of normalized text"""
        normalized = ' '.join(text.lower().split())[:500]
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate"""
        h = self.get_hash(text)
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False
    
    def __len__(self):
        return len(self.seen_hashes)


# ============================================================================
# TEXT FILTERING
# ============================================================================

def is_valid_text(text: str, min_words: int = 15, max_words: int = 1000) -> bool:
    """Check if text meets quality criteria"""
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip()
    if len(text) < 50:
        return False
    
    words = text.split()
    word_count = len(words)
    
    if word_count < min_words or word_count > max_words:
        return False
    
    # Check for too much repetition
    unique_ratio = len(set(words)) / word_count
    if unique_ratio < 0.3:
        return False
    
    # Check for too many special characters
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
    if alpha_ratio < 0.7:
        return False
    
    return True


def clean_text(text: str) -> str:
    """Clean text for training"""
    if not text:
        return ""
    
    # Basic cleaning
    text = text.strip()
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove common prefixes from AI responses
    prefixes_to_remove = [
        "Sure, ", "Sure! ", "Certainly! ", "Certainly, ",
        "Of course! ", "Of course, ", "Absolutely! ",
        "I'd be happy to ", "I would be happy to ",
    ]
    
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    
    return text.strip()


# ============================================================================
# DATA SOURCES - HUMAN TEXT
# ============================================================================

def collect_human_casual(dedup: Deduplicator, target: int) -> List[str]:
    """Collect casual human text (Reddit, social media style)"""
    texts = []
    
    print("\n" + "─"*60)
    print("📱 HUMAN - Casual/Informal Text")
    print("─"*60)
    
    # 1. ELI5 Questions (definitely human)
    print("\n  [1] ELI5 Questions...")
    try:
        ds = load_dataset("eli5", split="train_eli5", trust_remote_code=True)
        for item in tqdm(ds, desc="ELI5"):
            if len(texts) >= target // 4:
                break
            text = item.get('title', '')
            if is_valid_text(text, min_words=8) and not dedup.is_duplicate(text):
                texts.append(clean_text(text))
                Config.SOURCES['eli5'] += 1
        print(f"      ✓ Collected {Config.SOURCES['eli5']:,}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # 2. Writing Prompts (human creative writing)
    print("\n  [2] Writing Prompts Stories...")
    try:
        ds = load_dataset("euclaise/writingprompts", split="train", streaming=True)
        count = 0
        for item in tqdm(ds, desc="WritingPrompts", total=target//4):
            if count >= target // 4:
                break
            text = item.get('story', '')
            if is_valid_text(text, max_words=500) and not dedup.is_duplicate(text):
                texts.append(clean_text(text))
                Config.SOURCES['writingprompts'] += 1
                count += 1
        print(f"      ✓ Collected {Config.SOURCES['writingprompts']:,}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # 3. Showerthoughts (casual human thoughts)
    print("\n  [3] Casual thoughts/comments...")
    try:
        ds = load_dataset("reddit", split="train", streaming=True, trust_remote_code=True)
        count = 0
        for item in tqdm(ds, desc="Reddit", total=target//4):
            if count >= target // 4:
                break
            text = item.get('content', '') or item.get('body', '') or item.get('selftext', '')
            if is_valid_text(text, min_words=10, max_words=300) and not dedup.is_duplicate(text):
                texts.append(clean_text(text))
                Config.SOURCES['reddit'] += 1
                count += 1
        print(f"      ✓ Collected {Config.SOURCES['reddit']:,}")
    except Exception as e:
        print(f"      ✗ Skipping Reddit: {e}")
    
    print(f"\n  📊 Total casual human: {len(texts):,}")
    return texts


def collect_human_reviews(dedup: Deduplicator, target: int) -> List[str]:
    """Collect human reviews (opinions, real experiences)"""
    texts = []
    
    print("\n" + "─"*60)
    print("⭐ HUMAN - Reviews & Opinions")
    print("─"*60)
    
    # 1. IMDB Movie Reviews
    print("\n  [1] IMDB Movie Reviews...")
    try:
        ds = load_dataset("imdb", split="train")
        for item in tqdm(ds, desc="IMDB"):
            if len(texts) >= target // 3:
                break
            text = item.get('text', '')
            if is_valid_text(text, max_words=400) and not dedup.is_duplicate(text):
                texts.append(clean_text(text))
                Config.SOURCES['imdb'] += 1
        print(f"      ✓ Collected {Config.SOURCES['imdb']:,}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # 2. Yelp Reviews
    print("\n  [2] Yelp Business Reviews...")
    try:
        ds = load_dataset("yelp_review_full", split="train")
        for item in tqdm(ds, desc="Yelp"):
            if len(texts) >= target * 2 // 3:
                break
            text = item.get('text', '')
            if is_valid_text(text, max_words=300) and not dedup.is_duplicate(text):
                texts.append(clean_text(text))
                Config.SOURCES['yelp'] += 1
        print(f"      ✓ Collected {Config.SOURCES['yelp']:,}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # 3. Amazon Reviews
    print("\n  [3] Amazon Product Reviews...")
    try:
        ds = load_dataset("amazon_polarity", split="train", streaming=True)
        count = 0
        for item in tqdm(ds, desc="Amazon", total=target//3):
            if count >= target // 3:
                break
            text = item.get('content', '')
            if is_valid_text(text, max_words=300) and not dedup.is_duplicate(text):
                texts.append(clean_text(text))
                Config.SOURCES['amazon'] += 1
                count += 1
        print(f"      ✓ Collected {Config.SOURCES['amazon']:,}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    print(f"\n  📊 Total review human: {len(texts):,}")
    return texts


def collect_human_formal(dedup: Deduplicator, target: int) -> List[str]:
    """Collect formal human writing (articles, essays)"""
    texts = []
    
    print("\n" + "─"*60)
    print("📚 HUMAN - Formal/Professional Writing")
    print("─"*60)
    
    # 1. CNN/DailyMail Articles
    print("\n  [1] News Articles...")
    try:
        ds = load_dataset("cnn_dailymail", "3.0.0", split="train", streaming=True)
        count = 0
        for item in tqdm(ds, desc="CNN/DM", total=target//3):
            if count >= target // 3:
                break
            text = item.get('article', '')
            if is_valid_text(text, max_words=500) and not dedup.is_duplicate(text):
                texts.append(clean_text(text))
                Config.SOURCES['cnn_dm'] += 1
                count += 1
        print(f"      ✓ Collected {Config.SOURCES['cnn_dm']:,}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # 2. Scientific abstracts
    print("\n  [2] Scientific Abstracts...")
    try:
        ds = load_dataset("ccdv/pubmed-summarization", split="train", streaming=True)
        count = 0
        for item in tqdm(ds, desc="PubMed", total=target//3):
            if count >= target // 3:
                break
            text = item.get('abstract', '')
            if is_valid_text(text) and not dedup.is_duplicate(text):
                texts.append(clean_text(text))
                Config.SOURCES['pubmed'] += 1
                count += 1
        print(f"      ✓ Collected {Config.SOURCES['pubmed']:,}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # 3. Wikipedia (first paragraphs)
    print("\n  [3] Encyclopedia entries...")
    try:
        ds = load_dataset("wikipedia", "20220301.simple", split="train", streaming=True)
        count = 0
        for item in tqdm(ds, desc="Wikipedia", total=target//3):
            if count >= target // 3:
                break
            text = item.get('text', '')
            # Get first paragraph only
            paragraphs = text.split('\n\n')
            if paragraphs:
                text = paragraphs[0]
            if is_valid_text(text, max_words=300) and not dedup.is_duplicate(text):
                texts.append(clean_text(text))
                Config.SOURCES['wikipedia'] += 1
                count += 1
        print(f"      ✓ Collected {Config.SOURCES['wikipedia']:,}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    print(f"\n  📊 Total formal human: {len(texts):,}")
    return texts


# ============================================================================
# DATA SOURCES - AI TEXT
# ============================================================================

def collect_ai_chatgpt(dedup: Deduplicator, target: int) -> List[str]:
    """Collect ChatGPT responses"""
    texts = []
    
    print("\n" + "─"*60)
    print("🤖 AI - ChatGPT Responses")
    print("─"*60)
    
    # 1. HC3 Dataset (ChatGPT answers)
    print("\n  [1] HC3 ChatGPT Answers...")
    try:
        configs = ['all', 'finance', 'medicine', 'open_qa', 'wiki_csai', 'reddit_eli5']
        for config in configs:
            try:
                ds = load_dataset("Hello-SimpleAI/HC3", config, split="train", trust_remote_code=True)
                for item in tqdm(ds, desc=f"HC3-{config}"):
                    for ans in item.get('chatgpt_answers', []):
                        if is_valid_text(ans) and not dedup.is_duplicate(ans):
                            texts.append(clean_text(ans))
                            Config.SOURCES['hc3_chatgpt'] += 1
            except:
                continue
        print(f"      ✓ Collected {Config.SOURCES['hc3_chatgpt']:,}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # 2. ShareGPT (ChatGPT conversations)
    print("\n  [2] ShareGPT Conversations...")
    try:
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", trust_remote_code=True)
        for item in tqdm(ds, desc="ShareGPT"):
            convos = item.get('conversations', [])
            for turn in convos:
                if turn.get('from') in ['gpt', 'assistant', 'chatgpt']:
                    text = turn.get('value', '')
                    if is_valid_text(text) and not dedup.is_duplicate(text):
                        texts.append(clean_text(text))
                        Config.SOURCES['sharegpt'] += 1
        print(f"      ✓ Collected {Config.SOURCES['sharegpt']:,}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # 3. GPT-Wiki-Intro (GPT-generated)
    print("\n  [3] GPT-generated Wikipedia intros...")
    try:
        ds = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
        for item in tqdm(ds, desc="GPT-Wiki"):
            text = item.get('generated_intro', '')
            if is_valid_text(text) and not dedup.is_duplicate(text):
                texts.append(clean_text(text))
                Config.SOURCES['gpt_wiki'] += 1
        print(f"      ✓ Collected {Config.SOURCES['gpt_wiki']:,}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    print(f"\n  📊 Total ChatGPT: {len(texts):,}")
    return texts


def collect_ai_instruct(dedup: Deduplicator, target: int) -> List[str]:
    """Collect instruction-tuned model outputs (Alpaca, Dolly, etc.)"""
    texts = []
    
    print("\n" + "─"*60)
    print("🤖 AI - Instruction Model Outputs")
    print("─"*60)
    
    # 1. Alpaca (GPT-3.5 generated)
    print("\n  [1] Alpaca (GPT-3.5 generated)...")
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        for item in tqdm(ds, desc="Alpaca"):
            text = item.get('output', '')
            if is_valid_text(text) and not dedup.is_duplicate(text):
                texts.append(clean_text(text))
                Config.SOURCES['alpaca'] += 1
        print(f"      ✓ Collected {Config.SOURCES['alpaca']:,}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # 2. Dolly (instruction outputs)
    print("\n  [2] Dolly Responses...")
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        for item in tqdm(ds, desc="Dolly"):
            text = item.get('response', '')
            if is_valid_text(text) and not dedup.is_duplicate(text):
                texts.append(clean_text(text))
                Config.SOURCES['dolly'] += 1
        print(f"      ✓ Collected {Config.SOURCES['dolly']:,}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # 3. OpenAssistant (AI responses)
    print("\n  [3] OpenAssistant AI Responses...")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
        for item in tqdm(ds, desc="OASST"):
            if item.get('role') == 'assistant':
                text = item.get('text', '')
                if is_valid_text(text) and not dedup.is_duplicate(text):
                    texts.append(clean_text(text))
                    Config.SOURCES['oasst'] += 1
        print(f"      ✓ Collected {Config.SOURCES['oasst']:,}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # 4. WizardLM (GPT-4 generated)
    print("\n  [4] WizardLM Responses...")
    try:
        ds = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train")
        for item in tqdm(ds, desc="WizardLM"):
            convos = item.get('conversations', [])
            for turn in convos:
                if turn.get('from') == 'gpt':
                    text = turn.get('value', '')
                    if is_valid_text(text) and not dedup.is_duplicate(text):
                        texts.append(clean_text(text))
                        Config.SOURCES['wizardlm'] += 1
        print(f"      ✓ Collected {Config.SOURCES['wizardlm']:,}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # 5. FLAN (instruction outputs)
    print("\n  [5] FLAN Model Outputs...")
    try:
        ds = load_dataset("Muennighoff/flan", split="train", streaming=True)
        count = 0
        for item in tqdm(ds, desc="FLAN", total=100000):
            if count >= 100000:
                break
            text = item.get('target', '') or item.get('targets', '')
            if is_valid_text(text) and not dedup.is_duplicate(text):
                texts.append(clean_text(text))
                Config.SOURCES['flan'] += 1
                count += 1
        print(f"      ✓ Collected {Config.SOURCES['flan']:,}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    print(f"\n  📊 Total instruction AI: {len(texts):,}")
    return texts


def collect_ai_detection_datasets(dedup: Deduplicator, target: int) -> Tuple[List[str], List[str]]:
    """Collect from existing AI detection datasets (already labeled)"""
    human_texts = []
    ai_texts = []
    
    print("\n" + "─"*60)
    print("📊 Pre-labeled AI Detection Datasets")
    print("─"*60)
    
    datasets_to_try = [
        ("Hello-SimpleAI/chatgpt-detector-roberta", None),
        ("artem9k/ai-text-detection-pile", None),
        ("yaful/DeepfakeTextDetection", None),
    ]
    
    for ds_name, config in datasets_to_try:
        print(f"\n  Loading {ds_name}...")
        try:
            if config:
                ds = load_dataset(ds_name, config, split="train", trust_remote_code=True)
            else:
                ds = load_dataset(ds_name, split="train", trust_remote_code=True)
            
            for item in tqdm(ds, desc=ds_name.split("/")[-1][:25]):
                text = item.get('text', '')
                label = item.get('label', -1)
                
                if not is_valid_text(text):
                    continue
                
                text = clean_text(text)
                
                if dedup.is_duplicate(text):
                    continue
                
                if label == 0:  # Human
                    human_texts.append(text)
                    Config.SOURCES[f'{ds_name.split("/")[-1]}_human'] += 1
                elif label == 1:  # AI
                    ai_texts.append(text)
                    Config.SOURCES[f'{ds_name.split("/")[-1]}_ai'] += 1
            
            print(f"      ✓ Human: {len(human_texts):,}, AI: {len(ai_texts):,}")
            
        except Exception as e:
            print(f"      ✗ Error: {e}")
    
    return human_texts, ai_texts


def collect_hc3_human(dedup: Deduplicator, target: int) -> List[str]:
    """Collect human answers from HC3 (paired with ChatGPT)"""
    texts = []
    
    print("\n" + "─"*60)
    print("👤 HC3 Human Answers (same questions as ChatGPT)")
    print("─"*60)
    
    try:
        configs = ['all', 'finance', 'medicine', 'open_qa', 'wiki_csai', 'reddit_eli5']
        for config in configs:
            try:
                ds = load_dataset("Hello-SimpleAI/HC3", config, split="train", trust_remote_code=True)
                for item in tqdm(ds, desc=f"HC3-{config}"):
                    for ans in item.get('human_answers', []):
                        if is_valid_text(ans) and not dedup.is_duplicate(ans):
                            texts.append(clean_text(ans))
                            Config.SOURCES['hc3_human'] += 1
            except:
                continue
        print(f"  ✓ Collected {Config.SOURCES['hc3_human']:,}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    return texts


# ============================================================================
# MAIN COLLECTION
# ============================================================================

def collect_all(target_per_class: int) -> Tuple[List[str], List[str]]:
    """Collect all data"""
    
    print("\n" + "="*70)
    print("   MEGA DATA COLLECTION FOR AI DETECTION")
    print("="*70)
    print(f"\n  Target: {target_per_class:,} texts per class")
    print(f"  Output: {Config.OUTPUT_DIR}")
    
    # Initialize deduplicators (separate for human and AI)
    human_dedup = Deduplicator()
    ai_dedup = Deduplicator()
    
    human_texts = []
    ai_texts = []
    
    # Collect from pre-labeled datasets first (best quality)
    det_human, det_ai = collect_ai_detection_datasets(human_dedup, target_per_class)
    human_texts.extend(det_human)
    ai_texts.extend(det_ai)
    
    # Collect HC3 (human answers vs ChatGPT answers)
    hc3_human = collect_hc3_human(human_dedup, target_per_class)
    human_texts.extend(hc3_human)
    
    # Collect more human text (various types)
    human_casual = collect_human_casual(human_dedup, target_per_class // 3)
    human_texts.extend(human_casual)
    
    human_reviews = collect_human_reviews(human_dedup, target_per_class // 3)
    human_texts.extend(human_reviews)
    
    human_formal = collect_human_formal(human_dedup, target_per_class // 3)
    human_texts.extend(human_formal)
    
    # Collect AI text (various sources)
    ai_chatgpt = collect_ai_chatgpt(ai_dedup, target_per_class // 2)
    ai_texts.extend(ai_chatgpt)
    
    ai_instruct = collect_ai_instruct(ai_dedup, target_per_class // 2)
    ai_texts.extend(ai_instruct)
    
    # Remove any cross-contamination
    print("\n" + "="*70)
    print("CLEANING DATA")
    print("="*70)
    
    human_set = set(human_texts)
    ai_set = set(ai_texts)
    overlap = human_set & ai_set
    
    if overlap:
        print(f"  Removing {len(overlap):,} overlapping texts...")
        human_texts = [t for t in human_texts if t not in overlap]
        ai_texts = [t for t in ai_texts if t not in overlap]
    
    # Final deduplication
    print("  Final deduplication...")
    human_texts = list(set(human_texts))
    ai_texts = list(set(ai_texts))
    
    print(f"\n  Final counts:")
    print(f"    Human: {len(human_texts):,}")
    print(f"    AI:    {len(ai_texts):,}")
    
    return human_texts, ai_texts


def save_data(human_texts: List[str], ai_texts: List[str]):
    """Save collected data"""
    
    print("\n" + "="*70)
    print("SAVING DATA")
    print("="*70)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Balance
    min_size = min(len(human_texts), len(ai_texts))
    print(f"\n  Balancing to {min_size:,} per class...")
    
    random.seed(Config.SEED)
    random.shuffle(human_texts)
    random.shuffle(ai_texts)
    
    human_texts = human_texts[:min_size]
    ai_texts = ai_texts[:min_size]
    
    # Save
    print("  Saving pickle files...")
    
    with open(os.path.join(Config.OUTPUT_DIR, "human_texts.pkl"), "wb") as f:
        pickle.dump(human_texts, f)
    
    with open(os.path.join(Config.OUTPUT_DIR, "ai_texts.pkl"), "wb") as f:
        pickle.dump(ai_texts, f)
    
    # Save metadata
    metadata = {
        "human_count": len(human_texts),
        "ai_count": len(ai_texts),
        "created": datetime.now().isoformat(),
        "sources": dict(Config.SOURCES),
    }
    
    with open(os.path.join(Config.OUTPUT_DIR, "cache_info.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Calculate sizes
    human_size = os.path.getsize(os.path.join(Config.OUTPUT_DIR, "human_texts.pkl")) / (1024**2)
    ai_size = os.path.getsize(os.path.join(Config.OUTPUT_DIR, "ai_texts.pkl")) / (1024**2)
    
    print(f"\n  ✓ Saved!")
    print(f"    human_texts.pkl: {human_size:.1f} MB")
    print(f"    ai_texts.pkl: {ai_size:.1f} MB")


def show_samples(human_texts: List[str], ai_texts: List[str]):
    """Show sample texts for verification"""
    
    print("\n" + "="*70)
    print("SAMPLE DATA (Verify correctness!)")
    print("="*70)
    
    print("\n👤 HUMAN SAMPLES (should look like real human writing):")
    print("─"*60)
    for i, text in enumerate(random.sample(human_texts, min(5, len(human_texts)))):
        print(f"\n  [{i+1}] {text[:150]}...")
    
    print("\n\n🤖 AI SAMPLES (should look like AI assistant responses):")
    print("─"*60)
    for i, text in enumerate(random.sample(ai_texts, min(5, len(ai_texts)))):
        print(f"\n  [{i+1}] {text[:150]}...")


def main():
    parser = argparse.ArgumentParser(description="Collect AI detection training data")
    parser.add_argument('--target', type=int, default=500000,
                        help='Target samples per class (default: 500000)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    
    args = parser.parse_args()
    
    Config.TARGET_PER_CLASS = args.target
    if args.output:
        Config.OUTPUT_DIR = args.output
    
    # Collect
    human_texts, ai_texts = collect_all(Config.TARGET_PER_CLASS)
    
    # Save
    save_data(human_texts, ai_texts)
    
    # Show samples
    show_samples(human_texts, ai_texts)
    
    # Summary
    print("\n" + "="*70)
    print("✓ DATA COLLECTION COMPLETE!")
    print("="*70)
    print(f"\n  Location: {Config.OUTPUT_DIR}")
    print(f"  Human:    {len(human_texts):,}")
    print(f"  AI:       {len(ai_texts):,}")
    
    print("\n  Source breakdown:")
    for source, count in sorted(Config.SOURCES.items(), key=lambda x: -x[1])[:15]:
        print(f"    {source}: {count:,}")
    
    print(f"\n  NEXT STEP: Train with this data:")
    print(f"    python train.py --cache-dir \"{Config.OUTPUT_DIR}\" --samples 200000 --epochs 3")
    print("="*70)


if __name__ == "__main__":
    main()