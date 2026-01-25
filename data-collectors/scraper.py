"""
MASSIVE Scale Essay Dataset Downloader (20x Version)
Handles 100M+ texts with streaming, chunked processing, and memory optimization
"""

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("Checking packages...")
try:
    import pandas as pd
    from tqdm import tqdm
except ImportError:
    install("pandas")
    install("tqdm")
    import pandas as pd
    from tqdm import tqdm

try:
    from datasets import load_dataset, DatasetDict
except ImportError:
    install("datasets")
    from datasets import load_dataset

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:
    install("pyarrow")
    import pyarrow.parquet as pq
    import pyarrow as pa

try:
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
    import multiprocessing as mp
except ImportError:
    pass

import os
import gc
import json
import hashlib
import requests
import random
import time
from pathlib import Path
from typing import Generator, List, Tuple, Optional, Dict, Any
import numpy as np
from collections import defaultdict

# =============================================================================
# CONFIGURATION FOR 20x SCALE (100M+ texts)
# =============================================================================
CONFIG = {
    'target_human': 100_000_000,      # 100M human texts (was 5M, now 20x)
    'target_ai': 100_000_000,          # 100M AI texts (was 5M, now 20x)
    'chunk_size': 100_000,             # Larger chunks for efficiency (was 50k)
    'max_workers': 8,                  # More parallel downloads (was 4)
    'temp_dir': 'temp_chunks_20x',
    'output_dir': 'dataset_output_20x',
    'min_words': 15,                   # Slightly lower threshold for more data
    'max_words': 10000,                # Cap very long texts
    'max_file_size_mb': 2000,          # Larger chunk files
    'dedup_hash_limit': 500_000_000,   # 500M hash limit
    'streaming_buffer': 10000,         # Buffer size for streaming
    'save_interval': 25000,            # Save more frequently
    'max_per_dataset': 20_000_000,     # Max texts per dataset to ensure diversity
}

# Create directories
os.makedirs(CONFIG['temp_dir'], exist_ok=True)
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# Global counters
GLOBAL_CHUNK_ID = {'human': 0, 'ai': 0}


def get_text_hash(text: str) -> str:
    """Generate hash for deduplication"""
    return hashlib.md5(str(text)[:1000].encode()).hexdigest()[:16]


def save_chunk(chunk_data: List[dict], chunk_type: str):
    """Save chunk to parquet for memory efficiency"""
    if not chunk_data:
        return 0
    
    chunk_id = GLOBAL_CHUNK_ID[chunk_type]
    GLOBAL_CHUNK_ID[chunk_type] += 1
    
    df = pd.DataFrame(chunk_data)
    filepath = os.path.join(CONFIG['temp_dir'], f"{chunk_type}_chunk_{chunk_id:08d}.parquet")
    df.to_parquet(filepath, index=False, compression='snappy')
    saved = len(df)
    del df
    gc.collect()
    return saved


def load_chunk(filepath: str) -> pd.DataFrame:
    """Load chunk from parquet"""
    return pd.read_parquet(filepath)


class DeduplicationFilter:
    """Memory-efficient deduplication using bloom filter approximation"""
    
    def __init__(self, max_hashes: int = 500_000_000):
        self.seen_hashes = set()
        self.hash_file = os.path.join(CONFIG['temp_dir'], 'seen_hashes.bin')
        self.max_hashes = max_hashes
        self.collision_check = 0
        self._load_existing_hashes()
    
    def _load_existing_hashes(self):
        if os.path.exists(self.hash_file):
            try:
                with open(self.hash_file, 'rb') as f:
                    import pickle
                    self.seen_hashes = pickle.load(f)
                print(f"Loaded {len(self.seen_hashes):,} existing hashes")
            except:
                pass
    
    def is_duplicate(self, text: str) -> bool:
        text_hash = get_text_hash(text)
        if text_hash in self.seen_hashes:
            return True
        if len(self.seen_hashes) < self.max_hashes:
            self.seen_hashes.add(text_hash)
        return False
    
    def save_hashes(self):
        try:
            import pickle
            with open(self.hash_file, 'wb') as f:
                pickle.dump(self.seen_hashes, f)
        except Exception as e:
            print(f"Warning: Could not save hashes: {e}")
    
    def cleanup(self):
        self.save_hashes()
    
    def stats(self):
        return f"Unique hashes: {len(self.seen_hashes):,}"


def filter_text(text: str) -> bool:
    """Check if text meets quality criteria"""
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 50:
        return False
    words = text.split()
    word_count = len(words)
    if word_count < CONFIG['min_words'] or word_count > CONFIG['max_words']:
        return False
    # Remove very repetitive texts
    unique_ratio = len(set(words)) / word_count if word_count else 0
    if unique_ratio < 0.25:
        return False
    # Check for garbage text
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
    if alpha_ratio < 0.7:
        return False
    return True


def truncate_text(text: str, max_words: int = 2000) -> str:
    """Truncate very long texts to manageable size"""
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return text


class DatasetLoader:
    """Generic dataset loader with progress tracking"""
    
    def __init__(self, dedup_filter: DeduplicationFilter):
        self.dedup = dedup_filter
        self.stats = defaultdict(int)
    
    def process_texts(self, texts: List[str], label: str, source: str, max_count: int = None) -> Tuple[List[dict], int]:
        """Process a batch of texts with filtering and deduplication"""
        processed = []
        count = 0
        label_int = 0 if label == 'human' else 1
        
        for text in texts:
            if max_count and count >= max_count:
                break
            
            text = truncate_text(str(text).strip())
            
            if filter_text(text) and not self.dedup.is_duplicate(text):
                processed.append({
                    'text': text,
                    'label': label,
                    'label_int': label_int,
                    'source': source
                })
                count += 1
        
        return processed, count


# =============================================================================
# DATASET LOADING METHODS - EXPANDED FOR 20x SCALE
# =============================================================================

def method1_hc3_streaming(dedup_filter: DeduplicationFilter) -> Tuple[int, int]:
    """Load HC3 dataset with streaming - primary source"""
    print("\n" + "="*60)
    print("METHOD 1: HC3 Dataset (Streaming)")
    print("="*60)
    
    human_count = 0
    ai_count = 0
    human_chunk = []
    ai_chunk = []
    
    configs = ["all", "finance", "medicine", "open_qa", "wiki_csai", "reddit_eli5"]
    
    for config in configs:
        try:
            print(f"  Loading HC3 config: {config}...")
            dataset = load_dataset("Hello-SimpleAI/HC3", config, split="train", streaming=True)
            
            for item in tqdm(dataset, desc=f"HC3-{config}"):
                # Process human answers
                for ans in item.get('human_answers', []):
                    if ans and filter_text(ans) and not dedup_filter.is_duplicate(ans):
                        human_chunk.append({'text': truncate_text(ans), 'label': 'human', 'label_int': 0, 'source': f'hc3_{config}'})
                        human_count += 1
                
                # Process AI answers
                for ans in item.get('chatgpt_answers', []):
                    if ans and filter_text(ans) and not dedup_filter.is_duplicate(ans):
                        ai_chunk.append({'text': truncate_text(ans), 'label': 'ai', 'label_int': 1, 'source': f'hc3_{config}'})
                        ai_count += 1
                
                # Save chunks periodically
                if len(human_chunk) >= CONFIG['chunk_size']:
                    save_chunk(human_chunk, 'human')
                    human_chunk = []
                    gc.collect()
                if len(ai_chunk) >= CONFIG['chunk_size']:
                    save_chunk(ai_chunk, 'ai')
                    ai_chunk = []
                    gc.collect()
        
        except Exception as e:
            print(f"  ✗ HC3 {config} Error: {e}")
            continue
    
    # Save final chunks
    if human_chunk:
        save_chunk(human_chunk, 'human')
    if ai_chunk:
        save_chunk(ai_chunk, 'ai')
    
    print(f"✓ HC3: {human_count:,} human, {ai_count:,} AI texts")
    return human_count, ai_count


def method2_gpt_wiki_intro(dedup_filter: DeduplicationFilter) -> Tuple[int, int]:
    """Load GPT Wiki Intro dataset"""
    print("\n[2] Loading GPT Wiki Intro...")
    
    human_count = 0
    ai_count = 0
    human_chunk = []
    ai_chunk = []
    
    try:
        dataset = load_dataset("aadityaubhat/GPT-wiki-intro", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="GPT Wiki"):
            if 'wiki_intro' in item and item['wiki_intro']:
                text = item['wiki_intro']
                if filter_text(text) and not dedup_filter.is_duplicate(text):
                    human_chunk.append({'text': truncate_text(text), 'label': 'human', 'label_int': 0, 'source': 'gpt_wiki'})
                    human_count += 1
            
            if 'generated_intro' in item and item['generated_intro']:
                text = item['generated_intro']
                if filter_text(text) and not dedup_filter.is_duplicate(text):
                    ai_chunk.append({'text': truncate_text(text), 'label': 'ai', 'label_int': 1, 'source': 'gpt_wiki'})
                    ai_count += 1
            
            if len(human_chunk) >= CONFIG['chunk_size']:
                save_chunk(human_chunk, 'human')
                human_chunk = []
            if len(ai_chunk) >= CONFIG['chunk_size']:
                save_chunk(ai_chunk, 'ai')
                ai_chunk = []
        
        if human_chunk:
            save_chunk(human_chunk, 'human')
        if ai_chunk:
            save_chunk(ai_chunk, 'ai')
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print(f"   ✓ Got {human_count:,} human, {ai_count:,} AI texts")
    return human_count, ai_count


def method3_ai_detection_pile(dedup_filter: DeduplicationFilter) -> Tuple[int, int]:
    """Load AI Detection Pile - large dataset"""
    print("\n[3] Loading AI Detection Pile...")
    
    human_count = 0
    ai_count = 0
    human_chunk = []
    ai_chunk = []
    
    try:
        dataset = load_dataset("artem9k/ai-text-detection-pile", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="AI Detection Pile"):
            text = item.get('text', '')
            label = item.get('label', None)
            
            if not filter_text(text):
                continue
            
            text = truncate_text(text)
            
            if label == 0 and not dedup_filter.is_duplicate(text):
                human_chunk.append({'text': text, 'label': 'human', 'label_int': 0, 'source': 'ai_detection'})
                human_count += 1
            elif label == 1 and not dedup_filter.is_duplicate(text):
                ai_chunk.append({'text': text, 'label': 'ai', 'label_int': 1, 'source': 'ai_detection'})
                ai_count += 1
            
            if len(human_chunk) >= CONFIG['chunk_size']:
                save_chunk(human_chunk, 'human')
                human_chunk = []
            if len(ai_chunk) >= CONFIG['chunk_size']:
                save_chunk(ai_chunk, 'ai')
                ai_chunk = []
        
        if human_chunk:
            save_chunk(human_chunk, 'human')
        if ai_chunk:
            save_chunk(ai_chunk, 'ai')
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print(f"   ✓ Got {human_count:,} human, {ai_count:,} AI texts")
    return human_count, ai_count


def method4_wikipedia_massive(dedup_filter: DeduplicationFilter) -> int:
    """Load Wikipedia for human texts - MASSIVE scale"""
    print("\n[4] Loading Wikipedia (MASSIVE - all articles)...")
    
    count = 0
    chunk = []
    max_count = CONFIG['max_per_dataset']
    
    try:
        dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="Wikipedia"):
            if count >= max_count:
                break
                
            text = item.get('text', '')
            
            # Split into multiple paragraphs to get more samples
            paragraphs = text.split('\n\n')
            for para in paragraphs[:5]:  # Take first 5 paragraphs
                if len(para.split()) >= CONFIG['min_words']:
                    if filter_text(para) and not dedup_filter.is_duplicate(para):
                        chunk.append({'text': truncate_text(para), 'label': 'human', 'label_int': 0, 'source': 'wikipedia'})
                        count += 1
                        
                        if count >= max_count:
                            break
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'human')
                chunk = []
                gc.collect()
        
        if chunk:
            save_chunk(chunk, 'human')
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print(f"   ✓ Got {count:,} human texts")
    return count


def method5_cnn_dailymail(dedup_filter: DeduplicationFilter) -> int:
    """Load CNN/DailyMail for human texts"""
    print("\n[5] Loading CNN/DailyMail (human)...")
    
    count = 0
    chunk = []
    
    try:
        for split in ['train', 'validation', 'test']:
            dataset = load_dataset("cnn_dailymail", "3.0.0", split=split, streaming=True)
            
            for item in tqdm(dataset, desc=f"CNN/DM-{split}"):
                # Get article
                text = item.get('article', '')
                if filter_text(text) and not dedup_filter.is_duplicate(text):
                    chunk.append({'text': truncate_text(text), 'label': 'human', 'label_int': 0, 'source': 'cnn_dm'})
                    count += 1
                
                # Also get highlights as separate text
                highlights = item.get('highlights', '')
                if highlights and filter_text(highlights) and not dedup_filter.is_duplicate(highlights):
                    chunk.append({'text': truncate_text(highlights), 'label': 'human', 'label_int': 0, 'source': 'cnn_dm_highlights'})
                    count += 1
                
                if len(chunk) >= CONFIG['chunk_size']:
                    save_chunk(chunk, 'human')
                    chunk = []
        
        if chunk:
            save_chunk(chunk, 'human')
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print(f"   ✓ Got {count:,} human texts")
    return count


def method6_xsum(dedup_filter: DeduplicationFilter) -> int:
    """Load XSum for human texts"""
    print("\n[6] Loading XSum (human)...")
    
    count = 0
    chunk = []
    
    try:
        for split in ['train', 'validation', 'test']:
            dataset = load_dataset("xsum", split=split, streaming=True)
            
            for item in tqdm(dataset, desc=f"XSum-{split}"):
                text = item.get('document', '')
                if filter_text(text) and not dedup_filter.is_duplicate(text):
                    chunk.append({'text': truncate_text(text), 'label': 'human', 'label_int': 0, 'source': 'xsum'})
                    count += 1
                
                if len(chunk) >= CONFIG['chunk_size']:
                    save_chunk(chunk, 'human')
                    chunk = []
        
        if chunk:
            save_chunk(chunk, 'human')
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print(f"   ✓ Got {count:,} human texts")
    return count


def method7_openwebtext(dedup_filter: DeduplicationFilter) -> int:
    """Load OpenWebText - MASSIVE human text source"""
    print("\n[7] Loading OpenWebText (MASSIVE)...")
    
    count = 0
    chunk = []
    max_count = CONFIG['max_per_dataset']
    
    try:
        dataset = load_dataset("openwebtext", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="OpenWebText"):
            if count >= max_count:
                break
                
            text = item.get('text', '')
            if filter_text(text) and not dedup_filter.is_duplicate(text):
                chunk.append({'text': truncate_text(text), 'label': 'human', 'label_int': 0, 'source': 'openwebtext'})
                count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'human')
                chunk = []
                gc.collect()
        
        if chunk:
            save_chunk(chunk, 'human')
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print(f"   ✓ Got {count:,} human texts")
    return count


def method8_c4(dedup_filter: DeduplicationFilter) -> int:
    """Load C4 (Colossal Clean Crawled Corpus) - MASSIVE"""
    print("\n[8] Loading C4 (MASSIVE)...")
    
    count = 0
    chunk = []
    max_count = CONFIG['max_per_dataset']
    
    try:
        dataset = load_dataset("c4", "en", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="C4"):
            if count >= max_count:
                break
                
            text = item.get('text', '')
            if filter_text(text) and not dedup_filter.is_duplicate(text):
                chunk.append({'text': truncate_text(text), 'label': 'human', 'label_int': 0, 'source': 'c4'})
                count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'human')
                chunk = []
                gc.collect()
        
        if chunk:
            save_chunk(chunk, 'human')
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print(f"   ✓ Got {count:,} human texts")
    return count


def method9_reddit_writing(dedup_filter: DeduplicationFilter) -> int:
    """Load Reddit WritingPrompts and other writing subreddits"""
    print("\n[9] Loading Reddit Writing datasets...")
    
    count = 0
    chunk = []
    
    reddit_datasets = [
        ("reddit", "writing_prompts_train"),
        ("eli5", None),
    ]
    
    # WritingPrompts
    try:
        print("   Loading WritingPrompts...")
        dataset = load_dataset("euclaise/writingprompts", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="WritingPrompts"):
            text = item.get('story', '') or item.get('text', '')
            if filter_text(text) and not dedup_filter.is_duplicate(text):
                chunk.append({'text': truncate_text(text), 'label': 'human', 'label_int': 0, 'source': 'writing_prompts'})
                count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'human')
                chunk = []
    except Exception as e:
        print(f"   ✗ WritingPrompts Error: {e}")
    
    # ELI5
    try:
        print("   Loading ELI5...")
        dataset = load_dataset("eli5", split="train_eli5", streaming=True)
        
        for item in tqdm(dataset, desc="ELI5"):
            # Get answers
            answers = item.get('answers', {})
            if isinstance(answers, dict):
                texts = answers.get('text', [])
                for text in texts if isinstance(texts, list) else [texts]:
                    if filter_text(str(text)) and not dedup_filter.is_duplicate(str(text)):
                        chunk.append({'text': truncate_text(str(text)), 'label': 'human', 'label_int': 0, 'source': 'eli5'})
                        count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'human')
                chunk = []
    except Exception as e:
        print(f"   ✗ ELI5 Error: {e}")
    
    if chunk:
        save_chunk(chunk, 'human')
    
    print(f"   ✓ Got {count:,} human texts")
    return count


def method10_academic_papers(dedup_filter: DeduplicationFilter) -> int:
    """Load academic paper abstracts"""
    print("\n[10] Loading Academic Papers...")
    
    count = 0
    chunk = []
    max_count = CONFIG['max_per_dataset']
    
    # ArXiv
    try:
        print("   Loading ArXiv abstracts...")
        dataset = load_dataset("ccdv/arxiv-summarization", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="ArXiv"):
            if count >= max_count // 2:
                break
            text = item.get('abstract', '')
            if filter_text(text) and not dedup_filter.is_duplicate(text):
                chunk.append({'text': truncate_text(text), 'label': 'human', 'label_int': 0, 'source': 'arxiv'})
                count += 1
            
            # Also get article text
            article = item.get('article', '')
            if article and filter_text(article) and not dedup_filter.is_duplicate(article):
                chunk.append({'text': truncate_text(article, max_words=1500), 'label': 'human', 'label_int': 0, 'source': 'arxiv'})
                count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'human')
                chunk = []
    except Exception as e:
        print(f"   ✗ ArXiv Error: {e}")
    
    # PubMed
    try:
        print("   Loading PubMed abstracts...")
        dataset = load_dataset("ccdv/pubmed-summarization", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="PubMed"):
            if count >= max_count:
                break
            text = item.get('abstract', '')
            if filter_text(text) and not dedup_filter.is_duplicate(text):
                chunk.append({'text': truncate_text(text), 'label': 'human', 'label_int': 0, 'source': 'pubmed'})
                count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'human')
                chunk = []
    except Exception as e:
        print(f"   ✗ PubMed Error: {e}")
    
    if chunk:
        save_chunk(chunk, 'human')
    
    print(f"   ✓ Got {count:,} human texts")
    return count


def method11_reviews(dedup_filter: DeduplicationFilter) -> int:
    """Load review datasets - Yelp, Amazon, IMDB"""
    print("\n[11] Loading Review Datasets...")
    
    count = 0
    chunk = []
    max_per_source = CONFIG['max_per_dataset'] // 3
    
    # Yelp
    try:
        print("   Loading Yelp reviews...")
        dataset = load_dataset("yelp_review_full", split="train", streaming=True)
        yelp_count = 0
        
        for item in tqdm(dataset, desc="Yelp"):
            if yelp_count >= max_per_source:
                break
            text = item.get('text', '')
            if filter_text(text) and not dedup_filter.is_duplicate(text):
                chunk.append({'text': truncate_text(text), 'label': 'human', 'label_int': 0, 'source': 'yelp'})
                count += 1
                yelp_count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'human')
                chunk = []
    except Exception as e:
        print(f"   ✗ Yelp Error: {e}")
    
    # Amazon
    try:
        print("   Loading Amazon reviews...")
        dataset = load_dataset("amazon_polarity", split="train", streaming=True)
        amazon_count = 0
        
        for item in tqdm(dataset, desc="Amazon"):
            if amazon_count >= max_per_source:
                break
            text = item.get('content', '')
            if filter_text(text) and not dedup_filter.is_duplicate(text):
                chunk.append({'text': truncate_text(text), 'label': 'human', 'label_int': 0, 'source': 'amazon'})
                count += 1
                amazon_count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'human')
                chunk = []
    except Exception as e:
        print(f"   ✗ Amazon Error: {e}")
    
    # IMDB
    try:
        print("   Loading IMDB reviews...")
        dataset = load_dataset("imdb", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="IMDB"):
            text = item.get('text', '')
            if filter_text(text) and not dedup_filter.is_duplicate(text):
                chunk.append({'text': truncate_text(text), 'label': 'human', 'label_int': 0, 'source': 'imdb'})
                count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'human')
                chunk = []
    except Exception as e:
        print(f"   ✗ IMDB Error: {e}")
    
    if chunk:
        save_chunk(chunk, 'human')
    
    print(f"   ✓ Got {count:,} human texts")
    return count


def method12_books_stories(dedup_filter: DeduplicationFilter) -> int:
    """Load books and stories datasets"""
    print("\n[12] Loading Books and Stories...")
    
    count = 0
    chunk = []
    max_count = CONFIG['max_per_dataset']
    
    # BookCorpus-style
    try:
        print("   Loading BookCorpus...")
        dataset = load_dataset("bookcorpus", split="train", streaming=True)
        
        buffer = []
        for item in tqdm(dataset, desc="BookCorpus"):
            if count >= max_count:
                break
            text = item.get('text', '')
            buffer.append(text)
            
            # Combine sentences into paragraphs
            if len(buffer) >= 10:
                combined = ' '.join(buffer)
                if filter_text(combined) and not dedup_filter.is_duplicate(combined):
                    chunk.append({'text': truncate_text(combined), 'label': 'human', 'label_int': 0, 'source': 'bookcorpus'})
                    count += 1
                buffer = []
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'human')
                chunk = []
    except Exception as e:
        print(f"   ✗ BookCorpus Error: {e}")
    
    # PG-19 (Project Gutenberg)
    try:
        print("   Loading PG-19 (Project Gutenberg)...")
        dataset = load_dataset("pg19", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="PG-19"):
            if count >= max_count:
                break
            text = item.get('text', '')
            # Split into chunks
            words = text.split()
            for i in range(0, len(words), 500):
                chunk_text = ' '.join(words[i:i+500])
                if filter_text(chunk_text) and not dedup_filter.is_duplicate(chunk_text):
                    chunk.append({'text': chunk_text, 'label': 'human', 'label_int': 0, 'source': 'pg19'})
                    count += 1
                    if count >= max_count:
                        break
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'human')
                chunk = []
    except Exception as e:
        print(f"   ✗ PG-19 Error: {e}")
    
    if chunk:
        save_chunk(chunk, 'human')
    
    print(f"   ✓ Got {count:,} human texts")
    return count


def method13_news_articles(dedup_filter: DeduplicationFilter) -> int:
    """Load news article datasets"""
    print("\n[13] Loading News Articles...")
    
    count = 0
    chunk = []
    
    news_datasets = [
        ("multi_news", "train", "document"),
        ("gigaword", "train", "document"),
        ("newsroom", "train", "text"),
    ]
    
    for ds_name, split, text_field in news_datasets:
        try:
            print(f"   Loading {ds_name}...")
            
            if ds_name == "newsroom":
                dataset = load_dataset("newsroom", split=split, streaming=True, data_dir="release")
            else:
                dataset = load_dataset(ds_name, split=split, streaming=True)
            
            ds_count = 0
            for item in tqdm(dataset, desc=ds_name):
                if ds_count >= 2_000_000:  # Limit per dataset
                    break
                text = item.get(text_field, '') or item.get('text', '')
                if filter_text(text) and not dedup_filter.is_duplicate(text):
                    chunk.append({'text': truncate_text(text), 'label': 'human', 'label_int': 0, 'source': ds_name})
                    count += 1
                    ds_count += 1
                
                if len(chunk) >= CONFIG['chunk_size']:
                    save_chunk(chunk, 'human')
                    chunk = []
        except Exception as e:
            print(f"   ✗ {ds_name} Error: {e}")
            continue
    
    if chunk:
        save_chunk(chunk, 'human')
    
    print(f"   ✓ Got {count:,} human texts")
    return count


def method14_ai_generated_datasets(dedup_filter: DeduplicationFilter) -> int:
    """Load various AI-generated text datasets"""
    print("\n[14] Loading AI-Generated Datasets...")
    
    count = 0
    chunk = []
    
    ai_datasets = [
        ("Hello-SimpleAI/HC3-Chinese", "all", "chatgpt_answers"),
        ("aeslc", "train", None),  # Email summaries (often AI-like)
    ]
    
    # OpenAI Detector training data
    try:
        print("   Loading GPT-2 Output Dataset...")
        dataset = load_dataset("openai/gpt-2-output-dataset", "xl", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="GPT-2 Output"):
            text = item.get('text', '')
            if filter_text(text) and not dedup_filter.is_duplicate(text):
                chunk.append({'text': truncate_text(text), 'label': 'ai', 'label_int': 1, 'source': 'gpt2_output'})
                count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'ai')
                chunk = []
    except Exception as e:
        print(f"   ✗ GPT-2 Output Error: {e}")
    
    # Anthropic HH-RLHF (AI responses)
    try:
        print("   Loading Anthropic HH-RLHF...")
        dataset = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="HH-RLHF"):
            # Extract assistant responses
            chosen = item.get('chosen', '')
            if 'Assistant:' in chosen:
                parts = chosen.split('Assistant:')
                for part in parts[1:]:
                    text = part.split('Human:')[0].strip()
                    if filter_text(text) and not dedup_filter.is_duplicate(text):
                        chunk.append({'text': truncate_text(text), 'label': 'ai', 'label_int': 1, 'source': 'hh_rlhf'})
                        count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'ai')
                chunk = []
    except Exception as e:
        print(f"   ✗ HH-RLHF Error: {e}")
    
    # ShareGPT
    try:
        print("   Loading ShareGPT...")
        dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="ShareGPT"):
            conversations = item.get('conversations', [])
            for conv in conversations:
                if conv.get('from') in ['gpt', 'assistant', 'chatgpt']:
                    text = conv.get('value', '')
                    if filter_text(text) and not dedup_filter.is_duplicate(text):
                        chunk.append({'text': truncate_text(text), 'label': 'ai', 'label_int': 1, 'source': 'sharegpt'})
                        count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'ai')
                chunk = []
    except Exception as e:
        print(f"   ✗ ShareGPT Error: {e}")
    
    if chunk:
        save_chunk(chunk, 'ai')
    
    print(f"   ✓ Got {count:,} AI texts")
    return count


def method15_dolly_oasst(dedup_filter: DeduplicationFilter) -> int:
    """Load Dolly and OpenAssistant datasets (AI responses)"""
    print("\n[15] Loading Dolly and OpenAssistant...")
    
    count = 0
    chunk = []
    
    # Dolly
    try:
        print("   Loading Dolly...")
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="Dolly"):
            text = item.get('response', '')
            if filter_text(text) and not dedup_filter.is_duplicate(text):
                chunk.append({'text': truncate_text(text), 'label': 'ai', 'label_int': 1, 'source': 'dolly'})
                count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'ai')
                chunk = []
    except Exception as e:
        print(f"   ✗ Dolly Error: {e}")
    
    # OpenAssistant
    try:
        print("   Loading OpenAssistant...")
        dataset = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="OASST"):
            if item.get('role') == 'assistant':
                text = item.get('text', '')
                if filter_text(text) and not dedup_filter.is_duplicate(text):
                    chunk.append({'text': truncate_text(text), 'label': 'ai', 'label_int': 1, 'source': 'oasst'})
                    count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'ai')
                chunk = []
    except Exception as e:
        print(f"   ✗ OpenAssistant Error: {e}")
    
    # FLAN collection
    try:
        print("   Loading FLAN...")
        dataset = load_dataset("Muennighoff/flan", split="train", streaming=True)
        flan_count = 0
        
        for item in tqdm(dataset, desc="FLAN"):
            if flan_count >= 5_000_000:
                break
            text = item.get('target', '') or item.get('targets', '')
            if filter_text(text) and not dedup_filter.is_duplicate(text):
                chunk.append({'text': truncate_text(text), 'label': 'ai', 'label_int': 1, 'source': 'flan'})
                count += 1
                flan_count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'ai')
                chunk = []
    except Exception as e:
        print(f"   ✗ FLAN Error: {e}")
    
    if chunk:
        save_chunk(chunk, 'ai')
    
    print(f"   ✓ Got {count:,} AI texts")
    return count


def method16_alpaca_vicuna(dedup_filter: DeduplicationFilter) -> int:
    """Load Alpaca and Vicuna style datasets"""
    print("\n[16] Loading Alpaca/Vicuna datasets...")
    
    count = 0
    chunk = []
    
    alpaca_datasets = [
        "tatsu-lab/alpaca",
        "WizardLM/WizardLM_evol_instruct_V2_196k",
        "teknium/GPTeacher-General-Instruct",
    ]
    
    for ds_name in alpaca_datasets:
        try:
            print(f"   Loading {ds_name}...")
            dataset = load_dataset(ds_name, split="train", streaming=True)
            ds_count = 0
            
            for item in tqdm(dataset, desc=ds_name.split('/')[-1][:20]):
                if ds_count >= 2_000_000:
                    break
                text = item.get('output', '') or item.get('response', '') or item.get('text', '')
                if filter_text(text) and not dedup_filter.is_duplicate(text):
                    chunk.append({'text': truncate_text(text), 'label': 'ai', 'label_int': 1, 'source': ds_name.split('/')[-1]})
                    count += 1
                    ds_count += 1
                
                if len(chunk) >= CONFIG['chunk_size']:
                    save_chunk(chunk, 'ai')
                    chunk = []
        except Exception as e:
            print(f"   ✗ {ds_name} Error: {e}")
            continue
    
    if chunk:
        save_chunk(chunk, 'ai')
    
    print(f"   ✓ Got {count:,} AI texts")
    return count


def method17_essays_specific(dedup_filter: DeduplicationFilter) -> int:
    """Load essay-specific datasets"""
    print("\n[17] Loading Essay-Specific Datasets...")
    
    count = 0
    chunk = []
    
    essay_datasets = [
        "qwedsacf/grade-style-essays",
        "wikitext",
    ]
    
    for ds_name in essay_datasets:
        try:
            print(f"   Loading {ds_name}...")
            
            if ds_name == "wikitext":
                dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
            else:
                dataset = load_dataset(ds_name, split="train", streaming=True)
            
            buffer = []
            for item in tqdm(dataset, desc=ds_name.split('/')[-1]):
                text = item.get('essay', '') or item.get('text', '')
                
                if ds_name == "wikitext":
                    buffer.append(text)
                    if len(buffer) >= 20:
                        combined = ' '.join(buffer)
                        if filter_text(combined) and not dedup_filter.is_duplicate(combined):
                            chunk.append({'text': truncate_text(combined), 'label': 'human', 'label_int': 0, 'source': ds_name})
                            count += 1
                        buffer = []
                else:
                    if filter_text(text) and not dedup_filter.is_duplicate(text):
                        chunk.append({'text': truncate_text(text), 'label': 'human', 'label_int': 0, 'source': ds_name})
                        count += 1
                
                if len(chunk) >= CONFIG['chunk_size']:
                    save_chunk(chunk, 'human')
                    chunk = []
        except Exception as e:
            print(f"   ✗ {ds_name} Error: {e}")
            continue
    
    if chunk:
        save_chunk(chunk, 'human')
    
    print(f"   ✓ Got {count:,} human texts")
    return count


def method18_squad_qa(dedup_filter: DeduplicationFilter) -> int:
    """Load Q&A datasets for human context texts"""
    print("\n[18] Loading Q&A Datasets...")
    
    count = 0
    chunk = []
    
    # SQuAD
    try:
        print("   Loading SQuAD...")
        dataset = load_dataset("squad", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="SQuAD"):
            text = item.get('context', '')
            if filter_text(text) and not dedup_filter.is_duplicate(text):
                chunk.append({'text': truncate_text(text), 'label': 'human', 'label_int': 0, 'source': 'squad'})
                count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'human')
                chunk = []
    except Exception as e:
        print(f"   ✗ SQuAD Error: {e}")
    
    # Natural Questions
    try:
        print("   Loading Natural Questions...")
        dataset = load_dataset("natural_questions", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="NQ"):
            doc = item.get('document', {})
            text = doc.get('text', '') if isinstance(doc, dict) else str(doc)
            if filter_text(text) and not dedup_filter.is_duplicate(text):
                chunk.append({'text': truncate_text(text, 1000), 'label': 'human', 'label_int': 0, 'source': 'natural_questions'})
                count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'human')
                chunk = []
    except Exception as e:
        print(f"   ✗ Natural Questions Error: {e}")
    
    if chunk:
        save_chunk(chunk, 'human')
    
    print(f"   ✓ Got {count:,} human texts")
    return count


def method19_synthetic_ai_massive(dedup_filter: DeduplicationFilter, needed: int) -> int:
    """Generate massive synthetic AI texts"""
    print(f"\n[19] Generating {needed:,} Synthetic AI Texts...")
    
    count = 0
    chunk = []
    
    # More diverse templates
    templates = {
        'essay': [
            "In the contemporary landscape of {topic}, we observe significant developments that warrant careful examination. This analysis delves into the multifaceted aspects of this phenomenon, exploring both theoretical frameworks and practical implications.",
            "The evolution of {topic} represents a paradigm shift in our understanding. As we examine the underlying principles, it becomes evident that multiple factors contribute to this complex dynamic.",
            "When considering {topic}, one must acknowledge the intricate interplay between various elements. This essay provides a comprehensive overview of the current state of knowledge.",
            "The discourse surrounding {topic} has garnered considerable attention in recent years. This examination seeks to illuminate key aspects while acknowledging existing complexities.",
            "{topic} stands as a defining challenge of our era. Through systematic analysis, we can better understand the mechanisms at play and their broader implications.",
        ],
        'analytical': [
            "An examination of {topic} reveals several critical insights. First, we must consider the foundational principles that govern this domain.",
            "The analysis of {topic} necessitates a methodological approach. By examining the available evidence, we can draw meaningful conclusions.",
            "To understand {topic}, we must first establish a framework for analysis. The following discussion presents key considerations.",
        ],
        'explanatory': [
            "{topic} can be understood through multiple lenses. At its core, this concept encompasses several interrelated elements.",
            "The concept of {topic} merits careful explanation. Fundamentally, it involves the interaction of various components.",
            "Understanding {topic} requires familiarity with its basic principles. Let us examine the key aspects systematically.",
        ],
        'argumentative': [
            "The case for {topic} rests on substantial evidence. This argument presents the key supporting factors and addresses potential counterarguments.",
            "There are compelling reasons to consider {topic} seriously. The following analysis presents the primary arguments and their supporting evidence.",
        ]
    }
    
    topics = [
        "artificial intelligence and machine learning", "climate change mitigation strategies",
        "sustainable urban development", "digital transformation in education",
        "healthcare system optimization", "renewable energy technologies",
        "economic policy reform", "social media's impact on society",
        "cybersecurity challenges", "biodiversity conservation",
        "mental health awareness", "food security and agriculture",
        "space exploration advancements", "quantum computing applications",
        "blockchain technology", "autonomous vehicles",
        "gene editing ethics", "water resource management",
        "global trade dynamics", "democratic governance",
        "cultural preservation", "technological unemployment",
        "data privacy concerns", "public health infrastructure",
        "educational equity", "housing affordability",
        "transportation infrastructure", "waste management solutions",
        "energy efficiency", "marine ecosystem protection",
    ]
    
    transitions = [
        "Furthermore, ", "Additionally, ", "Moreover, ", "It is worth noting that ",
        "Significantly, ", "In this context, ", "Consequently, ", "As a result, ",
        "Building on this, ", "Extending this analysis, ", "This suggests that ",
    ]
    
    developments = [
        "research indicates substantial progress in this area.",
        "scholars have identified key patterns worthy of attention.",
        "the evidence suggests multiple interconnected factors.",
        "recent developments have highlighted new dimensions.",
        "this perspective aligns with contemporary theoretical frameworks.",
        "empirical studies provide supporting evidence for these claims.",
        "cross-disciplinary approaches have yielded valuable insights.",
        "the implications extend across multiple domains.",
    ]
    
    conclusions = [
        "In conclusion, this analysis demonstrates the multifaceted nature of the subject matter. Continued investigation remains essential for advancing our understanding.",
        "To summarize, the evidence presented supports a nuanced view of this topic. Future research should build upon these foundations.",
        "Ultimately, addressing these challenges requires collaborative, interdisciplinary efforts. The path forward demands both innovation and careful consideration.",
        "In sum, this examination reveals the complexity inherent in this domain. Stakeholders must consider multiple perspectives when developing solutions.",
    ]
    
    for i in tqdm(range(min(needed, 50_000_000)), desc="Synthetic AI"):
        # Select template type and template
        template_type = random.choice(list(templates.keys()))
        template = random.choice(templates[template_type])
        topic = random.choice(topics)
        
        # Build the text
        text = template.format(topic=topic)
        
        # Add body paragraphs
        num_paragraphs = random.randint(2, 5)
        for _ in range(num_paragraphs):
            paragraph = random.choice(transitions)
            paragraph += random.choice(developments) + " "
            paragraph += random.choice(transitions).lower()
            paragraph += random.choice(developments)
            text += "\n\n" + paragraph
        
        # Add conclusion
        text += "\n\n" + random.choice(conclusions)
        
        if filter_text(text) and not dedup_filter.is_duplicate(text):
            chunk.append({'text': text, 'label': 'ai', 'label_int': 1, 'source': 'synthetic_ai'})
            count += 1
        
        if len(chunk) >= CONFIG['chunk_size']:
            save_chunk(chunk, 'ai')
            chunk = []
            gc.collect()
    
    if chunk:
        save_chunk(chunk, 'ai')
    
    print(f"   ✓ Generated {count:,} synthetic AI texts")
    return count


def method20_additional_human_sources(dedup_filter: DeduplicationFilter) -> int:
    """Load additional human text sources"""
    print("\n[20] Loading Additional Human Sources...")
    
    count = 0
    chunk = []
    
    additional_datasets = [
        ("cc_news", "train", "text"),
        ("scientific_papers", "arxiv", "article"),
        ("wikihow", "all", "text"),
    ]
    
    # CC-News
    try:
        print("   Loading CC-News...")
        dataset = load_dataset("cc_news", split="train", streaming=True)
        cc_count = 0
        
        for item in tqdm(dataset, desc="CC-News"):
            if cc_count >= 5_000_000:
                break
            text = item.get('text', '')
            if filter_text(text) and not dedup_filter.is_duplicate(text):
                chunk.append({'text': truncate_text(text), 'label': 'human', 'label_int': 0, 'source': 'cc_news'})
                count += 1
                cc_count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'human')
                chunk = []
    except Exception as e:
        print(f"   ✗ CC-News Error: {e}")
    
    # WikiHow
    try:
        print("   Loading WikiHow...")
        dataset = load_dataset("wikihow", "all", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="WikiHow"):
            text = item.get('text', '')
            if filter_text(text) and not dedup_filter.is_duplicate(text):
                chunk.append({'text': truncate_text(text), 'label': 'human', 'label_int': 0, 'source': 'wikihow'})
                count += 1
            
            if len(chunk) >= CONFIG['chunk_size']:
                save_chunk(chunk, 'human')
                chunk = []
    except Exception as e:
        print(f"   ✗ WikiHow Error: {e}")
    
    if chunk:
        save_chunk(chunk, 'human')
    
    print(f"   ✓ Got {count:,} human texts")
    return count


def merge_chunks_to_final():
    """Merge all chunk files into final dataset files - optimized for massive scale"""
    print("\n" + "="*60)
    print("MERGING CHUNKS INTO FINAL DATASET (20x Scale)")
    print("="*60)
    
    human_files = sorted([f for f in os.listdir(CONFIG['temp_dir']) if f.startswith('human_') and f.endswith('.parquet')])
    ai_files = sorted([f for f in os.listdir(CONFIG['temp_dir']) if f.startswith('ai_') and f.endswith('.parquet')])
    
    print(f"Found {len(human_files)} human chunks, {len(ai_files)} AI chunks")
    
    # Process in larger batches for efficiency
    batch_size = 50
    
    # Merge human chunks
    print("\nMerging human texts...")
    human_count = 0
    human_output_idx = 0
    accumulated_dfs = []
    
    for i, f in enumerate(tqdm(human_files)):
        df = load_chunk(os.path.join(CONFIG['temp_dir'], f))
        accumulated_dfs.append(df)
        
        if len(accumulated_dfs) >= batch_size or i == len(human_files) - 1:
            merged = pd.concat(accumulated_dfs, ignore_index=True)
            human_count += len(merged)
            
            # Save as sharded files if very large
            if len(merged) > 10_000_000:
                for j in range(0, len(merged), 10_000_000):
                    shard = merged.iloc[j:j+10_000_000]
                    shard.to_parquet(os.path.join(CONFIG['output_dir'], f'human_essays_part{human_output_idx:03d}.parquet'), index=False)
                    human_output_idx += 1
            else:
                merged.to_parquet(os.path.join(CONFIG['output_dir'], f'human_essays_part{human_output_idx:03d}.parquet'), index=False)
                human_output_idx += 1
            
            accumulated_dfs = []
            gc.collect()
    
    print(f"✓ Saved {human_count:,} human texts in {human_output_idx} files")
    
    # Merge AI chunks
    print("\nMerging AI texts...")
    ai_count = 0
    ai_output_idx = 0
    accumulated_dfs = []
    
    for i, f in enumerate(tqdm(ai_files)):
        df = load_chunk(os.path.join(CONFIG['temp_dir'], f))
        accumulated_dfs.append(df)
        
        if len(accumulated_dfs) >= batch_size or i == len(ai_files) - 1:
            merged = pd.concat(accumulated_dfs, ignore_index=True)
            ai_count += len(merged)
            
            if len(merged) > 10_000_000:
                for j in range(0, len(merged), 10_000_000):
                    shard = merged.iloc[j:j+10_000_000]
                    shard.to_parquet(os.path.join(CONFIG['output_dir'], f'ai_essays_part{ai_output_idx:03d}.parquet'), index=False)
                    ai_output_idx += 1
            else:
                merged.to_parquet(os.path.join(CONFIG['output_dir'], f'ai_essays_part{ai_output_idx:03d}.parquet'), index=False)
                ai_output_idx += 1
            
            accumulated_dfs = []
            gc.collect()
    
    print(f"✓ Saved {ai_count:,} AI texts in {ai_output_idx} files")
    
    # Create balanced combined sample
    print("\nCreating balanced combined sample (10M each max)...")
    sample_size = min(10_000_000, human_count, ai_count)
    
    human_sample = []
    for f in os.listdir(CONFIG['output_dir']):
        if f.startswith('human_essays_part') and len(human_sample) < sample_size:
            df = pd.read_parquet(os.path.join(CONFIG['output_dir'], f))
            needed = sample_size - len(human_sample)
            human_sample.append(df.head(needed))
    
    ai_sample = []
    for f in os.listdir(CONFIG['output_dir']):
        if f.startswith('ai_essays_part') and len(ai_sample) < sample_size:
            df = pd.read_parquet(os.path.join(CONFIG['output_dir'], f))
            needed = sample_size - len(ai_sample)
            ai_sample.append(df.head(needed))
    
    if human_sample and ai_sample:
        human_df = pd.concat(human_sample, ignore_index=True).head(sample_size)
        ai_df = pd.concat(ai_sample, ignore_index=True).head(sample_size)
        
        combined = pd.concat([human_df, ai_df], ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        combined.to_parquet(os.path.join(CONFIG['output_dir'], 'combined_balanced_sample.parquet'), index=False)
        
        print(f"✓ Saved combined sample: {len(combined):,} texts")
    
    return human_count, ai_count


def print_statistics():
    """Print comprehensive dataset statistics"""
    print("\n" + "="*60)
    print("DATASET STATISTICS (20x Scale)")
    print("="*60)
    
    human_count = 0
    ai_count = 0
    
    for f in os.listdir(CONFIG['output_dir']):
        if f.endswith('.parquet'):
            filepath = os.path.join(CONFIG['output_dir'], f)
            df = pd.read_parquet(filepath)
            if 'human' in f:
                human_count += len(df)
            elif 'ai' in f:
                ai_count += len(df)
    
    print(f"\nTotal Human Texts: {human_count:,}")
    print(f"Total AI Texts: {ai_count:,}")
    print(f"Combined Total: {human_count + ai_count:,}")
    
    # Get file sizes
    total_size = 0
    for f in os.listdir(CONFIG['output_dir']):
        if f.endswith('.parquet'):
            total_size += os.path.getsize(os.path.join(CONFIG['output_dir'], f))
    
    print(f"\nTotal Dataset Size: {total_size / 1e9:.2f} GB")


def cleanup_temp_files():
    """Remove temporary chunk files"""
    print("\nCleaning up temporary files...")
    import shutil
    if os.path.exists(CONFIG['temp_dir']):
        shutil.rmtree(CONFIG['temp_dir'])
    print("✓ Cleanup complete")


def main():
    """Main function for 20x scale dataset download"""
    
    print("="*70)
    print("   MASSIVE ESSAY DATASET DOWNLOADER - 20x SCALE")
    print("   Target: 100M+ Human and AI Texts Each")
    print("="*70)
    
    start_time = time.time()
    
    # Initialize deduplication filter
    dedup_filter = DeduplicationFilter(max_hashes=CONFIG['dedup_hash_limit'])
    
    total_human = 0
    total_ai = 0
    
    try:
        # =================================================================
        # HUMAN + AI PAIRED DATASETS
        # =================================================================
        
        # Method 1: HC3 (primary paired source)
        h, a = method1_hc3_streaming(dedup_filter)
        total_human += h
        total_ai += a
        
        # Method 2: GPT Wiki Intro
        h, a = method2_gpt_wiki_intro(dedup_filter)
        total_human += h
        total_ai += a
        
        # Method 3: AI Detection Pile
        h, a = method3_ai_detection_pile(dedup_filter)
        total_human += h
        total_ai += a
        
        # =================================================================
        # MASSIVE HUMAN TEXT SOURCES
        # =================================================================
        
        # Method 4: Wikipedia (MASSIVE)
        h = method4_wikipedia_massive(dedup_filter)
        total_human += h
        
        # Method 5: CNN/DailyMail
        h = method5_cnn_dailymail(dedup_filter)
        total_human += h
        
        # Method 6: XSum
        h = method6_xsum(dedup_filter)
        total_human += h
        
        # Method 7: OpenWebText (MASSIVE)
        h = method7_openwebtext(dedup_filter)
        total_human += h
        
        # Method 8: C4 (MASSIVE)
        h = method8_c4(dedup_filter)
        total_human += h
        
        # Method 9: Reddit Writing
        h = method9_reddit_writing(dedup_filter)
        total_human += h
        
        # Method 10: Academic Papers
        h = method10_academic_papers(dedup_filter)
        total_human += h
        
        # Method 11: Reviews
        h = method11_reviews(dedup_filter)
        total_human += h
        
        # Method 12: Books and Stories
        h = method12_books_stories(dedup_filter)
        total_human += h
        
        # Method 13: News Articles
        h = method13_news_articles(dedup_filter)
        total_human += h
        
        # Method 17: Essay-specific
        h = method17_essays_specific(dedup_filter)
        total_human += h
        
        # Method 18: Q&A datasets
        h = method18_squad_qa(dedup_filter)
        total_human += h
        
        # Method 20: Additional sources
        h = method20_additional_human_sources(dedup_filter)
        total_human += h
        
        # =================================================================
        # AI-GENERATED TEXT SOURCES
        # =================================================================
        
        # Method 14: AI-generated datasets
        a = method14_ai_generated_datasets(dedup_filter)
        total_ai += a
        
        # Method 15: Dolly and OpenAssistant
        a = method15_dolly_oasst(dedup_filter)
        total_ai += a
        
        # Method 16: Alpaca/Vicuna
        a = method16_alpaca_vicuna(dedup_filter)
        total_ai += a
        
        print(f"\n{'='*70}")
        print(f"CURRENT TOTALS: {total_human:,} human, {total_ai:,} AI texts")
        print(f"{'='*70}")
        
        # =================================================================
        # SYNTHETIC AI GENERATION (to balance dataset)
        # =================================================================
        
        # Generate synthetic AI to match human texts
        if total_ai < total_human:
            needed = min(total_human - total_ai, CONFIG['target_ai'] - total_ai)
            if needed > 0:
                a = method19_synthetic_ai_massive(dedup_filter, int(needed))
                total_ai += a
        
        # Save deduplication state
        dedup_filter.save_hashes()
        
        # =================================================================
        # FINAL MERGE
        # =================================================================
        
        human_final, ai_final = merge_chunks_to_final()
        
        # Print statistics
        print_statistics()
        
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        print("\n" + "="*70)
        print("✓ DOWNLOAD COMPLETE!")
        print("="*70)
        print(f"\nFinal Dataset (20x Scale):")
        print(f"  - Human texts: {human_final:,}")
        print(f"  - AI texts: {ai_final:,}")
        print(f"  - Total: {human_final + ai_final:,}")
        print(f"\nTime elapsed: {hours}h {minutes}m")
        print(f"\nFiles saved in '{CONFIG['output_dir']}'")
        print(f"\nDeduplication stats: {dedup_filter.stats()}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted! Saving progress...")
        dedup_filter.save_hashes()
        merge_chunks_to_final()
        print("Progress saved. You can resume by running again.")
        
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        dedup_filter.save_hashes()
        
    finally:
        dedup_filter.cleanup()
        # Optionally cleanup temp files (comment out to keep for debugging)
        # cleanup_temp_files()


if __name__ == "__main__":
    main()