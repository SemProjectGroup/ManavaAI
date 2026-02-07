"""
Human Text Collection Module
============================
Collects human-written text from various ethical sources:
- Project Gutenberg (public domain books)
- Wikipedia (free encyclopedia)
- OpenWebText (public web text)
- Public news datasets
"""

import os
import re
import json
import time
import random
import asyncio
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Generator, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from loguru import logger

# Try to import optional dependencies
try:
    import wikipediaapi
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False
    logger.warning("wikipedia-api not installed. Wikipedia collection will be limited.")

try:
    from gutenbergpy import textget
    from gutenbergpy.gutenbergcache import GutenbergCache
    GUTENBERG_AVAILABLE = True
except ImportError:
    GUTENBERG_AVAILABLE = False
    logger.warning("gutenbergpy not installed. Will use alternative method.")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.warning("datasets library not installed. HuggingFace datasets won't be available.")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config


@dataclass
class TextSample:
    """Represents a collected text sample."""
    text: str
    source: str
    source_id: str
    metadata: Dict
    label: int = 0  # 0 = human, 1 = AI


class HumanTextCollector:
    """
    Collects human-written text from various ethical sources.
    
    Sources:
        - Project Gutenberg: Public domain books
        - Wikipedia: Free encyclopedia articles
        - OpenWebText: Public web text dataset
        - News datasets: Public news articles
        - Academic datasets: Public research papers
    """
    
    def __init__(self, output_path: Optional[Path] = None):
        """
        Initialize the human text collector.
        
        Args:
            output_path: Path to save collected texts
        """
        self.output_path = output_path or config.data_collection.raw_human_text_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.config = config.data_collection
        self.collected_hashes: Set[str] = set()
        self.samples_collected = 0
        
        # Initialize Wikipedia API if available
        if WIKIPEDIA_AVAILABLE:
            self.wiki_api = wikipediaapi.Wikipedia(
                user_agent='ManavAI/1.0 (Educational Project)',
                language='en'
            )
        
        logger.info(f"HumanTextCollector initialized. Output: {self.output_path}")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for deduplication."""
        return hashlib.md5(text.strip().lower().encode()).hexdigest()
    
    def _is_valid_text(self, text: str) -> bool:
        """Check if text meets quality criteria."""
        if not text or not text.strip():
            return False
        
        text = text.strip()
        word_count = len(text.split())
        
        # Length checks
        if len(text) < self.config.min_text_length:
            return False
        if len(text) > self.config.max_text_length:
            return False
        if word_count < self.config.min_word_count:
            return False
        if word_count > self.config.max_word_count:
            return False
        
        # Quality checks
        # Check for too many special characters
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.6:
            return False
        
        # Check for repetitive content
        words = text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return False
        
        # Deduplication
        text_hash = self._get_text_hash(text)
        if text_hash in self.collected_hashes:
            return False
        
        return True
    
    def _save_sample(self, sample: TextSample):
        """Save a single sample to the output file."""
        with open(self.output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'text': sample.text,
                'source': sample.source,
                'source_id': sample.source_id,
                'metadata': sample.metadata,
                'label': sample.label
            }) + '\n')
        
        self.collected_hashes.add(self._get_text_hash(sample.text))
        self.samples_collected += 1
    
    def _extract_paragraphs(self, text: str, min_length: int = 100) -> List[str]:
        """Extract paragraphs from a long text."""
        # Split by double newlines or paragraph markers
        paragraphs = re.split(r'\n\n+|\r\n\r\n+', text)
        
        # Clean and filter paragraphs
        valid_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            para = re.sub(r'\s+', ' ', para)  # Normalize whitespace
            
            if len(para) >= min_length and self._is_valid_text(para):
                valid_paragraphs.append(para)
        
        return valid_paragraphs
    
    # =========================================================================
    # PROJECT GUTENBERG COLLECTION
    # =========================================================================
    
    def collect_from_gutenberg(self, num_books: int = 500) -> int:
        """
        Collect text from Project Gutenberg public domain books.
        
        Args:
            num_books: Number of books to process
            
        Returns:
            Number of samples collected
        """
        logger.info(f"Collecting from Project Gutenberg ({num_books} books)...")
        initial_count = self.samples_collected
        
        if GUTENBERG_AVAILABLE:
            return self._collect_gutenberg_with_library(num_books)
        else:
            return self._collect_gutenberg_alternative(num_books)
    
    def _collect_gutenberg_with_library(self, num_books: int) -> int:
        """Collect using gutenbergpy library."""
        try:
            # Update cache if needed
            GutenbergCache.create()
            cache = GutenbergCache.get_cache()
            
            # Get list of English books
            books = list(cache.query(
                languages=['en'],
                has_text=True
            ))[:num_books]
            
            for book_id in tqdm(books, desc="Processing Gutenberg books"):
                try:
                    # Get book text
                    raw_text = textget.get_text_by_id(book_id)
                    if not raw_text:
                        continue
                    
                    text = raw_text.decode('utf-8', errors='ignore')
                    
                    # Extract paragraphs
                    paragraphs = self._extract_paragraphs(text)
                    
                    # Sample paragraphs
                    for i, para in enumerate(paragraphs[:self.config.gutenberg_paragraphs_per_book]):
                        sample = TextSample(
                            text=para,
                            source="gutenberg",
                            source_id=f"gutenberg_{book_id}_{i}",
                            metadata={"book_id": book_id},
                            label=0
                        )
                        self._save_sample(sample)
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    logger.debug(f"Error processing book {book_id}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Gutenberg library error: {e}")
            return self._collect_gutenberg_alternative(num_books)
        
        return self.samples_collected
    
    def _collect_gutenberg_alternative(self, num_books: int) -> int:
        """Alternative method using direct API access."""
        logger.info("Using alternative Gutenberg collection method...")
        
        # Popular Gutenberg book IDs (known to have good text)
        popular_books = [
            1342,   # Pride and Prejudice
            84,     # Frankenstein
            1661,   # Sherlock Holmes
            11,     # Alice's Adventures
            2701,   # Moby Dick
            1952,   # The Yellow Wallpaper
            98,     # A Tale of Two Cities
            174,    # Picture of Dorian Gray
            345,    # Dracula
            1232,   # The Prince
            2542,   # A Doll's House
            76,     # Adventures of Huckleberry Finn
            74,     # Tom Sawyer
            1400,   # Great Expectations
            46,     # A Christmas Carol
            5200,   # Metamorphosis
            43,     # Jekyll and Hyde
            1080,   # A Modest Proposal
            844,    # The Importance of Being Earnest
            2600,   # War and Peace
            135,    # Les Misérables
            16328,  # Beowulf
            2591,   # Grimm's Fairy Tales
            996,    # Don Quixote
            28054,  # Brothers Karamazov
        ]
        
        # Extend with sequential IDs
        all_book_ids = popular_books + list(range(1, num_books + 1))
        all_book_ids = list(set(all_book_ids))[:num_books]
        
        for book_id in tqdm(all_book_ids, desc="Fetching Gutenberg texts"):
            try:
                # Try different URL formats
                urls = [
                    f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
                    f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
                    f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
                ]
                
                text = None
                for url in urls:
                    try:
                        response = requests.get(url, timeout=30)
                        if response.status_code == 200:
                            text = response.text
                            break
                    except:
                        continue
                
                if not text:
                    continue
                
                # Extract paragraphs
                paragraphs = self._extract_paragraphs(text)
                
                # Sample paragraphs
                sampled = random.sample(
                    paragraphs, 
                    min(len(paragraphs), self.config.gutenberg_paragraphs_per_book)
                )
                
                for i, para in enumerate(sampled):
                    sample = TextSample(
                        text=para,
                        source="gutenberg",
                        source_id=f"gutenberg_{book_id}_{i}",
                        metadata={"book_id": book_id},
                        label=0
                    )
                    self._save_sample(sample)
                
                time.sleep(1 / self.config.requests_per_second)
                
            except Exception as e:
                logger.debug(f"Error fetching book {book_id}: {e}")
                continue
        
        return self.samples_collected
    
    # =========================================================================
    # WIKIPEDIA COLLECTION
    # =========================================================================
    
    def collect_from_wikipedia(self, num_articles: int = 2000) -> int:
        """
        Collect text from Wikipedia articles.
        
        Args:
            num_articles: Number of articles to collect
            
        Returns:
            Number of samples collected
        """
        logger.info(f"Collecting from Wikipedia ({num_articles} articles)...")
        
        if not WIKIPEDIA_AVAILABLE:
            return self._collect_wikipedia_alternative(num_articles)
        
        articles_processed = 0
        categories = self.config.wikipedia_categories
        
        for category in tqdm(categories, desc="Processing Wikipedia categories"):
            if articles_processed >= num_articles:
                break
            
            try:
                # Get articles from category
                cat_page = self.wiki_api.page(f"Category:{category}")
                
                if not cat_page.exists():
                    continue
                
                # Get category members
                members = list(cat_page.categorymembers.values())
                random.shuffle(members)
                
                for page in members[:num_articles // len(categories)]:
                    if articles_processed >= num_articles:
                        break
                    
                    try:
                        if page.ns != 0:  # Skip non-article pages
                            continue
                        
                        # Get article text
                        text = page.text
                        if not text:
                            continue
                        
                        # Extract paragraphs
                        paragraphs = self._extract_paragraphs(text)
                        
                        for i, para in enumerate(paragraphs[:5]):
                            sample = TextSample(
                                text=para,
                                source="wikipedia",
                                source_id=f"wiki_{page.pageid}_{i}",
                                metadata={
                                    "title": page.title,
                                    "category": category,
                                    "pageid": page.pageid
                                },
                                label=0
                            )
                            self._save_sample(sample)
                        
                        articles_processed += 1
                        time.sleep(0.5)
                        
                    except Exception as e:
                        logger.debug(f"Error processing article: {e}")
                        continue
                        
            except Exception as e:
                logger.debug(f"Error processing category {category}: {e}")
                continue
        
        return self.samples_collected
    
    def _collect_wikipedia_alternative(self, num_articles: int) -> int:
        """Alternative Wikipedia collection using API directly."""
        logger.info("Using alternative Wikipedia collection method...")
        
        base_url = "https://en.wikipedia.org/w/api.php"
        articles_processed = 0
        
        # Get random articles
        for _ in tqdm(range(num_articles // 10), desc="Fetching Wikipedia articles"):
            try:
                # Get random articles
                params = {
                    "action": "query",
                    "format": "json",
                    "generator": "random",
                    "grnnamespace": 0,
                    "grnlimit": 10,
                    "prop": "extracts",
                    "exlimit": "max",
                    "explaintext": True,
                    "exsectionformat": "plain"
                }
                
                response = requests.get(base_url, params=params, timeout=30)
                data = response.json()
                
                if "query" not in data or "pages" not in data["query"]:
                    continue
                
                for page_id, page_data in data["query"]["pages"].items():
                    if "extract" not in page_data:
                        continue
                    
                    text = page_data["extract"]
                    paragraphs = self._extract_paragraphs(text)
                    
                    for i, para in enumerate(paragraphs[:3]):
                        sample = TextSample(
                            text=para,
                            source="wikipedia",
                            source_id=f"wiki_{page_id}_{i}",
                            metadata={
                                "title": page_data.get("title", "Unknown"),
                                "pageid": page_id
                            },
                            label=0
                        )
                        self._save_sample(sample)
                    
                    articles_processed += 1
                
                time.sleep(1 / self.config.requests_per_second)
                
            except Exception as e:
                logger.debug(f"Wikipedia API error: {e}")
                continue
        
        return self.samples_collected
    
    # =========================================================================
    # HUGGINGFACE DATASETS COLLECTION
    # =========================================================================
    
    def collect_from_openwebtext(self, num_samples: int = 10000) -> int:
        """
        Collect text from OpenWebText dataset.
        
        Args:
            num_samples: Number of samples to collect
            
        Returns:
            Number of samples collected
        """
        logger.info(f"Collecting from OpenWebText ({num_samples} samples)...")
        
        if not DATASETS_AVAILABLE:
            logger.warning("datasets library not available. Skipping OpenWebText.")
            return self.samples_collected
        
        try:
            # Load OpenWebText dataset
            dataset = load_dataset(
                "openwebtext",
                split="train",
                streaming=True,
                trust_remote_code=True
            )
            
            count = 0
            for item in tqdm(dataset, desc="Processing OpenWebText", total=num_samples):
                if count >= num_samples:
                    break
                
                text = item.get("text", "")
                
                # Extract valid paragraphs
                paragraphs = self._extract_paragraphs(text)
                
                for i, para in enumerate(paragraphs[:2]):
                    sample = TextSample(
                        text=para,
                        source="openwebtext",
                        source_id=f"owt_{count}_{i}",
                        metadata={},
                        label=0
                    )
                    self._save_sample(sample)
                
                count += 1
                
        except Exception as e:
            logger.error(f"Error loading OpenWebText: {e}")
            # Try alternative datasets
            return self._collect_alternative_dataset(num_samples)
        
        return self.samples_collected
    
    def _collect_alternative_dataset(self, num_samples: int) -> int:
        """Collect from alternative HuggingFace datasets."""
        logger.info("Trying alternative datasets...")
        
        alternative_datasets = [
            ("wikitext", "wikitext-103-v1", "train"),
            ("bookcorpus", None, "train"),
            ("cc_news", None, "train"),
            ("ag_news", None, "train"),
        ]
        
        for dataset_name, config_name, split in alternative_datasets:
            try:
                logger.info(f"Trying dataset: {dataset_name}")
                
                if config_name:
                    dataset = load_dataset(
                        dataset_name, 
                        config_name, 
                        split=split,
                        streaming=True,
                        trust_remote_code=True
                    )
                else:
                    dataset = load_dataset(
                        dataset_name,
                        split=split,
                        streaming=True,
                        trust_remote_code=True
                    )
                
                count = 0
                for item in tqdm(dataset, desc=f"Processing {dataset_name}", total=num_samples):
                    if count >= num_samples:
                        break
                    
                    # Get text field (varies by dataset)
                    text = item.get("text", item.get("content", item.get("sentence", "")))
                    
                    if self._is_valid_text(text):
                        sample = TextSample(
                            text=text.strip(),
                            source=dataset_name,
                            source_id=f"{dataset_name}_{count}",
                            metadata={},
                            label=0
                        )
                        self._save_sample(sample)
                        count += 1
                
                if count > 0:
                    logger.info(f"Collected {count} samples from {dataset_name}")
                    break
                    
            except Exception as e:
                logger.debug(f"Error with {dataset_name}: {e}")
                continue
        
        return self.samples_collected
    
    def collect_from_news(self, num_samples: int = 5000) -> int:
        """
        Collect text from news datasets.
        
        Args:
            num_samples: Number of samples to collect
            
        Returns:
            Number of samples collected
        """
        logger.info(f"Collecting from news datasets ({num_samples} samples)...")
        
        if not DATASETS_AVAILABLE:
            logger.warning("datasets library not available. Skipping news.")
            return self.samples_collected
        
        news_datasets = [
            ("cnn_dailymail", "3.0.0", "train", "article"),
            ("multi_news", None, "train", "document"),
            ("xsum", None, "train", "document"),
        ]
        
        samples_per_dataset = num_samples // len(news_datasets)
        
        for dataset_name, config_name, split, text_field in news_datasets:
            try:
                logger.info(f"Loading {dataset_name}...")
                
                if config_name:
                    dataset = load_dataset(
                        dataset_name,
                        config_name,
                        split=split,
                        streaming=True,
                        trust_remote_code=True
                    )
                else:
                    dataset = load_dataset(
                        dataset_name,
                        split=split,
                        streaming=True,
                        trust_remote_code=True
                    )
                
                count = 0
                for item in tqdm(dataset, desc=f"Processing {dataset_name}", total=samples_per_dataset):
                    if count >= samples_per_dataset:
                        break
                    
                    text = item.get(text_field, "")
                    paragraphs = self._extract_paragraphs(text)
                    
                    for i, para in enumerate(paragraphs[:2]):
                        sample = TextSample(
                            text=para,
                            source=dataset_name,
                            source_id=f"{dataset_name}_{count}_{i}",
                            metadata={},
                            label=0
                        )
                        self._save_sample(sample)
                    
                    count += 1
                    
            except Exception as e:
                logger.debug(f"Error with {dataset_name}: {e}")
                continue
        
        return self.samples_collected
    
    # =========================================================================
    # MAIN COLLECTION METHOD
    # =========================================================================
    
    def collect_all(self, target_samples: Optional[int] = None) -> int:
        """
        Collect human text from all sources.
        
        Args:
            target_samples: Target number of samples to collect
            
        Returns:
            Total number of samples collected
        """
        target = target_samples or self.config.human_samples
        logger.info(f"Starting human text collection. Target: {target} samples")
        
        # Clear output file if it exists
        if self.output_path.exists():
            self.output_path.unlink()
        
        # Calculate samples per source
        sources_count = 4  # Gutenberg, Wikipedia, OpenWebText, News
        samples_per_source = target // sources_count
        
        # Collect from each source
        self.collect_from_gutenberg(num_books=samples_per_source // 10)
        self.collect_from_wikipedia(num_articles=samples_per_source)
        self.collect_from_openwebtext(num_samples=samples_per_source)
        self.collect_from_news(num_samples=samples_per_source)
        
        logger.info(f"Human text collection complete. Total samples: {self.samples_collected}")
        return self.samples_collected


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for human text collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect human-written text from various sources")
    parser.add_argument("--target", type=int, default=50000, help="Target number of samples")
    parser.add_argument("--source", type=str, choices=["all", "gutenberg", "wikipedia", "openwebtext", "news"],
                       default="all", help="Source to collect from")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else None
    collector = HumanTextCollector(output_path=output_path)
    
    if args.source == "all":
        collector.collect_all(target_samples=args.target)
    elif args.source == "gutenberg":
        collector.collect_from_gutenberg(num_books=args.target // 50)
    elif args.source == "wikipedia":
        collector.collect_from_wikipedia(num_articles=args.target)
    elif args.source == "openwebtext":
        collector.collect_from_openwebtext(num_samples=args.target)
    elif args.source == "news":
        collector.collect_from_news(num_samples=args.target)
    
    print(f"\nCollection complete! Total samples: {collector.samples_collected}")
    print(f"Output saved to: {collector.output_path}")


if __name__ == "__main__":
    main()