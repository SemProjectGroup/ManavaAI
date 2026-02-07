"""
Enhanced Data Collection Script for ManavAI
Collects diverse human text and generates AI text with modern patterns.
Optimized for training a highly accurate detector.
"""

import os
import sys
import json
import random
import time
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


class EnhancedDataCollector:
    """
    Collects high-quality training data:
    - Human text from multiple sources
    - AI text generated with various prompts and styles
    """
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.output_dir = self.config.RAW_DATA_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.collected_hashes = set()  # For deduplication
        
        print("="*60)
        print("ENHANCED DATA COLLECTOR")
        print("="*60)
    
    def collect_human_text(self, target_samples=50000):
        """Collect human-written text from multiple sources."""
        
        print(f"\n📚 Collecting {target_samples} human text samples...")
        
        samples = []
        
        # Source 1: OpenWebText (Reddit submissions - very human-like)
        print("\n1. Loading OpenWebText...")
        samples.extend(self._collect_openwebtext(target_samples // 3))
        
        # Source 2: Wikipedia (factual but human-written)
        print("\n2. Loading Wikipedia...")
        samples.extend(self._collect_wikipedia(target_samples // 4))
        
        # Source 3: Reddit comments (very casual human text)
        print("\n3. Loading Reddit data...")
        samples.extend(self._collect_reddit(target_samples // 4))
        
        # Source 4: Blog posts / Personal narratives
        print("\n4. Loading blog/narrative data...")
        samples.extend(self._collect_blogs(target_samples // 6))
        
        # Source 5: Books (Project Gutenberg - classic human writing)
        print("\n5. Loading book excerpts...")
        samples.extend(self._collect_gutenberg(target_samples // 6))
        
        # Shuffle and deduplicate
        random.shuffle(samples)
        samples = self._deduplicate(samples)
        
        # Save
        output_file = os.path.join(self.output_dir, "human_texts_enhanced.jsonl")
        self._save_samples(samples[:target_samples], output_file, label=0)
        
        print(f"\n✅ Collected {len(samples[:target_samples])} human samples")
        return samples[:target_samples]
    
    def _collect_openwebtext(self, n_samples):
        """Collect from OpenWebText dataset."""
        samples = []
        
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("openwebtext", split="train", streaming=True)
            
            for i, item in enumerate(tqdm(dataset, total=n_samples, desc="OpenWebText")):
                if i >= n_samples * 2:  # Get extra for filtering
                    break
                
                text = item.get('text', '')
                if self._is_valid_human_text(text):
                    # Extract a good chunk
                    text = self._extract_chunk(text, 100, 500)
                    if text:
                        samples.append({
                            'text': text,
                            'source': 'openwebtext',
                            'label': 0
                        })
                
                if len(samples) >= n_samples:
                    break
                    
        except Exception as e:
            print(f"   ⚠️ OpenWebText error: {e}")
        
        return samples
    
    def _collect_wikipedia(self, n_samples):
        """Collect from Wikipedia."""
        samples = []
        
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
            
            for i, item in enumerate(tqdm(dataset, total=n_samples, desc="Wikipedia")):
                if i >= n_samples * 2:
                    break
                
                text = item.get('text', '')
                if len(text.split()) > 100:
                    text = self._extract_chunk(text, 100, 400)
                    if text and self._is_valid_human_text(text):
                        samples.append({
                            'text': text,
                            'source': 'wikipedia',
                            'label': 0
                        })
                
                if len(samples) >= n_samples:
                    break
                    
        except Exception as e:
            print(f"   ⚠️ Wikipedia error: {e}")
        
        return samples
    
    def _collect_reddit(self, n_samples):
        """Collect Reddit-style casual text."""
        samples = []
        
        try:
            from datasets import load_dataset
            
            # Try multiple Reddit datasets
            reddit_datasets = [
                ("eli5", "train", "answers.text"),
                ("reddit_tifu", "train", "documents"),
            ]
            
            for ds_name, split, field in reddit_datasets:
                try:
                    dataset = load_dataset(ds_name, split=split, streaming=True)
                    
                    for i, item in enumerate(tqdm(dataset, total=n_samples//2, desc=f"Reddit/{ds_name}")):
                        if i >= n_samples:
                            break
                        
                        # Handle different field structures
                        if '.' in field:
                            parts = field.split('.')
                            text = item
                            for part in parts:
                                if isinstance(text, dict):
                                    text = text.get(part, '')
                                elif isinstance(text, list) and text:
                                    text = text[0].get(part, '') if isinstance(text[0], dict) else str(text[0])
                                else:
                                    text = ''
                                    break
                        else:
                            text = item.get(field, '')
                        
                        if isinstance(text, list):
                            text = ' '.join(str(t) for t in text)
                        
                        text = str(text)
                        
                        if len(text.split()) > 50 and self._is_valid_human_text(text):
                            text = self._extract_chunk(text, 50, 400)
                            if text:
                                samples.append({
                                    'text': text,
                                    'source': f'reddit_{ds_name}',
                                    'label': 0
                                })
                        
                        if len(samples) >= n_samples:
                            break
                            
                except Exception as e:
                    print(f"   ⚠️ {ds_name} error: {e}")
                    continue
                    
        except Exception as e:
            print(f"   ⚠️ Reddit collection error: {e}")
        
        return samples
    
    def _collect_blogs(self, n_samples):
        """Collect blog-style personal narratives."""
        samples = []
        
        try:
            from datasets import load_dataset
            
            # Blog authorship corpus
            try:
                dataset = load_dataset("blog_authorship_corpus", split="train", streaming=True)
                
                for i, item in enumerate(tqdm(dataset, total=n_samples, desc="Blogs")):
                    if i >= n_samples * 2:
                        break
                    
                    text = item.get('text', '')
                    if len(text.split()) > 80 and self._is_valid_human_text(text):
                        text = self._extract_chunk(text, 80, 400)
                        if text:
                            samples.append({
                                'text': text,
                                'source': 'blogs',
                                'label': 0
                            })
                    
                    if len(samples) >= n_samples:
                        break
                        
            except Exception as e:
                print(f"   ⚠️ Blog corpus error: {e}")
                
        except Exception as e:
            print(f"   ⚠️ Blog collection error: {e}")
        
        return samples
    
    def _collect_gutenberg(self, n_samples):
        """Collect from Project Gutenberg books."""
        samples = []
        
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("pg19", split="train", streaming=True)
            
            for i, item in enumerate(tqdm(dataset, total=n_samples//10, desc="Gutenberg")):
                if i >= n_samples // 5:
                    break
                
                text = item.get('text', '')
                
                # Get multiple chunks from each book
                chunks = self._extract_multiple_chunks(text, 100, 400, 10)
                for chunk in chunks:
                    if self._is_valid_human_text(chunk):
                        samples.append({
                            'text': chunk,
                            'source': 'gutenberg',
                            'label': 0
                        })
                    
                    if len(samples) >= n_samples:
                        break
                
                if len(samples) >= n_samples:
                    break
                    
        except Exception as e:
            print(f"   ⚠️ Gutenberg error: {e}")
        
        return samples
    
    def _is_valid_human_text(self, text):
        """Check if text is valid for training."""
        if not text or not isinstance(text, str):
            return False
        
        words = text.split()
        if len(words) < 50:
            return False
        
        # Filter out non-English or garbage
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
        if ascii_ratio < 0.85:
            return False
        
        # Filter out code/scripts
        code_indicators = ['def ', 'function ', 'var ', 'const ', '<?php', '<html', 'import ', 'from ', '#!/']
        if any(ind in text.lower() for ind in code_indicators):
            return False
        
        # Filter out tables/lists only
        if text.count('|') > 10 or text.count('\t') > 10:
            return False
        
        return True
    
    def _extract_chunk(self, text, min_words, max_words):
        """Extract a clean chunk from text."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) < 3:
            return None
        
        # Find a good starting point
        start_idx = random.randint(0, max(0, len(sentences) - 10))
        
        chunk_sentences = []
        word_count = 0
        
        for sent in sentences[start_idx:]:
            sent = sent.strip()
            if not sent:
                continue
            
            sent_words = len(sent.split())
            if word_count + sent_words > max_words:
                break
            
            chunk_sentences.append(sent)
            word_count += sent_words
            
            if word_count >= min_words and random.random() > 0.7:
                break
        
        if word_count < min_words:
            return None
        
        return ' '.join(chunk_sentences)
    
    def _extract_multiple_chunks(self, text, min_words, max_words, n_chunks):
        """Extract multiple non-overlapping chunks."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        
        i = 0
        while i < len(sentences) and len(chunks) < n_chunks:
            chunk_sentences = []
            word_count = 0
            
            while i < len(sentences) and word_count < max_words:
                sent = sentences[i].strip()
                i += 1
                
                if not sent:
                    continue
                
                chunk_sentences.append(sent)
                word_count += len(sent.split())
                
                if word_count >= min_words and random.random() > 0.7:
                    break
            
            if word_count >= min_words:
                chunk = ' '.join(chunk_sentences)
                if self._is_valid_human_text(chunk):
                    chunks.append(chunk)
            
            # Skip some sentences to get diverse chunks
            i += random.randint(5, 20)
        
        return chunks
    
    def _deduplicate(self, samples):
        """Remove duplicate samples."""
        unique = []
        
        for sample in samples:
            text = sample['text']
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            if text_hash not in self.collected_hashes:
                self.collected_hashes.add(text_hash)
                unique.append(sample)
        
        return unique
    
    def _save_samples(self, samples, output_file, label):
        """Save samples to JSONL file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                sample['label'] = label
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"   Saved to: {output_file}")
    
    def generate_ai_text(self, target_samples=50000):
        """Generate AI text using various models and prompts."""
        
        print(f"\n🤖 Generating {target_samples} AI text samples...")
        
        samples = []
        
        # Models to use for generation
        models = [
            ("gpt2", target_samples // 4),
            ("gpt2-medium", target_samples // 4),
            ("gpt2-large", target_samples // 6),
            ("EleutherAI/gpt-neo-125M", target_samples // 6),
            ("EleutherAI/gpt-neo-1.3B", target_samples // 6),
        ]
        
        for model_name, n_samples in models:
            print(f"\n   Generating with {model_name}...")
            model_samples = self._generate_with_model(model_name, n_samples)
            samples.extend(model_samples)
            print(f"   Generated {len(model_samples)} samples")
        
        # Add synthetic AI patterns
        print("\n   Adding synthetic AI-style samples...")
        synthetic = self._generate_synthetic_ai_samples(target_samples // 5)
        samples.extend(synthetic)
        
        # Shuffle and save
        random.shuffle(samples)
        samples = self._deduplicate(samples)
        
        output_file = os.path.join(self.output_dir, "ai_texts_enhanced.jsonl")
        self._save_samples(samples[:target_samples], output_file, label=1)
        
        print(f"\n✅ Generated {len(samples[:target_samples])} AI samples")
        return samples[:target_samples]
    
    def _generate_with_model(self, model_name, n_samples):
        """Generate samples using a specific model."""
        samples = []
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            print(f"      Loading {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Diverse prompts for generation
            prompts = self._get_diverse_prompts()
            
            batch_size = 4
            max_length = 300
            
            for i in tqdm(range(0, n_samples, batch_size), desc=f"Generating"):
                batch_prompts = [random.choice(prompts) for _ in range(batch_size)]
                
                try:
                    inputs = tokenizer(
                        batch_prompts,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=50
                    ).to(device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_length=max_length,
                            num_return_sequences=1,
                            do_sample=True,
                            temperature=random.uniform(0.7, 1.0),
                            top_p=random.uniform(0.85, 0.95),
                            top_k=50,
                            repetition_penalty=1.1,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    
                    for output in outputs:
                        text = tokenizer.decode(output, skip_special_tokens=True)
                        text = self._clean_generated_text(text)
                        
                        if len(text.split()) >= 80:
                            samples.append({
                                'text': text,
                                'source': model_name,
                                'label': 1
                            })
                            
                except Exception as e:
                    continue
                
                if len(samples) >= n_samples:
                    break
            
            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"      ⚠️ Error with {model_name}: {e}")
        
        return samples
    
    def _get_diverse_prompts(self):
        """Get diverse prompts for AI text generation."""
        return [
            # Essay-style prompts
            "The importance of education in modern society cannot be",
            "Technology has transformed the way we live and work in",
            "Climate change is one of the most pressing issues facing",
            "The role of social media in contemporary communication has",
            "Healthcare systems around the world are facing significant",
            "Economic development is closely linked to environmental",
            "The impact of artificial intelligence on employment is",
            "Education plays a crucial role in shaping the future of",
            "The relationship between technology and privacy has become",
            "Sustainable development requires a balanced approach to",
            
            # Explanatory prompts
            "There are several key factors that contribute to",
            "It is important to understand that",
            "Research has shown that the primary causes of",
            "The main advantages of this approach include",
            "When considering the implications of",
            "Studies have demonstrated that",
            "The significance of this development lies in",
            "Experts agree that the best way to",
            "One of the most important aspects of",
            "The evidence suggests that",
            
            # Analytical prompts
            "In analyzing the current situation, it becomes clear that",
            "A comprehensive examination of the data reveals",
            "When we consider the various perspectives on",
            "The historical context of this issue shows",
            "From an economic standpoint, the implications are",
            "The scientific consensus indicates that",
            "Looking at the broader picture, we can see",
            "The fundamental principles underlying this concept",
            "A critical analysis of the evidence suggests",
            "The long-term consequences of this trend",
        ]
    
    def _clean_generated_text(self, text):
        """Clean up generated text."""
        # Remove the prompt if it appears
        lines = text.split('\n')
        if len(lines) > 1:
            # Skip first line if it's just the prompt
            text = '\n'.join(lines)
        
        # Clean up
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove incomplete sentences at the end
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if sentences and not sentences[-1].endswith(('.', '!', '?')):
            sentences = sentences[:-1]
        
        return ' '.join(sentences)
    
    def _generate_synthetic_ai_samples(self, n_samples):
        """Generate synthetic samples with strong AI patterns."""
        
        samples = []
        
        # Templates with AI patterns
        templates = self._get_ai_templates()
        
        for i in range(n_samples):
            template = random.choice(templates)
            text = self._fill_template(template)
            
            if text and len(text.split()) >= 80:
                samples.append({
                    'text': text,
                    'source': 'synthetic_ai',
                    'label': 1
                })
        
        return samples
    
    def _get_ai_templates(self):
        """Get templates with strong AI patterns."""
        return [
            """The importance of {topic} in {context} cannot be overstated. It is essential to understand that {topic} plays a crucial role in shaping {outcome}. Furthermore, research has consistently demonstrated that {benefit1}. Additionally, experts agree that {benefit2}. In conclusion, it is imperative that we recognize the significance of {topic} and take appropriate action to {action}.""",
            
            """In today's rapidly evolving world, {topic} has become increasingly significant. There are several key factors that contribute to this phenomenon. Firstly, {reason1}. Secondly, {reason2}. Moreover, studies have shown that {evidence}. It is worth noting that {observation}. Ultimately, the impact of {topic} on {area} represents a paradigm shift in how we approach {challenge}.""",
            
            """The relationship between {topic1} and {topic2} has garnered significant attention in recent years. It is important to note that these two concepts are intrinsically linked. Furthermore, the implications of this connection extend far beyond {scope}. Research suggests that {finding}. Consequently, it is crucial to develop comprehensive strategies that address both {topic1} and {topic2} simultaneously.""",
            
            """{Topic} represents one of the most pressing challenges facing {group} in the 21st century. The significance of this issue cannot be understated. There are numerous factors that contribute to {problem}. Additionally, the consequences of inaction could be severe. It is therefore essential that stakeholders work collaboratively to implement effective solutions. By doing so, we can ensure {positive_outcome} for future generations.""",
            
            """When examining the impact of {topic} on {area}, several key observations emerge. First and foremost, {observation1}. Moreover, it has become increasingly apparent that {observation2}. The evidence clearly indicates that {conclusion}. As such, it is paramount that decision-makers take these findings into consideration when formulating policies related to {topic}.""",
        ]
    
    def _fill_template(self, template):
        """Fill template with random content."""
        
        topics = ['education', 'technology', 'healthcare', 'sustainability', 'innovation',
                 'communication', 'globalization', 'digitalization', 'urbanization', 'automation']
        
        contexts = ['modern society', 'contemporary organizations', 'educational institutions',
                   'healthcare systems', 'business environments', 'global markets']
        
        outcomes = ['societal progress', 'economic growth', 'sustainable development',
                   'technological advancement', 'social well-being', 'organizational success']
        
        benefits = ['improved efficiency and productivity', 'enhanced quality of life',
                   'greater accessibility and inclusivity', 'reduced environmental impact',
                   'increased innovation and creativity', 'better health outcomes']
        
        actions = ['embrace change', 'invest in resources', 'develop new strategies',
                  'foster collaboration', 'implement best practices', 'prioritize sustainability']
        
        # Simple replacement
        text = template
        text = text.replace('{topic}', random.choice(topics))
        text = text.replace('{Topic}', random.choice(topics).capitalize())
        text = text.replace('{topic1}', random.choice(topics))
        text = text.replace('{topic2}', random.choice(topics))
        text = text.replace('{context}', random.choice(contexts))
        text = text.replace('{outcome}', random.choice(outcomes))
        text = text.replace('{benefit1}', random.choice(benefits))
        text = text.replace('{benefit2}', random.choice(benefits))
        text = text.replace('{action}', random.choice(actions))
        text = text.replace('{area}', random.choice(contexts))
        text = text.replace('{group}', random.choice(['humanity', 'organizations', 'communities', 'nations']))
        text = text.replace('{scope}', random.choice(['initial expectations', 'traditional boundaries', 'conventional understanding']))
        text = text.replace('{finding}', random.choice(benefits))
        text = text.replace('{problem}', random.choice(['this challenge', 'these issues', 'the current situation']))
        text = text.replace('{positive_outcome}', random.choice(outcomes))
        text = text.replace('{reason1}', random.choice(benefits))
        text = text.replace('{reason2}', random.choice(benefits))
        text = text.replace('{evidence}', random.choice(benefits))
        text = text.replace('{observation}', random.choice(benefits))
        text = text.replace('{observation1}', random.choice(benefits))
        text = text.replace('{observation2}', random.choice(benefits))
        text = text.replace('{conclusion}', random.choice(benefits))
        text = text.replace('{challenge}', random.choice(['complex problems', 'modern challenges', 'critical issues']))
        
        return text


def main():
    """Main data collection function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Data Collection')
    parser.add_argument('--human', type=int, default=50000, help='Number of human samples')
    parser.add_argument('--ai', type=int, default=50000, help='Number of AI samples')
    parser.add_argument('--human-only', action='store_true', help='Only collect human text')
    parser.add_argument('--ai-only', action='store_true', help='Only generate AI text')
    args = parser.parse_args()
    
    collector = EnhancedDataCollector()
    
    if not args.ai_only:
        collector.collect_human_text(args.human)
    
    if not args.human_only:
        collector.generate_ai_text(args.ai)
    
    print("\n" + "="*60)
    print("✅ DATA COLLECTION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()