"""
Prepare training data for the Humanizer model.
Creates AI-to-Human paraphrase pairs for training.
"""

import os
import sys
import json
import random
import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


class HumanizerDataPreparer:
    """
    Prepares training data for the humanizer.
    Creates pairs of (AI text, Human text) for paraphrase training.
    """
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.data_dir = self.config.DATA_DIR
        self.output_dir = os.path.join(self.data_dir, "humanizer")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # AI patterns to transform
        self.ai_phrases_to_remove = [
            ("it is important to note that", ["note that", "keep in mind", "remember,"]),
            ("it is worth noting that", ["notably,", "interestingly,", ""]),
            ("it is essential to", ["you need to", "we should", "it helps to"]),
            ("it is crucial to", ["you gotta", "we need to", "it's key to"]),
            ("furthermore,", ["also,", "plus,", "and"]),
            ("moreover,", ["also,", "on top of that,", "and"]),
            ("additionally,", ["also,", "plus,", "and"]),
            ("consequently,", ["so,", "because of this,", "that's why"]),
            ("nevertheless,", ["still,", "but,", "even so,"]),
            ("nonetheless,", ["still,", "but,", "yet,"]),
            ("therefore,", ["so,", "that's why", "which means"]),
            ("thus,", ["so,", "this means", ""]),
            ("hence,", ["so,", "that's why", ""]),
            ("in conclusion,", ["so basically,", "to wrap up,", "anyway,"]),
            ("to summarize,", ["basically,", "long story short,", "so yeah,"]),
            ("in today's world,", ["nowadays,", "these days,", ""]),
            ("in today's society,", ["nowadays,", "today,", ""]),
            ("in the modern era,", ["now,", "today,", "these days,"]),
            ("plays a crucial role", ["is really important", "matters a lot", "is key"]),
            ("plays a vital role", ["is super important", "really matters", "is crucial"]),
            ("cannot be overstated", ["is huge", "is really important", "matters so much"]),
            ("a wide range of", ["lots of", "many", "all kinds of"]),
            ("a variety of", ["different", "various", "all sorts of"]),
            ("a plethora of", ["tons of", "lots of", "so many"]),
            ("a myriad of", ["tons of", "countless", "so many"]),
            ("utilize", ["use"]),
            ("implement", ["use", "do", "try"]),
            ("facilitate", ["help", "make easier", "enable"]),
            ("comprehensive", ["complete", "full", "thorough"]),
            ("subsequently", ["then", "after that", "later"]),
            ("prior to", ["before"]),
            ("in order to", ["to"]),
            ("due to the fact that", ["because", "since"]),
            ("for the purpose of", ["to", "for"]),
            ("in light of", ["because of", "given", "considering"]),
            ("with regard to", ["about", "regarding", "on"]),
            ("in terms of", ["for", "regarding", "when it comes to"]),
            ("it can be argued that", ["some say", "you could say", "maybe"]),
            ("research suggests that", ["studies show", "apparently,", ""]),
            ("studies have shown that", ["research shows", "it turns out", ""]),
            ("—", ["-", ",", " - "]),  # Em dash!
        ]
        
        # Casual expressions to add
        self.casual_insertions = [
            "honestly,", "basically,", "like,", "you know,", "I mean,",
            "actually,", "pretty much", "kind of", "sort of", "really",
            "just", "so", "anyway,", "well,", "I think", "probably",
            "maybe", "definitely", "obviously", "clearly", "tbh",
        ]
        
        # Contractions to use
        self.contractions = {
            "it is": "it's",
            "that is": "that's",
            "there is": "there's",
            "what is": "what's",
            "who is": "who's",
            "how is": "how's",
            "he is": "he's",
            "she is": "she's",
            "it has": "it's",
            "that has": "that's",
            "who has": "who's",
            "I am": "I'm",
            "you are": "you're",
            "we are": "we're",
            "they are": "they're",
            "I have": "I've",
            "you have": "you've",
            "we have": "we've",
            "they have": "they've",
            "I will": "I'll",
            "you will": "you'll",
            "we will": "we'll",
            "they will": "they'll",
            "I would": "I'd",
            "you would": "you'd",
            "we would": "we'd",
            "they would": "they'd",
            "do not": "don't",
            "does not": "doesn't",
            "did not": "didn't",
            "will not": "won't",
            "would not": "wouldn't",
            "can not": "can't",
            "cannot": "can't",
            "could not": "couldn't",
            "should not": "shouldn't",
            "is not": "isn't",
            "are not": "aren't",
            "was not": "wasn't",
            "were not": "weren't",
            "have not": "haven't",
            "has not": "hasn't",
            "had not": "hadn't",
            "let us": "let's",
        }
        
        print("="*60)
        print("HUMANIZER DATA PREPARER")
        print("="*60)
    
    def prepare_data(self, target_pairs=50000):
        """Main data preparation function."""
        
        print(f"\n📚 Preparing {target_pairs} training pairs...")
        
        pairs = []
        
        # Method 1: Load existing AI texts and humanize them
        print("\n1. Creating pairs from AI texts...")
        ai_pairs = self._create_pairs_from_ai_texts(target_pairs // 2)
        pairs.extend(ai_pairs)
        print(f"   Created {len(ai_pairs)} pairs from AI texts")
        
        # Method 2: Create synthetic pairs with templates
        print("\n2. Creating synthetic pairs...")
        synthetic_pairs = self._create_synthetic_pairs(target_pairs // 3)
        pairs.extend(synthetic_pairs)
        print(f"   Created {len(synthetic_pairs)} synthetic pairs")
        
        # Method 3: Match AI and human texts by topic/length
        print("\n3. Matching AI and human texts...")
        matched_pairs = self._match_ai_human_texts(target_pairs // 4)
        pairs.extend(matched_pairs)
        print(f"   Created {len(matched_pairs)} matched pairs")
        
        # Shuffle and deduplicate
        random.shuffle(pairs)
        pairs = self._deduplicate_pairs(pairs)
        
        print(f"\n✅ Total pairs: {len(pairs)}")
        
        # Split into train/val/test
        n = len(pairs)
        train_end = int(n * 0.85)
        val_end = int(n * 0.95)
        
        train_pairs = pairs[:train_end]
        val_pairs = pairs[train_end:val_end]
        test_pairs = pairs[val_end:]
        
        # Save
        self._save_pairs(train_pairs, "train_pairs.jsonl")
        self._save_pairs(val_pairs, "val_pairs.jsonl")
        self._save_pairs(test_pairs, "test_pairs.jsonl")
        
        print(f"\n📁 Saved to {self.output_dir}:")
        print(f"   Train: {len(train_pairs)} pairs")
        print(f"   Val:   {len(val_pairs)} pairs")
        print(f"   Test:  {len(test_pairs)} pairs")
        
        return pairs
    
    def _create_pairs_from_ai_texts(self, n_pairs):
        """Create pairs by humanizing AI texts."""
        pairs = []
        
        # Load AI texts from detector training data
        ai_file = os.path.join(self.config.RAW_DATA_DIR, "ai_texts_enhanced.jsonl")
        
        if not os.path.exists(ai_file):
            # Try alternative location
            ai_file = os.path.join(self.data_dir, "train.jsonl")
        
        if not os.path.exists(ai_file):
            print(f"   ⚠️ No AI texts found at {ai_file}")
            return pairs
        
        ai_texts = []
        with open(ai_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    if sample.get('label', 0) == 1:  # AI text
                        ai_texts.append(sample['text'])
        
        print(f"   Loaded {len(ai_texts)} AI texts")
        
        for text in tqdm(ai_texts[:n_pairs], desc="Humanizing"):
            humanized = self._humanize_text(text)
            
            if humanized and humanized != text:
                pairs.append({
                    'source': text,
                    'target': humanized,
                    'type': 'humanized'
                })
        
        return pairs
    
    def _humanize_text(self, text):
        """Apply humanization transformations to text."""
        result = text
        
        # 1. Replace AI phrases
        for ai_phrase, replacements in self.ai_phrases_to_remove:
            if ai_phrase.lower() in result.lower():
                replacement = random.choice(replacements)
                # Case-insensitive replacement
                pattern = re.compile(re.escape(ai_phrase), re.IGNORECASE)
                result = pattern.sub(replacement, result)
        
        # 2. Apply contractions
        for full, contracted in self.contractions.items():
            pattern = re.compile(r'\b' + re.escape(full) + r'\b', re.IGNORECASE)
            if random.random() > 0.3:  # 70% chance to contract
                result = pattern.sub(contracted, result)
        
        # 3. Add casual insertions occasionally
        sentences = result.split('. ')
        new_sentences = []
        
        for i, sent in enumerate(sentences):
            if len(sent.split()) > 5 and random.random() > 0.7:
                # Add casual word at start
                casual = random.choice(self.casual_insertions)
                if sent and sent[0].isupper():
                    sent = casual.capitalize() + " " + sent[0].lower() + sent[1:]
                else:
                    sent = casual + " " + sent
            new_sentences.append(sent)
        
        result = '. '.join(new_sentences)
        
        # 4. Occasionally add questions or exclamations
        if random.random() > 0.8:
            additions = [
                " Right?",
                " You know?",
                "!",
                " Makes sense?",
            ]
            if result.endswith('.'):
                result = result[:-1] + random.choice(additions)
        
        # 5. Clean up
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\s+([.,!?])', r'\1', result)
        result = result.strip()
        
        return result
    
    def _create_synthetic_pairs(self, n_pairs):
        """Create synthetic AI-human pairs with templates."""
        pairs = []
        
        # AI-style templates and their human equivalents
        template_pairs = [
            # Education
            (
                "Education plays a crucial role in shaping the future of society. It is essential to understand that quality education enables individuals to develop critical thinking skills and contribute meaningfully to their communities. Furthermore, research has demonstrated that educational attainment is strongly correlated with economic prosperity.",
                "Education is super important for society's future. Good education helps people think critically and give back to their communities. Studies show that the more education you have, the better you tend to do financially."
            ),
            (
                "The implementation of technology in educational settings has transformed the learning experience. It is worth noting that digital tools facilitate personalized learning approaches. Additionally, online platforms have democratized access to educational resources.",
                "Tech in schools has really changed how we learn. Digital tools make it easier to learn at your own pace. Plus, online stuff means anyone can access learning materials now."
            ),
            # Technology
            (
                "Artificial intelligence represents a paradigm shift in technological advancement. The implications of AI extend far beyond automation, encompassing areas such as healthcare, finance, and creative industries. It is imperative that society adapts to these changes.",
                "AI is a huge change in tech. It's not just about automating stuff - it's affecting healthcare, money stuff, and even creative work. We really need to adapt to all this."
            ),
            (
                "The proliferation of social media platforms has fundamentally altered human communication patterns. These platforms facilitate instantaneous global connectivity. However, concerns regarding privacy and misinformation have emerged.",
                "Social media has totally changed how we talk to each other. You can connect with anyone anywhere instantly. But there are real worries about privacy and fake news."
            ),
            # Health
            (
                "Maintaining optimal health requires a comprehensive approach that encompasses physical activity, nutrition, and mental well-being. Research indicates that regular exercise contributes to longevity. Furthermore, balanced dietary habits are essential for disease prevention.",
                "Staying healthy means taking care of your body and mind. Working out regularly helps you live longer. And eating well helps prevent diseases - pretty simple really."
            ),
            (
                "Mental health awareness has gained significant attention in contemporary society. It is crucial to recognize that psychological well-being is equally important as physical health. Support systems and professional interventions can facilitate recovery.",
                "Mental health is finally getting the attention it deserves. Your mental health matters just as much as your physical health. Having support and getting professional help when you need it really makes a difference."
            ),
            # Environment
            (
                "Climate change represents one of the most pressing challenges facing humanity. Scientific evidence indicates that anthropogenic activities are the primary drivers of global warming. It is imperative that immediate action is taken to mitigate these effects.",
                "Climate change is probably the biggest problem we're facing right now. Science shows that humans are causing global warming. We really need to do something about it ASAP."
            ),
            (
                "Sustainable development necessitates a balanced approach to economic growth and environmental conservation. The implementation of renewable energy sources is essential for reducing carbon emissions. Additionally, individual actions contribute to collective impact.",
                "We need to balance making money with protecting the planet. Using renewable energy is key to cutting carbon emissions. And hey, what each of us does actually adds up!"
            ),
            # Work
            (
                "The modern workplace has undergone significant transformation in recent years. Remote work arrangements have become increasingly prevalent. Furthermore, organizations are recognizing the importance of work-life balance for employee well-being.",
                "Work has changed a lot lately. Remote work is everywhere now. Companies are finally realizing that work-life balance matters for keeping employees happy."
            ),
            (
                "Professional development is essential for career advancement in today's competitive job market. Continuous learning enables individuals to adapt to evolving industry requirements. Additionally, networking facilitates access to new opportunities.",
                "If you want to get ahead at work, you gotta keep learning. Things change fast and you need to keep up. And networking? That's how you find new opportunities."
            ),
        ]
        
        # Generate pairs from templates
        for _ in range(n_pairs // len(template_pairs) + 1):
            for ai_text, human_text in template_pairs:
                # Add some variation
                ai_varied = self._add_variation(ai_text, is_ai=True)
                human_varied = self._add_variation(human_text, is_ai=False)
                
                pairs.append({
                    'source': ai_varied,
                    'target': human_varied,
                    'type': 'synthetic'
                })
                
                if len(pairs) >= n_pairs:
                    break
            
            if len(pairs) >= n_pairs:
                break
        
        return pairs[:n_pairs]
    
    def _add_variation(self, text, is_ai=True):
        """Add random variation to text."""
        # Simple word replacements for variation
        if is_ai:
            variations = {
                'crucial': ['vital', 'essential', 'critical', 'important'],
                'significant': ['substantial', 'considerable', 'notable'],
                'demonstrates': ['shows', 'indicates', 'reveals'],
                'facilitates': ['enables', 'allows', 'permits'],
                'comprehensive': ['thorough', 'complete', 'extensive'],
            }
        else:
            variations = {
                'really': ['super', 'pretty', 'very', 'so'],
                'important': ['big deal', 'key', 'crucial', 'huge'],
                'stuff': ['things', 'items', 'bits'],
                'gonna': ['going to', 'about to'],
            }
        
        result = text
        for word, replacements in variations.items():
            if word in result.lower() and random.random() > 0.5:
                replacement = random.choice(replacements)
                result = re.sub(r'\b' + word + r'\b', replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def _match_ai_human_texts(self, n_pairs):
        """Match AI texts with human texts by similar length/topic."""
        pairs = []
        
        # Load both AI and human texts
        train_file = os.path.join(self.data_dir, "train.jsonl")
        
        if not os.path.exists(train_file):
            return pairs
        
        ai_texts = []
        human_texts = []
        
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    if sample['label'] == 1:
                        ai_texts.append(sample['text'])
                    else:
                        human_texts.append(sample['text'])
        
        if not ai_texts or not human_texts:
            return pairs
        
        # Sort by length for matching
        ai_texts = sorted(ai_texts, key=len)
        human_texts = sorted(human_texts, key=len)
        
        # Match by similar length
        for ai_text in tqdm(ai_texts[:n_pairs], desc="Matching"):
            ai_len = len(ai_text)
            
            # Find human text of similar length
            best_match = None
            best_diff = float('inf')
            
            for human_text in random.sample(human_texts, min(100, len(human_texts))):
                diff = abs(len(human_text) - ai_len)
                if diff < best_diff:
                    best_diff = diff
                    best_match = human_text
            
            if best_match and best_diff < ai_len * 0.5:  # Within 50% length
                pairs.append({
                    'source': ai_text,
                    'target': best_match,
                    'type': 'matched'
                })
        
        return pairs
    
    def _deduplicate_pairs(self, pairs):
        """Remove duplicate pairs."""
        seen = set()
        unique = []
        
        for pair in pairs:
            key = (pair['source'][:100], pair['target'][:100])
            if key not in seen:
                seen.add(key)
                unique.append(pair)
        
        return unique
    
    def _save_pairs(self, pairs, filename):
        """Save pairs to JSONL file."""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare Humanizer Data')
    parser.add_argument('--pairs', type=int, default=50000, help='Target number of pairs')
    args = parser.parse_args()
    
    preparer = HumanizerDataPreparer()
    preparer.prepare_data(target_pairs=args.pairs)


if __name__ == "__main__":
    main()