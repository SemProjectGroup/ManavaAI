"""
================================================================================
FIX: Add Short Human Text to Training Data
================================================================================
The problem: Model thinks short, clean sentences are AI.
The fix: Add lots of short human text examples.
================================================================================
"""

import os
import sys
import pickle
import random
from typing import List
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
    from datasets import load_dataset

# ============================================================================
# CONFIG
# ============================================================================

EXISTING_CACHE = r"F:\data\mega_ai_dataset"  # Your current data
OUTPUT_DIR = r"F:\data\fixed_ai_dataset"     # New fixed data
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# COLLECT SHORT HUMAN TEXT
# ============================================================================

def collect_short_human_text() -> List[str]:
    """Collect short, simple human sentences"""
    
    texts = []
    seen = set()
    
    def add_text(text: str):
        text = text.strip()
        if text and text not in seen and 5 <= len(text.split()) <= 30:
            seen.add(text)
            texts.append(text)
    
    print("\n" + "="*60)
    print("COLLECTING SHORT HUMAN TEXT")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # 1. Twitter/Tweets (naturally short)
    # -------------------------------------------------------------------------
    print("\n[1/7] Twitter data...")
    try:
        ds = load_dataset("tweet_eval", "sentiment", split="train")
        for item in tqdm(ds, desc="Tweets"):
            add_text(item.get('text', ''))
        print(f"  ✓ Collected {len(texts):,} tweets")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # -------------------------------------------------------------------------
    # 2. SMS Messages
    # -------------------------------------------------------------------------
    print("\n[2/7] SMS messages...")
    try:
        ds = load_dataset("sms_spam", split="train")
        for item in tqdm(ds, desc="SMS"):
            add_text(item.get('sms', ''))
        print(f"  ✓ Total: {len(texts):,}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # -------------------------------------------------------------------------
    # 3. Short movie reviews
    # -------------------------------------------------------------------------
    print("\n[3/7] Short movie reviews...")
    try:
        ds = load_dataset("rotten_tomatoes", split="train")
        for item in tqdm(ds, desc="Rotten Tomatoes"):
            add_text(item.get('text', ''))
        print(f"  ✓ Total: {len(texts):,}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # -------------------------------------------------------------------------
    # 4. Sentiment sentences (short by nature)
    # -------------------------------------------------------------------------
    print("\n[4/7] Sentiment sentences...")
    try:
        ds = load_dataset("sst2", split="train")
        for item in tqdm(ds, desc="SST2"):
            add_text(item.get('sentence', ''))
        print(f"  ✓ Total: {len(texts):,}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # -------------------------------------------------------------------------
    # 5. Question titles (short questions)
    # -------------------------------------------------------------------------
    print("\n[5/7] Short questions...")
    try:
        ds = load_dataset("yahoo_answers_topics", split="train")
        count = 0
        for item in tqdm(ds, desc="Yahoo Answers"):
            if count >= 50000:
                break
            title = item.get('question_title', '')
            add_text(title)
            count += 1
        print(f"  ✓ Total: {len(texts):,}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # -------------------------------------------------------------------------
    # 6. Headlines (very short)
    # -------------------------------------------------------------------------
    print("\n[6/7] News headlines...")
    try:
        ds = load_dataset("ag_news", split="train")
        for item in tqdm(ds, desc="AG News"):
            # Headlines are usually the first sentence
            text = item.get('text', '').split('.')[0]
            add_text(text)
        print(f"  ✓ Total: {len(texts):,}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # -------------------------------------------------------------------------
    # 7. Generate synthetic short human sentences
    # -------------------------------------------------------------------------
    print("\n[7/7] Generating synthetic short human text...")
    
    synthetic_templates = [
        # Daily activities
        "I went to the {place} and {action}.",
        "Just {action} at the {place}.",
        "The {thing} is really {adjective}.",
        "My {person} called me about {topic}.",
        "I need to {action} tomorrow.",
        "The {thing} got {verb} yesterday.",
        "Going to {place} later today.",
        "Can't believe {thing} happened.",
        "Finally {action} after so long.",
        "The {time} meeting got moved to {day}.",
        
        # Opinions
        "I think {thing} is {adjective}.",
        "Not sure about {thing} honestly.",
        "The {thing} was pretty {adjective}.",
        "{thing} is my favorite.",
        "I really {verb} the {thing}.",
        
        # Short reactions
        "That's so {adjective}!",
        "Wait what happened?",
        "No way that's real.",
        "Sounds good to me.",
        "Makes sense I guess.",
        "Yeah I agree with that.",
        "Not really sure about this.",
        "This is pretty interesting.",
        "I didn't know that.",
        "That explains a lot.",
    ]
    
    places = ["store", "mall", "gym", "office", "park", "school", "hospital", "restaurant", 
              "beach", "library", "bank", "airport", "station", "market", "cafe"]
    actions = ["bought some stuff", "met a friend", "had coffee", "worked out", "grabbed lunch",
               "finished early", "got stuck in traffic", "waited forever", "ran into someone",
               "forgot my wallet", "lost my keys", "found a deal", "saw something weird"]
    things = ["weather", "movie", "food", "meeting", "traffic", "news", "game", "show",
              "book", "song", "app", "update", "project", "deadline", "price"]
    adjectives = ["nice", "weird", "good", "bad", "crazy", "boring", "interesting", 
                  "expensive", "cheap", "different", "same", "okay", "fine", "great"]
    persons = ["mom", "dad", "friend", "boss", "brother", "sister", "coworker", "neighbor"]
    topics = ["the party", "the trip", "work stuff", "the news", "family stuff", "the plan",
              "the weekend", "the meeting", "the project", "dinner plans"]
    verbs = ["cancelled", "delayed", "changed", "moved", "finished", "started", "broke"]
    times = ["morning", "afternoon", "3pm", "Monday", "Friday", "weekly", "team"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "next week"]
    
    for _ in tqdm(range(50000), desc="Synthetic"):
        template = random.choice(synthetic_templates)
        sentence = template.format(
            place=random.choice(places),
            action=random.choice(actions),
            thing=random.choice(things),
            adjective=random.choice(adjectives),
            person=random.choice(persons),
            topic=random.choice(topics),
            verb=random.choice(verbs),
            time=random.choice(times),
            day=random.choice(days)
        )
        add_text(sentence)
    
    print(f"  ✓ Total short human texts: {len(texts):,}")
    
    return texts


def collect_simple_complete_sentences() -> List[str]:
    """Collect grammatically complete but simple human sentences"""
    
    texts = []
    seen = set()
    
    def add_text(text: str):
        text = text.strip()
        # Only complete sentences that end with punctuation
        if (text and text not in seen and 
            text[-1] in '.!?' and 
            5 <= len(text.split()) <= 25):
            seen.add(text)
            texts.append(text)
    
    print("\n" + "="*60)
    print("COLLECTING SIMPLE COMPLETE SENTENCES")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # Common Crawl sentences (real web text)
    # -------------------------------------------------------------------------
    print("\n[1/3] Tatoeba sentences (human translations)...")
    try:
        ds = load_dataset("tatoeba", lang1="en", lang2="fr", split="train")
        count = 0
        for item in tqdm(ds, desc="Tatoeba", total=100000):
            if count >= 100000:
                break
            text = item.get('translation', {}).get('en', '')
            add_text(text)
            count += 1
        print(f"  ✓ Total: {len(texts):,}")
    except Exception as e:
        print(f"  ✗ Tatoeba: {e}")
    
    # -------------------------------------------------------------------------
    # Simple Wikipedia (simpler language)
    # -------------------------------------------------------------------------
    print("\n[2/3] Simple Wikipedia sentences...")
    try:
        ds = load_dataset("wikipedia", "20220301.simple", split="train", streaming=True)
        count = 0
        for item in tqdm(ds, desc="SimpleWiki", total=50000):
            if count >= 50000:
                break
            text = item.get('text', '')
            # Get individual sentences
            for sent in text.split('.'):
                sent = sent.strip() + '.'
                if 5 <= len(sent.split()) <= 20:
                    add_text(sent)
                    count += 1
                    if count >= 50000:
                        break
        print(f"  ✓ Total: {len(texts):,}")
    except Exception as e:
        print(f"  ✗ SimpleWiki: {e}")
    
    # -------------------------------------------------------------------------
    # Common short patterns humans write
    # -------------------------------------------------------------------------
    print("\n[3/3] Common human patterns...")
    
    patterns = [
        # Weather
        "The weather is nice today.",
        "It's raining outside.",
        "The sun is really bright.",
        "It's getting cold.",
        "Perfect weather for a walk.",
        
        # Daily life
        "I went to the store.",
        "I bought some groceries.",
        "I made dinner.",
        "I'm going to sleep now.",
        "I woke up late.",
        "I need to do laundry.",
        "I have a meeting tomorrow.",
        "I finished my homework.",
        "I'm watching a movie.",
        "I'm reading a book.",
        
        # Communication
        "I called my mom.",
        "My friend texted me.",
        "I got an email.",
        "I sent the message.",
        "I'll call you later.",
        "Talk to you soon.",
        "Let me know what you think.",
        "I'll get back to you.",
        
        # Opinions
        "I think it's good.",
        "I'm not sure about that.",
        "That sounds interesting.",
        "I agree with you.",
        "I don't know.",
        "Maybe you're right.",
        "That makes sense.",
        "I hadn't thought of that.",
        
        # Time references
        "I have to go now.",
        "I'll be there soon.",
        "It happened yesterday.",
        "I'm running late.",
        "The meeting is at 3.",
        "I'll do it tomorrow.",
        "It's almost Friday.",
        "The weekend is here.",
    ]
    
    # Add with small variations
    for pattern in patterns:
        texts.append(pattern)
        
        # Add variations
        variations = [
            pattern.replace("I ", "We "),
            pattern.replace(".", "!"),
            pattern.replace("I'm", "I am"),
            pattern.replace("I'll", "I will"),
            pattern.lower(),
        ]
        
        for var in variations:
            if var != pattern and var not in seen:
                texts.append(var)
                seen.add(var)
    
    print(f"  ✓ Total simple sentences: {len(texts):,}")
    
    return texts


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("   FIXING SHORT TEXT PROBLEM")
    print("="*70)
    
    random.seed(SEED)
    
    # Load existing data
    print("\n[STEP 1] Loading existing data...")
    
    with open(os.path.join(EXISTING_CACHE, "human_texts.pkl"), "rb") as f:
        human_texts = pickle.load(f)
    with open(os.path.join(EXISTING_CACHE, "ai_texts.pkl"), "rb") as f:
        ai_texts = pickle.load(f)
    
    print(f"  Existing human: {len(human_texts):,}")
    print(f"  Existing AI: {len(ai_texts):,}")
    
    # Analyze length distribution
    human_lengths = [len(t.split()) for t in human_texts[:10000]]
    ai_lengths = [len(t.split()) for t in ai_texts[:10000]]
    
    print(f"\n  Current length distribution (sample):")
    print(f"    Human avg: {sum(human_lengths)/len(human_lengths):.0f} words")
    print(f"    AI avg: {sum(ai_lengths)/len(ai_lengths):.0f} words")
    print(f"    Human short (<20 words): {sum(1 for l in human_lengths if l < 20)}")
    print(f"    AI short (<20 words): {sum(1 for l in ai_lengths if l < 20)}")
    
    # Collect new short human text
    print("\n[STEP 2] Collecting short human text...")
    short_texts = collect_short_human_text()
    simple_texts = collect_simple_complete_sentences()
    
    # Combine
    new_human = list(set(short_texts + simple_texts))
    print(f"\n  New short human texts: {len(new_human):,}")
    
    # Add to existing human texts
    print("\n[STEP 3] Combining data...")
    
    # Remove duplicates
    existing_set = set(human_texts)
    new_unique = [t for t in new_human if t not in existing_set]
    print(f"  New unique texts: {len(new_unique):,}")
    
    # Add new texts (prioritize them by putting at front)
    combined_human = new_unique + human_texts
    
    # Also need to make sure we have SHORT AI text for balance
    # Filter AI texts by length
    short_ai = [t for t in ai_texts if len(t.split()) <= 30]
    medium_ai = [t for t in ai_texts if 30 < len(t.split()) <= 100]
    long_ai = [t for t in ai_texts if len(t.split()) > 100]
    
    print(f"\n  AI by length:")
    print(f"    Short (≤30 words): {len(short_ai):,}")
    print(f"    Medium (31-100): {len(medium_ai):,}")
    print(f"    Long (>100): {len(long_ai):,}")
    
    # Balance the AI set to have similar length distribution
    # We want more long/medium AI to contrast with short human
    combined_ai = medium_ai + long_ai + short_ai[:len(short_ai)//3]
    
    # Shuffle
    random.shuffle(combined_human)
    random.shuffle(combined_ai)
    
    # Balance
    min_size = min(len(combined_human), len(combined_ai))
    combined_human = combined_human[:min_size]
    combined_ai = combined_ai[:min_size]
    
    print(f"\n[STEP 4] Final dataset:")
    print(f"  Human: {len(combined_human):,}")
    print(f"  AI: {len(combined_ai):,}")
    
    # Verify new length distribution
    new_human_lengths = [len(t.split()) for t in combined_human[:10000]]
    new_ai_lengths = [len(t.split()) for t in combined_ai[:10000]]
    
    print(f"\n  New length distribution:")
    print(f"    Human avg: {sum(new_human_lengths)/len(new_human_lengths):.0f} words")
    print(f"    AI avg: {sum(new_ai_lengths)/len(new_ai_lengths):.0f} words")
    print(f"    Human short (<20 words): {sum(1 for l in new_human_lengths if l < 20)}")
    print(f"    AI short (<20 words): {sum(1 for l in new_ai_lengths if l < 20)}")
    
    # Save
    print("\n[STEP 5] Saving...")
    
    with open(os.path.join(OUTPUT_DIR, "human_texts.pkl"), "wb") as f:
        pickle.dump(combined_human, f)
    
    with open(os.path.join(OUTPUT_DIR, "ai_texts.pkl"), "wb") as f:
        pickle.dump(combined_ai, f)
    
    import json
    with open(os.path.join(OUTPUT_DIR, "cache_info.json"), "w") as f:
        json.dump({
            "human_count": len(combined_human),
            "ai_count": len(combined_ai),
            "fix": "Added short human text"
        }, f, indent=2)
    
    print(f"\n  ✓ Saved to {OUTPUT_DIR}")
    
    # Show samples
    print("\n" + "="*70)
    print("SAMPLE SHORT HUMAN TEXTS (verify these look human!)")
    print("="*70)
    
    short_samples = [t for t in combined_human if len(t.split()) <= 15][:10]
    for i, text in enumerate(short_samples):
        print(f"\n  [{i+1}] {text}")
    
    print("\n" + "="*70)
    print("✓ DONE!")
    print("="*70)
    print(f"\nNow train with the fixed data:")
    print(f'  python train.py --cache-dir "{OUTPUT_DIR}" --samples 200000 --epochs 3')


if __name__ == "__main__":
    main()