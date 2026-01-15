"""
Download PROPERLY LABELED AI Detection Datasets
These datasets have verified human vs AI text
"""

import os
import gc
import pickle
import random
from tqdm import tqdm

# Install datasets if needed
try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'datasets'])
    from datasets import load_dataset

# ============================================================================
# CONFIG
# ============================================================================

OUTPUT_DIR = r"F:\data\proper_ai_dataset"
TARGET_PER_CLASS = 200_000  # 200K each

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# COLLECTION
# ============================================================================

human_texts = []
ai_texts = []

print("="*70)
print("DOWNLOADING PROPERLY LABELED AI DETECTION DATA")
print("="*70)


# -----------------------------------------------------------------------------
# 1. HC3 Dataset - Human answers vs ChatGPT answers (BEST QUALITY)
# -----------------------------------------------------------------------------
print("\n[1/6] HC3 Dataset (Human vs ChatGPT)...")
try:
    for config in ['all', 'finance', 'medicine', 'open_qa', 'wiki_csai']:
        try:
            ds = load_dataset("Hello-SimpleAI/HC3", config, split="train", trust_remote_code=True)
            for item in tqdm(ds, desc=f"HC3-{config}"):
                # Human answers
                for ans in item.get('human_answers', []):
                    if ans and len(ans.split()) >= 15:
                        human_texts.append(ans.strip())
                # ChatGPT answers  
                for ans in item.get('chatgpt_answers', []):
                    if ans and len(ans.split()) >= 15:
                        ai_texts.append(ans.strip())
        except Exception as e:
            print(f"  Skipping HC3-{config}: {e}")
    print(f"  ✓ Human: {len(human_texts):,}, AI: {len(ai_texts):,}")
except Exception as e:
    print(f"  ✗ Error: {e}")


# -----------------------------------------------------------------------------
# 2. GPT-Wiki-Intro - Wikipedia vs GPT-generated intros
# -----------------------------------------------------------------------------
print("\n[2/6] GPT-Wiki-Intro...")
try:
    ds = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
    for item in tqdm(ds, desc="GPT-Wiki"):
        wiki = item.get('wiki_intro', '')
        gpt = item.get('generated_intro', '')
        if wiki and len(wiki.split()) >= 15:
            human_texts.append(wiki.strip())
        if gpt and len(gpt.split()) >= 15:
            ai_texts.append(gpt.strip())
    print(f"  ✓ Human: {len(human_texts):,}, AI: {len(ai_texts):,}")
except Exception as e:
    print(f"  ✗ Error: {e}")


# -----------------------------------------------------------------------------
# 3. OpenAI Detection Dataset 
# -----------------------------------------------------------------------------
print("\n[3/6] AI Detection Datasets...")
detection_datasets = [
    ("Hello-SimpleAI/chatgpt-detector-roberta", None),
    ("yaful/DeepfakeTextDetection", None),
]

for ds_name, config in detection_datasets:
    try:
        print(f"  Loading {ds_name}...")
        if config:
            ds = load_dataset(ds_name, config, split="train", trust_remote_code=True)
        else:
            ds = load_dataset(ds_name, split="train", trust_remote_code=True)
        
        for item in tqdm(ds, desc=ds_name.split("/")[-1][:20]):
            text = item.get('text', '')
            label = item.get('label', -1)
            
            if not text or len(text.split()) < 15:
                continue
                
            if label == 0:  # Human
                human_texts.append(text.strip())
            elif label == 1:  # AI
                ai_texts.append(text.strip())
                
        print(f"  ✓ Human: {len(human_texts):,}, AI: {len(ai_texts):,}")
    except Exception as e:
        print(f"  ✗ {ds_name}: {e}")


# -----------------------------------------------------------------------------
# 4. Reddit/Twitter - Casual Human Text (IMPORTANT!)
# -----------------------------------------------------------------------------
print("\n[4/6] Casual Human Text (Reddit/Social)...")

# Reddit Writing
try:
    ds = load_dataset("eli5", split="train_eli5", trust_remote_code=True)
    count = 0
    for item in tqdm(ds, desc="ELI5", total=50000):
        if count >= 50000:
            break
        # Questions are definitely human
        q = item.get('title', '')
        if q and len(q.split()) >= 10:
            human_texts.append(q.strip())
            count += 1
    print(f"  ✓ Added {count:,} ELI5 questions")
except Exception as e:
    print(f"  ✗ ELI5: {e}")

# Writing Prompts stories (human creative writing)
try:
    ds = load_dataset("euclaise/writingprompts", split="train", streaming=True)
    count = 0
    for item in tqdm(ds, desc="WritingPrompts", total=30000):
        if count >= 30000:
            break
        story = item.get('story', '')
        if story and 20 <= len(story.split()) <= 500:
            human_texts.append(story.strip())
            count += 1
    print(f"  ✓ Added {count:,} writing prompts")
except Exception as e:
    print(f"  ✗ WritingPrompts: {e}")


# -----------------------------------------------------------------------------
# 5. More AI Text - Alpaca/Dolly/ShareGPT Responses
# -----------------------------------------------------------------------------
print("\n[5/6] AI Assistant Responses...")

# Alpaca (GPT-generated)
try:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    for item in tqdm(ds, desc="Alpaca"):
        output = item.get('output', '')
        if output and len(output.split()) >= 15:
            ai_texts.append(output.strip())
    print(f"  ✓ Human: {len(human_texts):,}, AI: {len(ai_texts):,}")
except Exception as e:
    print(f"  ✗ Alpaca: {e}")

# Dolly (GPT-generated)
try:
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    for item in tqdm(ds, desc="Dolly"):
        response = item.get('response', '')
        if response and len(response.split()) >= 15:
            ai_texts.append(response.strip())
    print(f"  ✓ Human: {len(human_texts):,}, AI: {len(ai_texts):,}")
except Exception as e:
    print(f"  ✗ Dolly: {e}")

# OpenAssistant (AI responses)
try:
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    for item in tqdm(ds, desc="OASST"):
        if item.get('role') == 'assistant':
            text = item.get('text', '')
            if text and len(text.split()) >= 15:
                ai_texts.append(text.strip())
    print(f"  ✓ Human: {len(human_texts):,}, AI: {len(ai_texts):,}")
except Exception as e:
    print(f"  ✗ OASST: {e}")


# -----------------------------------------------------------------------------
# 6. IMDB/Yelp Reviews - Human opinions
# -----------------------------------------------------------------------------
print("\n[6/6] Human Reviews & Opinions...")
try:
    ds = load_dataset("imdb", split="train")
    count = 0
    for item in tqdm(ds, desc="IMDB"):
        if count >= 25000:
            break
        text = item.get('text', '')
        if text and 20 <= len(text.split()) <= 300:
            human_texts.append(text.strip())
            count += 1
    print(f"  ✓ Added {count:,} IMDB reviews")
except Exception as e:
    print(f"  ✗ IMDB: {e}")

try:
    ds = load_dataset("yelp_review_full", split="train")
    count = 0
    for item in tqdm(ds, desc="Yelp"):
        if count >= 25000:
            break
        text = item.get('text', '')
        if text and 20 <= len(text.split()) <= 300:
            human_texts.append(text.strip())
            count += 1
    print(f"  ✓ Added {count:,} Yelp reviews")
except Exception as e:
    print(f"  ✗ Yelp: {e}")


# ============================================================================
# FINALIZE
# ============================================================================

print("\n" + "="*70)
print("FINALIZING DATASET")
print("="*70)

print(f"\nCollected:")
print(f"  Human: {len(human_texts):,}")
print(f"  AI: {len(ai_texts):,}")

# Remove duplicates
print("\nRemoving duplicates...")
human_texts = list(set(human_texts))
ai_texts = list(set(ai_texts))
print(f"  Human (unique): {len(human_texts):,}")
print(f"  AI (unique): {len(ai_texts):,}")

# Remove any overlap
print("\nRemoving overlap...")
human_set = set(human_texts)
ai_set = set(ai_texts)
overlap = human_set & ai_set
if overlap:
    print(f"  Found {len(overlap):,} overlapping texts, removing from both...")
    human_texts = [t for t in human_texts if t not in overlap]
    ai_texts = [t for t in ai_texts if t not in overlap]

# Balance
min_size = min(len(human_texts), len(ai_texts), TARGET_PER_CLASS)
print(f"\nBalancing to {min_size:,} per class...")

random.seed(42)
random.shuffle(human_texts)
random.shuffle(ai_texts)

human_texts = human_texts[:min_size]
ai_texts = ai_texts[:min_size]

# Save
print("\nSaving...")

with open(os.path.join(OUTPUT_DIR, "human_texts.pkl"), "wb") as f:
    pickle.dump(human_texts, f)

with open(os.path.join(OUTPUT_DIR, "ai_texts.pkl"), "wb") as f:
    pickle.dump(ai_texts, f)

import json
with open(os.path.join(OUTPUT_DIR, "cache_info.json"), "w") as f:
    json.dump({
        "human_count": len(human_texts),
        "ai_count": len(ai_texts),
        "sources": [
            "HC3 (Human vs ChatGPT)", 
            "GPT-Wiki-Intro",
            "AI Detection datasets",
            "ELI5 questions",
            "WritingPrompts",
            "Alpaca/Dolly/OASST",
            "IMDB/Yelp reviews"
        ]
    }, f, indent=2)

print(f"\n{'='*70}")
print("✓ DONE!")
print(f"{'='*70}")
print(f"\n  Saved to: {OUTPUT_DIR}")
print(f"  Human: {len(human_texts):,}")
print(f"  AI: {len(ai_texts):,}")

# Show samples
print(f"\n{'='*70}")
print("SAMPLE DATA (Verify these look correct!)")
print(f"{'='*70}")

print("\n[HUMAN SAMPLES] - Should be casual/natural writing:")
for i, text in enumerate(human_texts[:5]):
    print(f"  {i+1}. {text[:100]}...")

print("\n[AI SAMPLES] - Should be AI assistant responses:")
for i, text in enumerate(ai_texts[:5]):
    print(f"  {i+1}. {text[:100]}...")

print(f"\n{'='*70}")
print("NEXT STEP: Train with this data:")
print(f"  python train.py --cache-dir \"{OUTPUT_DIR}\" --samples 100000 --epochs 3")
print(f"{'='*70}")