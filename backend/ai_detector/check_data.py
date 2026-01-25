# check_data.py
"""
Check data quality
"""

import os
import pandas as pd
from collections import Counter

DATA_DIR = r"F:\data\ai_detector_cache"  # or your cache dir

print("Checking cached data...")

# Load samples
import pickle

with open(os.path.join(DATA_DIR, "human_texts.pkl"), "rb") as f:
    human_texts = pickle.load(f)

with open(os.path.join(DATA_DIR, "ai_texts.pkl"), "rb") as f:
    ai_texts = pickle.load(f)

print(f"\nHuman texts: {len(human_texts):,}")
print(f"AI texts: {len(ai_texts):,}")

# Check samples
print("\n" + "="*60)
print("SAMPLE HUMAN TEXTS (are these actually human-written?)")
print("="*60)
for i, text in enumerate(human_texts[:10]):
    print(f"\n[{i+1}] {text[:200]}...")

print("\n" + "="*60)
print("SAMPLE AI TEXTS (are these actually AI-generated?)")
print("="*60)
for i, text in enumerate(ai_texts[:10]):
    print(f"\n[{i+1}] {text[:200]}...")

# Check for issues
print("\n" + "="*60)
print("DATA QUALITY CHECKS")
print("="*60)

# Length distribution
human_lens = [len(t.split()) for t in human_texts[:10000]]
ai_lens = [len(t.split()) for t in ai_texts[:10000]]

print(f"\nAverage word count:")
print(f"  Human: {sum(human_lens)/len(human_lens):.0f} words")
print(f"  AI:    {sum(ai_lens)/len(ai_lens):.0f} words")

# Check for duplicates
human_set = set(human_texts[:10000])
ai_set = set(ai_texts[:10000])
overlap = human_set & ai_set

print(f"\nDuplicates/Overlap:")
print(f"  Unique human texts: {len(human_set):,}")
print(f"  Unique AI texts: {len(ai_set):,}")
print(f"  Overlap (same text in both!): {len(overlap):,}")

if len(overlap) > 0:
    print("\n⚠️  WARNING: Some texts appear in BOTH human and AI sets!")
    print("    This will confuse the model.")

# Check for very short texts
short_human = sum(1 for t in human_texts[:10000] if len(t.split()) < 10)
short_ai = sum(1 for t in ai_texts[:10000] if len(t.split()) < 10)

print(f"\nVery short texts (<10 words):")
print(f"  Human: {short_human}")
print(f"  AI: {short_ai}")