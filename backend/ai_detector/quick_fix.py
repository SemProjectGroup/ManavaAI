"""
Quick fix: Add the exact failing examples to training data
"""

import os
import pickle
import random

CACHE_DIR = r"F:\data\mega_ai_dataset"  # Your current data
OUTPUT_DIR = r"F:\data\quick_fixed_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# These are examples the model is getting WRONG
# They should be classified as HUMAN
SHORT_HUMAN_EXAMPLES = [
    # Casual chat
    "lol cant believe she said that",
    "gonna tell everyone tomorrow",
    "Bro I literally just woke up",
    "it's already 2pm oops",
    "just got home so tired",
    "need coffee badly rn",
    "anyone else think that movie was mid",
    "idk felt overhyped",
    "smh why is she like this",
    "My cat knocked over my plant AGAIN",
    
    # Personal/emotional
    "I've been feeling really anxious lately",
    "My grandmother passed away last month",
    "I still can't believe she's gone",
    "Finally finished my degree after 5 years",
    "Mom cried when I told her",
    "Had the worst day ever",
    "Spilled coffee missed my bus then it rained",
    "I love my dog so much",
    "he always knows when I'm sad",
    
    # Simple daily sentences
    "I went to the store and bought some milk",
    "The weather is nice today so I walked to work",
    "My friend called me yesterday about the party",
    "I made pasta for dinner",
    "It was pretty good",
    "The meeting got moved to Thursday instead",
    "I need to do laundry tomorrow",
    "I'm going to bed early tonight",
    "I forgot my umbrella again",
    "The bus was late this morning",
    "I ran into an old friend at the grocery store",
    "My phone battery died",
    "I left my keys at home",
    "The coffee machine is broken again",
    "I'm so tired today",
    "I didn't sleep well last night",
    "My back hurts from sitting all day",
    "I should probably exercise more",
    "I ate too much at dinner",
    "The traffic was terrible",
    
    # Short opinions
    "I think it's gonna rain",
    "This food is pretty good",
    "I don't really like this song",
    "That movie was okay I guess",
    "I'm not sure what to do",
    "Maybe we should wait",
    "I agree with you",
    "That's a good point",
    "I hadn't thought of that",
    "Sounds good to me",
    
    # Questions humans ask
    "What time is it",
    "Where did you put the keys",
    "Did you eat yet",
    "Are you coming tonight",
    "What do you want for dinner",
    "How was your day",
    "What happened",
    "Why is it so cold",
    "When does the store close",
    "Who told you that",
]

# Generate more variations
def generate_variations(examples):
    variations = []
    
    for text in examples:
        variations.append(text)
        
        # Add punctuation variations
        if not text.endswith(('.', '!', '?')):
            variations.append(text + '.')
            variations.append(text + '!')
        
        # Capitalize variations
        variations.append(text.lower())
        variations.append(text.capitalize())
        
        # Add filler words
        if len(text.split()) < 10:
            variations.append("So " + text.lower())
            variations.append("Yeah " + text.lower())
            variations.append("I mean " + text.lower())
            variations.append(text + " honestly")
            variations.append(text + " tbh")
    
    return list(set(variations))

print("Loading existing data...")
with open(os.path.join(CACHE_DIR, "human_texts.pkl"), "rb") as f:
    human_texts = pickle.load(f)
with open(os.path.join(CACHE_DIR, "ai_texts.pkl"), "rb") as f:
    ai_texts = pickle.load(f)

print(f"Existing: {len(human_texts):,} human, {len(ai_texts):,} AI")

# Generate variations
print("Generating short human examples...")
short_human = generate_variations(SHORT_HUMAN_EXAMPLES)

# Repeat them to make them significant in training
# Each example appears ~100 times to really hammer it home
short_human_repeated = short_human * 100
random.shuffle(short_human_repeated)

print(f"Generated {len(short_human):,} unique short examples")
print(f"Repeated to {len(short_human_repeated):,} training examples")

# Add to front of human texts (so they're more likely to be in training)
combined_human = short_human_repeated + human_texts

# Shuffle
random.shuffle(combined_human)

# Balance
min_size = min(len(combined_human), len(ai_texts))
combined_human = combined_human[:min_size]
combined_ai = ai_texts[:min_size]

print(f"Final: {len(combined_human):,} human, {len(combined_ai):,} AI")

# Save
print("Saving...")
with open(os.path.join(OUTPUT_DIR, "human_texts.pkl"), "wb") as f:
    pickle.dump(combined_human, f)
with open(os.path.join(OUTPUT_DIR, "ai_texts.pkl"), "wb") as f:
    pickle.dump(combined_ai, f)

import json
with open(os.path.join(OUTPUT_DIR, "cache_info.json"), "w") as f:
    json.dump({"human_count": len(combined_human), "ai_count": len(combined_ai)}, f)

print(f"\n✓ Saved to {OUTPUT_DIR}")
print(f"\nNow train:")
print(f'python train.py --cache-dir "{OUTPUT_DIR}" --samples 200000 --epochs 3')