# diagnose.py
"""
Diagnose why your AI detector isn't accurate
"""

import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model
MODEL_PATH = "trained_model/best"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=1)
    return {
        "ai_pct": round(probs[0][1].item() * 100, 1),
        "human_pct": round(probs[0][0].item() * 100, 1)
    }

print("\n" + "="*70)
print("DIAGNOSTIC TEST")
print("="*70)

# Test cases organized by expected result
test_cases = {
    "SHOULD BE HUMAN (Casual/Informal)": [
        "lol cant believe she said that!! gonna tell everyone tomorrow",
        "Just got home, so tired. Need coffee badly rn",
        "My cat knocked over my plant AGAIN. Why is she like this smh",
        "anyone else think that movie was kinda mid? idk felt overhyped",
        "Bro I literally just woke up and it's already 2pm oops",
    ],
    
    "SHOULD BE HUMAN (Personal/Emotional)": [
        "I've been feeling really anxious lately about my job interview next week.",
        "My grandmother passed away last month and I still can't believe she's gone.",
        "Finally finished my degree after 5 years! Mom cried when I told her.",
        "Had the worst day ever. Spilled coffee, missed my bus, then it rained.",
        "I love my dog so much, he always knows when I'm sad and cuddles me.",
    ],
    
    "SHOULD BE HUMAN (Simple/Direct)": [
        "I went to the store and bought some milk.",
        "The weather is nice today so I walked to work.",
        "My friend called me yesterday about the party.",
        "I made pasta for dinner. It was pretty good.",
        "The meeting got moved to Thursday instead.",
    ],
    
    "SHOULD BE AI (Formal/Academic)": [
        "The implementation of artificial intelligence in modern healthcare systems represents a paradigm shift in diagnostic methodologies.",
        "In conclusion, the multifaceted nature of this phenomenon necessitates a comprehensive examination of its various implications.",
        "Furthermore, it is imperative to acknowledge that the aforementioned considerations play a crucial role.",
        "This analysis delves into the intricate complexities of contemporary socioeconomic frameworks.",
        "The synthesis of these perspectives illuminates the fundamental principles underlying this discourse.",
    ],
    
    "SHOULD BE AI (Structured/Listy)": [
        "There are several key factors to consider: First, the economic implications. Second, the social impact. Third, the environmental consequences.",
        "In summary, the main points are: 1) increased efficiency, 2) reduced costs, and 3) improved outcomes.",
        "To address this issue, we must: understand the root causes, develop strategic solutions, and implement systematic changes.",
        "The benefits include: enhanced productivity, streamlined processes, and optimized resource allocation.",
        "Key considerations encompass: regulatory compliance, stakeholder engagement, and sustainable practices.",
    ],
    
    "SHOULD BE AI (Overly Helpful)": [
        "I'd be happy to help you with that! Here's a comprehensive overview of the topic.",
        "Great question! Let me break this down into manageable parts for you.",
        "Absolutely! Here are some suggestions that might be helpful for your situation.",
        "That's an interesting point. Let me provide some context and additional information.",
        "I understand your concern. Here's what I would recommend based on the information provided.",
    ],
    
    "TRICKY - Could be either": [
        "The new iPhone has some interesting features but I'm not sure it's worth upgrading.",
        "Climate change is affecting weather patterns in unexpected ways.",
        "Learning a new language takes time and consistent practice.",
        "The restaurant downtown has really good food but it's a bit expensive.",
        "Technology has changed how we communicate with each other.",
    ],
}

results = {"correct": 0, "wrong": 0, "details": []}

for category, texts in test_cases.items():
    print(f"\n{'─'*70}")
    print(f"📋 {category}")
    print(f"{'─'*70}")
    
    expected_ai = "AI" in category
    
    for text in texts:
        result = predict(text)
        predicted_ai = result["ai_pct"] > 50
        
        if "TRICKY" in category:
            status = "⚪"  # Neutral for tricky cases
        elif predicted_ai == expected_ai:
            status = "✅"
            results["correct"] += 1
        else:
            status = "❌"
            results["wrong"] += 1
            results["details"].append({
                "text": text[:50] + "...",
                "expected": "AI" if expected_ai else "Human",
                "got": f"AI {result['ai_pct']}%" if predicted_ai else f"Human {result['human_pct']}%"
            })
        
        print(f"{status} [{result['ai_pct']:5.1f}% AI] {text[:60]}...")

# Summary
total = results["correct"] + results["wrong"]
accuracy = results["correct"] / total * 100 if total > 0 else 0

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Correct: {results['correct']}/{total} ({accuracy:.1f}%)")
print(f"Wrong:   {results['wrong']}/{total}")

if results["details"]:
    print(f"\n❌ MISCLASSIFIED EXAMPLES:")
    for d in results["details"][:10]:
        print(f"   Expected {d['expected']}, Got {d['got']}: {d['text']}")

# Check training results
print(f"\n{'='*70}")
print("TRAINING RESULTS")
print(f"{'='*70}")

results_file = os.path.join(MODEL_PATH.replace("/best", ""), "results.json")
if os.path.exists(results_file):
    with open(results_file) as f:
        train_results = json.load(f)
    print(f"  Samples used:  {train_results.get('samples', 'N/A'):,}")
    print(f"  Epochs:        {train_results.get('epochs', 'N/A')}")
    print(f"  Test Accuracy: {train_results.get('test_acc', 0)*100:.1f}%")
    print(f"  Test F1:       {train_results.get('test_f1', 0)*100:.1f}%")
else:
    print("  No results.json found")

print(f"\n{'='*70}")
print("RECOMMENDATIONS")
print(f"{'='*70}")

if accuracy < 70:
    print("⚠️  Accuracy is LOW. Major issues detected.")
    print("    → Your training data might have quality issues")
    print("    → Labels might be swapped or incorrect")
    print("    → Need to check data source files")
elif accuracy < 85:
    print("⚠️  Accuracy is MODERATE. Room for improvement.")
    print("    → Try training with more samples (200K+)")
    print("    → Try a better model (RoBERTa, DeBERTa)")
    print("    → May need more diverse training data")
else:
    print("✅ Accuracy looks GOOD on test cases.")
    print("    → If still having issues, it might be edge cases")
    print("    → Consider what specific texts are failing")

print("="*70)