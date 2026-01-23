import json

# Load your chunked dataset
with open("chunked_gutenberg.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Take only the first 10–20 chunks for manual humanization
sample_chunks = data[:10]  # you can change 10 → 20

# Create a list to store your humanized examples
humanized_data = []

for i, entry in enumerate(sample_chunks):
    original = entry['text']
    
    # <-- MANUALLY FILL IN YOUR HUMANIZED TEXT HERE -->
    # Example:
    humanized = ""  # <-- replace the empty string with your humanized version

    humanized_data.append({
        "original": original,
        "humanized": humanized
    })

# Save your humanized examples
with open("humanized_sample.json", "w", encoding="utf-8") as f:
    json.dump(humanized_data, f, indent=2, ensure_ascii=False)

print("✅ Humanized sample saved as humanized_sample.json")
