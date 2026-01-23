import json
import os
import textwrap

# ------------------------------
# Input & output files
# ------------------------------
INPUT_FILE = "manava_training.json"
OUTPUT_FILE = "chunked_training.json"

# Maximum chunk size (in words)
MAX_WORDS = 200  # adjust as needed

# Load the training data
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    training_data = json.load(f)

chunked_data = []

for item in training_data:
    text = item["output"]
    # Split text into chunks of MAX_WORDS
    chunks = textwrap.wrap(text, width=MAX_WORDS, break_long_words=False, replace_whitespace=False)
    for chunk in chunks:
        chunked_data.append({
            "instruction": item["instruction"],
            "input": item["input"],
            "output": chunk.strip()
        })

# Save chunked training dataset
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunked_data, f, indent=2, ensure_ascii=False)

print("✅ Chunking complete!")
print(f"Original entries: {len(training_data)}")
print(f"Chunked entries:  {len(chunked_data)}")
print(f"Saved as: {OUTPUT_FILE}")
