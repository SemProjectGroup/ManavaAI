import json
import os

# ------------------------------
# Input & output files
# ------------------------------
INPUT_FILE = "cleaned_all_docs.json"  # from cleaning step
OUTPUT_FILE = "manava_training.json"  # training-ready dataset

# Load cleaned data
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    cleaned_data = json.load(f)

training_data = []

# Convert each paragraph into an instruction-response pair
for item in cleaned_data:
    text = item["text"]
    training_data.append({
        "instruction": "Humanize the following text",
        "input": "",
        "output": text
    })

# Save training dataset
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(training_data, f, indent=2, ensure_ascii=False)

print("✅ Training dataset created successfully!")
print(f"📁 Output saved as: {OUTPUT_FILE}")
print(f"🧾 Total training samples: {len(training_data)}")
