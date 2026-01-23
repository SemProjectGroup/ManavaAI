import json
import openai

# ----------------------------
# CONFIGURATION
# ----------------------------
CHUNKED_FILE = "chunked_gutenberg.json"       # your chunked dataset
HUMANIZED_SAMPLE_FILE = "humanized_sample.json"  # your manual examples
OUTPUT_FILE = "humanized_full.json"           # output file

USE_AI = True
API_KEY = "YOUR_OPENAI_API_KEY_HERE"  # <-- Replace with your key
# ----------------------------

openai.api_key = API_KEY

# Load chunked dataset
with open(CHUNKED_FILE, "r", encoding="utf-8") as f:
    chunked_data = json.load(f)

# Load humanized examples (style guide)
with open(HUMANIZED_SAMPLE_FILE, "r", encoding="utf-8") as f:
    sample_humanized = json.load(f)

style_examples = ""
for ex in sample_humanized:
    style_examples += f"Original: {ex['original']}\nHumanized: {ex['humanized']}\n\n"

humanized_full = []

# Optional: process only first N chunks for testing
# chunked_data = chunked_data[:20]

for i, entry in enumerate(chunked_data):
    original = entry['text']

    if USE_AI:
        prompt = f"""Rewrite the following text in a friendly, clear, and conversational style,
following these examples:

{style_examples}

Text: {original}

Humanized:"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that humanizes text in a friendly, clear style."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )

        humanized = response.choices[0].message.content.strip()
    else:
        # If not using AI, just copy original text (placeholder)
        humanized = original

    humanized_full.append({
        "original": original,
        "humanized": humanized
    })

    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{len(chunked_data)} chunks")

# Save the fully humanized dataset
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(humanized_full, f, indent=2, ensure_ascii=False)

print(f"✅ Humanized full dataset saved as {OUTPUT_FILE}")
