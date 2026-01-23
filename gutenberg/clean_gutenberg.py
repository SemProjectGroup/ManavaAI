import json
import re

INPUT_FILE = "gutenberg_dataset.json"
OUTPUT_FILE = "cleaned_gutenberg.json"

GUTENBERG_NOISE = [
    "PROJECT GUTENBERG",
    "PRODUCED BY",
    "START OF THIS",
    "END OF THIS",
    "GUTENBERG LICENSE"
]

def is_noise(text):
    upper = text.upper()
    return any(word in upper for word in GUTENBERG_NOISE)

def is_chapter_title(text):
    return text.isupper() and len(text) < 60

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = []

    for text in data:
        if not isinstance(text, str):
            continue

        text = text.strip()

        if not text:
            continue

        if is_noise(text):
            continue

        if is_chapter_title(text):
            continue

        if len(text) < 50:
            continue

        text = clean_text(text)

        cleaned_data.append(text)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print("✅ Cleaning complete!")
    print(f"Original entries: {len(data)}")
    print(f"Cleaned entries:  {len(cleaned_data)}")
    print(f"Saved as: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
