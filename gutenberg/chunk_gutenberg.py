import json
import re

INPUT_FILE = "cleaned_gutenberg.json"
OUTPUT_FILE = "chunked_gutenberg.json"
CHUNK_WORDS = 150  # approx words per chunk

def split_into_chunks(text, words_per_chunk=CHUNK_WORDS):
    words = text.split()
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i:i+words_per_chunk])
        chunks.append(chunk)
    return chunks

def clean_extra_spaces(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunked_data = []

    for text in data:
        text = clean_extra_spaces(text)
        chunks = split_into_chunks(text)
        for chunk in chunks:
            # remove very short chunks
            if len(chunk) < 50:
                continue
            chunked_data.append({"text": chunk})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunked_data, f, indent=2, ensure_ascii=False)

    print("✅ Chunking complete!")
    print(f"Original entries: {len(data)}")
    print(f"Chunked entries:  {len(chunked_data)}")
    print(f"Saved as: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
