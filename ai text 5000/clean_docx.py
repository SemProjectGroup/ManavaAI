from docx import Document
import json
import re
import os

# ------------------------------
# Current folder (where script + DOCX are)
# ------------------------------
INPUT_FOLDER = "."
OUTPUT_FILE = "cleaned_all_docs.json"

# Regex to remove emojis
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "\U00002B00-\U00002BFF"
    "]+", flags=re.UNICODE
)

# Remove leftover symbols
symbol_pattern = re.compile(r"[^\w\s.,;:!?'-]")

cleaned_data = []

# Loop through all DOCX files
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(".docx"):
        file_path = os.path.join(INPUT_FOLDER, filename)
        doc = Document(file_path)
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            text = emoji_pattern.sub('', text)
            text = re.sub(r"\s+", " ", text)
            if len(text) < 30:
                continue
            text = symbol_pattern.sub('', text)

            cleaned_data.append({"text": text})

# Save cleaned data
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

print("✅ DOCX files cleaned!")
print(f"📁 Saved as: {OUTPUT_FILE}")
print(f"🧾 Total paragraphs: {len(cleaned_data)}")
