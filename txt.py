import json

# Step 1: Read the text file
with open("gutenberg_dataset.txt", "r", encoding="utf-8") as f:
    # Split by double newlines between books
    books = f.read().split("\n\n")

# Step 2: Optional cleanup: remove empty strings
books = [book.strip() for book in books if book.strip()]

# Step 3: Write to JSON file
with open("gutenberg_dataset.json", "w", encoding="utf-8") as f:
    json.dump(books, f, indent=4)

print("Conversion complete! Saved as 'gutenberg_dataset.json'")
