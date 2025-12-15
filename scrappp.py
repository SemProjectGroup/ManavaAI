import requests

# List of Gutenberg text file URLs
books_urls = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
    "https://www.gutenberg.org/files/11/11-0.txt",      # Alice in Wonderland
    "https://www.gutenberg.org/files/1661/1661-0.txt"   # Sherlock Holmes
]

all_text = []

for url in books_urls:
    response = requests.get(url)
    if response.status_code == 200:
        text = response.text
        
        # Remove Gutenberg header/footer
        start_idx = text.find("*** START OF THIS PROJECT GUTENBERG EBOOK")
        if start_idx != -1:
            text = text[start_idx:]
        end_idx = text.find("*** END OF THIS PROJECT GUTENBERG EBOOK")
        if end_idx != -1:
            text = text[:end_idx]
        
        all_text.append(text.strip())
        print(f"Downloaded book from: {url}")
    else:
        print(f"Failed to download: {url}")

# Combine all books into one dataset
dataset_text = "\n\n".join(all_text)

# Save to a file
with open("gutenberg_dataset.txt", "w", encoding="utf-8") as f:
    f.write(dataset_text)

print("All books saved to gutenberg_dataset.txt")
