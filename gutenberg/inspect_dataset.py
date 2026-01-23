import json

with open("chunked_gutenberg.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for i in range(10):
    print(data[i])
    print("------")
