from openai import OpenAI
import json, random

client = OpenAI()

prompts = [
    "Explain a simple concept in a human way: ",
    "Write a short paragraph about daily life: ",
    "Describe a random event: ",
    "Give an opinion on a random topic: ",
    "Write a small story about a person: ",
    "Write a short argument about something: "
]

data = []

for i in range(5000):
    p = random.choice(prompts)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": p}]
    )

    text = resp.choices[0].message.content

    data.append({
        "source": "ai",
        "text": text
    })

    print(f"Generated {i+1}/5000")

with open("ai_texts.json", "w") as f:
    json.dump(data, f, indent=4)
