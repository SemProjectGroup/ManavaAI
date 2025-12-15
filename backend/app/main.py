import asyncio
import json
import random
import os
from groq import AsyncGroq
import google.generativeai as genai


RAW_DATA_DIR = "data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)
FILE_PATH = os.path.join(RAW_DATA_DIR, "raw_ai_texts.jsonl")


GROQ_API_KEY = ""
GEMINI_API_KEY = ""

groq_client = AsyncGroq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite") # Use Flash-Lite for max free quota


# Topics and styles for training data diversity
TOPICS = ["Cybersecurity trends", "Micro-economies", "Sustainable architecture", "Marine biology", "Renaissance history"]
STYLES = ["Technical guide", "First-person blog", "Academic abstract", "Journalistic report"]

def get_random_prompt():
    topic = random.choice(TOPICS)
    style = random.choice(STYLES)
    return f"Write a {style} of about 350 words regarding {topic}. Focus on deep detail."

# 4. FREE API WRAPPERS
async def call_groq(prompt):
    try:
        response = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", # High free-tier throughput
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception:
        return None

async def call_gemini(prompt):
    try:
        response = await gemini_model.generate_content_async(prompt)
        return response.text
    except Exception:
        return None


async def main():
    TOTAL_TARGET = 5000
    current_count = 0
    print(f"🚀 Generation started. Target: {TOTAL_TARGET} texts.")

    while current_count < TOTAL_TARGET:
        # Create a batch of requests (respecting RPM limits)
        tasks = []
        # Groq can handle more, but we stay safe at 5 per 15-second cycle
        for _ in range(5): tasks.append(call_groq(get_random_prompt()))
        for _ in range(2): tasks.append(call_gemini(get_random_prompt()))

        results = await asyncio.gather(*tasks)

        
        with open(FILE_PATH, "a", encoding="utf-8") as f:
            for text in results:
                if text:
                    data_entry = {
                        "id": current_count,
                        "text": text,
                        "engine": "free_tier_mix"
                    }
                    f.write(json.dumps(data_entry) + "\n")
                    current_count += 1
        
        print(f"📊 Progress: {current_count}/{TOTAL_TARGET} texts saved to {RAW_DATA_DIR}")
        
        # Pause to avoid "429 Rate Limit" errors on free tiers
        await asyncio.sleep(15)

if __name__ == "__main__":
    asyncio.run(main())
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
