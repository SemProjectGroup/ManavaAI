import random

def humanize_text_logic(text):
    """
    Step 7 Pipeline:
    1. Sentence Variation (Burstiness)
    2. Randomness (Perplexity)
    3. Natural flow adjustments
    """
    sentences = text.split('. ')
    humanized_sentences = []

    for i, sentence in enumerate(sentences):
        words = sentence.split()
        
        # 1. ADD BURSTINESS: Randomly change sentence lengths
        if i % 2 == 0 and len(words) > 5:
            # Shorten or punch up every second sentence
            sentence = "Basically, " + sentence[0].lower() + sentence[1:]
        
        # 2. ADD RANDOMNESS: Inject 'filler' words common in human speech
        fillers = ["honestly", "frankly", "you see", "actually"]
        if random.random() > 0.7: # 30% chance to add a filler
            sentence = random.choice(fillers).capitalize() + ", " + sentence[0].lower() + sentence[1:]

        humanized_sentences.append(sentence)

    # Rejoin and return
    final_text = ". ".join(humanized_sentences)
    if not final_text.endswith('.'):
        final_text += "."
        
    return final_text

# TEST THE SCRIPT
if __name__ == "__main__":
    sample = "AI generated text is very predictable. It uses the same patterns. It is very robotic."
    print("ORIGINAL:", sample)
    print("HUMANIZED:", humanize_text_logic(sample))