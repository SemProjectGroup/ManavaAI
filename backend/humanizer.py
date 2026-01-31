import torch
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer

class TextHumanizer:
    def __init__(self, model_name="google/flan-t5-small"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def humanize_sentence(self, text):
        # The 'Secret Sauce' calibration we developed
        words = re.findall(r'\w+', text.lower())
        ban_words = [w for w in words if len(w) > 4]
        bad_words_ids = [self.tokenizer.encode(w, add_special_tokens=False) for w in ban_words]

        input_text = f"Rewrite this clearly: {text}"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs, 
            max_length=128,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
            repetition_penalty=3.5,
            bad_words_ids=bad_words_ids if bad_words_ids else None
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def process_batch(self, texts):
        # Use this for processing rows in a dataframe
        return [self.humanize_sentence(t) for t in texts]