import numpy as np
from datasets import load_metric
from textstat import flesch_kincaid_grade
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class HumanizerMetrics:
    def __init__(self, device: str = "cpu"):
        self.bleu = load_metric("sacrebleu")
        self.device = device
        
        # Perplexity Model (GPT-2 is standard for PPL)
        self.ppl_model_id = "gpt2"
        self.ppl_model = None
        self.ppl_tokenizer = None

    def _load_ppl_model(self):
        if self.ppl_model is None:
            self.ppl_model = GPT2LMHeadModel.from_pretrained(self.ppl_model_id).to(self.device)
            self.ppl_tokenizer = GPT2TokenizerFast.from_pretrained(self.ppl_model_id)

    def calculate_bleu(self, predictions, references):
        """
        Calculates BLEU score.
        Args:
            predictions: List of generated strings.
            references: List of lists of reference strings [[ref1], [ref2], ...].
        """
        results = self.bleu.compute(predictions=predictions, references=references)
        return results["score"]

    def calculate_readability(self, text):
        return flesch_kincaid_grade(text)

    def calculate_perplexity(self, text):
        self._load_ppl_model()
        encodings = self.ppl_tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.ppl_model(input_ids, labels=input_ids)
            loss = outputs.loss
            ppl = torch.exp(loss)
        
        return ppl.item()

    def compute_metrics(self, eval_preds, tokenizer):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        bleu = self.calculate_bleu(decoded_preds, decoded_labels)
        
        # Calculate average readability on predictions
        readability = np.mean([self.calculate_readability(pred) for pred in decoded_preds])

        return {
            "bleu": bleu,
            "readability_grade": readability
        }
