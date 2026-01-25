"""
Perplexity Calculator for AI Text Detection
Uses GPT-2 model to calculate perplexity scores for text evaluation
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from typing import Union, List, Dict
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerplexityCalculator:
    """Calculate perplexity scores using GPT-2 model"""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = None,
        max_length: int = 1024
    ):
        """
        Initialize the perplexity calculator
        
        Args:
            model_name: HuggingFace model name (default: "gpt2")
                       Options: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
            device: Device to run model on ("cuda", "cpu", or None for auto)
            max_length: Maximum sequence length for processing
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing PerplexityCalculator with {model_name} on {self.device}")
        
        # Load tokenizer and model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load GPT-2 tokenizer and model"""
        try:
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
            
            logger.info(f"Loading model: {self.model_name}")
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def calculate_perplexity(
        self,
        text: str,
        stride: int = 512
    ) -> float:
        """
        Calculate perplexity score for a single text
        
        Args:
            text: Input text to evaluate
            stride: Stride for sliding window (helps with long texts)
        
        Returns:
            Perplexity score (float). Lower = more predictable (AI-like)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided")
            return float('inf')
        
        try:
            # Tokenize the text
            encodings = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            )
            
            input_ids = encodings.input_ids.to(self.device)
            seq_len = input_ids.size(1)
            
            # For short texts, calculate directly
            if seq_len <= self.max_length:
                return self._calculate_perplexity_simple(input_ids)
            
            # For long texts, use sliding window approach
            return self._calculate_perplexity_sliding(input_ids, stride)
            
        except Exception as e:
            logger.error(f"Error calculating perplexity: {str(e)}")
            return float('inf')
    
    def _calculate_perplexity_simple(self, input_ids: torch.Tensor) -> float:
        """Calculate perplexity for short texts"""
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def _calculate_perplexity_sliding(
        self,
        input_ids: torch.Tensor,
        stride: int
    ) -> float:
        """Calculate perplexity using sliding window for long texts"""
        seq_len = input_ids.size(1)
        nlls = []  # negative log-likelihoods
        prev_end_loc = 0
        
        with torch.no_grad():
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + self.max_length, seq_len)
                trg_len = end_loc - prev_end_loc
                
                input_ids_chunk = input_ids[:, begin_loc:end_loc]
                target_ids = input_ids_chunk.clone()
                target_ids[:, :-trg_len] = -100  # Ignore context tokens
                
                outputs = self.model(input_ids_chunk, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
                
                nlls.append(neg_log_likelihood)
                
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
        
        # Calculate average perplexity
        total_nll = torch.stack(nlls).sum()
        perplexity = torch.exp(total_nll / seq_len).item()
        
        return perplexity
    
    def calculate_batch_perplexity(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[float]:
        """
        Calculate perplexity for multiple texts
        
        Args:
            texts: List of texts to evaluate
            show_progress: Show progress bar
        
        Returns:
            List of perplexity scores
        """
        perplexities = []
        
        iterator = tqdm(texts, desc="Calculating perplexity") if show_progress else texts
        
        for text in iterator:
            perplexity = self.calculate_perplexity(text)
            perplexities.append(perplexity)
        
        return perplexities
    
    def evaluate_with_stats(self, text: str) -> Dict[str, float]:
        """
        Calculate perplexity with additional statistics
        
        Returns:
            Dictionary with perplexity and interpretation
        """
        perplexity = self.calculate_perplexity(text)
        
        # Interpretation thresholds (these are general guidelines)
        if perplexity < 50:
            interpretation = "Very low (likely AI-generated)"
        elif perplexity < 100:
            interpretation = "Low (possibly AI-generated)"
        elif perplexity < 200:
            interpretation = "Moderate (mixed or edited)"
        elif perplexity < 500:
            interpretation = "High (likely human-written)"
        else:
            interpretation = "Very high (likely human-written)"
        
        return {
            "perplexity": perplexity,
            "interpretation": interpretation,
            "log_perplexity": np.log(perplexity) if perplexity > 0 else float('-inf')
        }
    
    def compare_texts(
        self,
        text1: str,
        text2: str,
        label1: str = "Text 1",
        label2: str = "Text 2"
    ) -> Dict[str, any]:
        """
        Compare perplexity scores of two texts
        
        Returns:
            Dictionary with comparison results
        """
        ppl1 = self.calculate_perplexity(text1)
        ppl2 = self.calculate_perplexity(text2)
        
        difference = abs(ppl1 - ppl2)
        percent_change = (difference / ppl1) * 100 if ppl1 > 0 else 0
        
        more_human_like = label1 if ppl1 > ppl2 else label2
        
        return {
            label1: ppl1,
            label2: ppl2,
            "difference": difference,
            "percent_change": percent_change,
            "more_human_like": more_human_like
        }
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "vocab_size": self.tokenizer.vocab_size
        }


def main():
    """Example usage and testing"""
    
    # Initialize calculator
    calculator = PerplexityCalculator(model_name="gpt2", device="cpu")
    
    # Print model info
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    info = calculator.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    print("="*60 + "\n")
    
    # Example texts
    ai_text = """
    Artificial intelligence is revolutionizing the way we approach 
    problem-solving in the modern era. Machine learning algorithms 
    are becoming increasingly sophisticated, enabling computers to 
    perform tasks that were once thought to be exclusively human.
    """
    
    human_text = """
    So yeah, I was walking down the street yesterday and bumped into 
    my old friend from college. We grabbed coffee and just chatted 
    for hours! Time really flies when you're catching up with someone 
    you haven't seen in ages. Made me realize I should reach out more.
    """
    
    # Calculate perplexity scores
    print("Evaluating AI-generated text...")
    ai_result = calculator.evaluate_with_stats(ai_text)
    print(f"Perplexity: {ai_result['perplexity']:.2f}")
    print(f"Interpretation: {ai_result['interpretation']}")
    print()
    
    print("Evaluating Human-written text...")
    human_result = calculator.evaluate_with_stats(human_text)
    print(f"Perplexity: {human_result['perplexity']:.2f}")
    print(f"Interpretation: {human_result['interpretation']}")
    print()
    
    # Compare texts
    print("Comparison:")
    comparison = calculator.compare_texts(
        ai_text, 
        human_text, 
        label1="AI Text", 
        label2="Human Text"
    )
    print(f"AI Text Perplexity: {comparison['AI Text']:.2f}")
    print(f"Human Text Perplexity: {comparison['Human Text']:.2f}")
    print(f"Difference: {comparison['difference']:.2f}")
    print(f"More human-like: {comparison['more_human_like']}")


if __name__ == "__main__":
    main()