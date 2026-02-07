"""
AI Text Generation Module
=========================
Generates AI-written text using various open-source language models.
This creates the "AI" portion of the training dataset.

Supported Models:
    - GPT-2 (all sizes)
    - GPT-Neo (EleutherAI)
    - GPT-J (EleutherAI)
    - BLOOM (BigScience)
    - OPT (Meta)
    - LLaMA-based models (if available)
"""

import os
import gc
import json
import random
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass

import torch
from tqdm import tqdm
from loguru import logger

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
        set_seed,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.error("transformers library not installed!")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config


@dataclass
class GeneratedSample:
    """Represents a generated text sample."""
    text: str
    model: str
    prompt: str
    generation_params: Dict
    source_id: str
    label: int = 1  # 1 = AI-generated


class AITextGenerator:
    """
    Generates AI-written text using various open-source language models.
    
    This class manages multiple models and generates diverse text samples
    with varying parameters to create a robust training dataset.
    """
    
    def __init__(self, output_path: Optional[Path] = None, device: str = "auto"):
        """
        Initialize the AI text generator.
        
        Args:
            output_path: Path to save generated texts
            device: Device to use for generation (auto, cuda, cpu, mps)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for AI text generation")
        
        self.output_path = output_path or config.data_collection.raw_ai_text_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.config = config.ai_generation
        self.device = self._get_device(device)
        
        self.generated_hashes: Set[str] = set()
        self.samples_generated = 0
        
        # Currently loaded model
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        
        logger.info(f"AITextGenerator initialized. Device: {self.device}")
        logger.info(f"Output path: {self.output_path}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device != "auto":
            return device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for deduplication."""
        return hashlib.md5(text.strip().lower().encode()).hexdigest()
    
    def _clear_gpu_memory(self):
        """Clear GPU memory."""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
        if self.current_tokenizer is not None:
            del self.current_tokenizer
            self.current_tokenizer = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _load_model(self, model_name: str) -> bool:
        """
        Load a model for text generation.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            True if successful, False otherwise
        """
        if self.current_model_name == model_name:
            return True
        
        logger.info(f"Loading model: {model_name}")
        self._clear_gpu_memory()
        
        try:
            # Determine model loading parameters based on device and model size
            load_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            # Check if model might be too large
            large_models = ["6b", "7b", "13b", "2.7b", "1.3b"]
            is_large = any(size in model_name.lower() for size in large_models)
            
            if self.device == "cuda":
                if is_large:
                    # Use 8-bit quantization for large models
                    try:
                        load_kwargs["load_in_8bit"] = True
                        load_kwargs["device_map"] = "auto"
                    except:
                        load_kwargs["torch_dtype"] = torch.float16
                        load_kwargs["device_map"] = "auto"
                else:
                    load_kwargs["torch_dtype"] = torch.float16
            elif self.device == "mps":
                load_kwargs["torch_dtype"] = torch.float16
            
            # Load tokenizer
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Set pad token if not set
            if self.current_tokenizer.pad_token is None:
                self.current_tokenizer.pad_token = self.current_tokenizer.eos_token
            
            # Load model
            self.current_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            )
            
            # Move to device if not using device_map
            if "device_map" not in load_kwargs:
                self.current_model = self.current_model.to(self.device)
            
            self.current_model.eval()
            self.current_model_name = model_name
            
            logger.info(f"Successfully loaded {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            self._clear_gpu_memory()
            return False
    
    def _generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.9,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> Optional[str]:
        """
        Generate text using the currently loaded model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling
            
        Returns:
            Generated text or None if generation fails
        """
        if self.current_model is None or self.current_tokenizer is None:
            logger.error("No model loaded!")
            return None
        
        try:
            # Tokenize prompt
            inputs = self.current_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            )
            
            # Move to device
            if hasattr(self.current_model, "device"):
                device = self.current_model.device
            else:
                device = next(self.current_model.parameters()).device
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.current_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=self.config.min_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    pad_token_id=self.current_tokenizer.pad_token_id,
                    eos_token_id=self.current_tokenizer.eos_token_id,
                    num_return_sequences=1,
                )
            
            # Decode
            generated_text = self.current_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Remove the prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.debug(f"Generation error: {e}")
            return None
    
    def _is_valid_text(self, text: str) -> bool:
        """Check if generated text meets quality criteria."""
        if not text or not text.strip():
            return False
        
        text = text.strip()
        
        # Length checks
        if len(text) < config.data_collection.min_text_length:
            return False
        if len(text) > config.data_collection.max_text_length:
            return False
        
        word_count = len(text.split())
        if word_count < config.data_collection.min_word_count:
            return False
        if word_count > config.data_collection.max_word_count:
            return False
        
        # Quality checks
        # Check for coherence (basic)
        if text.count('...') > 5:
            return False
        
        # Check for repetition
        words = text.lower().split()
        if len(words) > 10:
            # Check for consecutive repeated words
            consecutive_repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i-1])
            if consecutive_repeats > len(words) * 0.2:
                return False
            
            # Check overall uniqueness
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return False
        
        # Deduplication
        text_hash = self._get_text_hash(text)
        if text_hash in self.generated_hashes:
            return False
        
        return True
    
    def _save_sample(self, sample: GeneratedSample):
        """Save a generated sample to the output file."""
        with open(self.output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'text': sample.text,
                'source': sample.model,
                'source_id': sample.source_id,
                'metadata': {
                    'model': sample.model,
                    'prompt': sample.prompt,
                    'generation_params': sample.generation_params
                },
                'label': sample.label
            }) + '\n')
        
        self.generated_hashes.add(self._get_text_hash(sample.text))
        self.samples_generated += 1
    
    def _get_random_params(self) -> Dict:
        """Get randomized generation parameters."""
        return {
            "temperature": random.uniform(*self.config.temperature_range),
            "top_p": random.uniform(*self.config.top_p_range),
            "top_k": random.randint(*self.config.top_k_range),
            "repetition_penalty": random.uniform(*self.config.repetition_penalty_range),
            "max_new_tokens": random.randint(100, self.config.max_new_tokens),
        }
    
    def _get_prompt(self) -> Tuple[str, str]:
        """Get a random prompt with a topic."""
        topic = random.choice(self.config.topics)
        template = random.choice(self.config.prompt_templates)
        prompt = template.format(topic=topic)
        return prompt, topic
    
    def generate_from_model(
        self,
        model_name: str,
        num_samples: int = 1000,
        show_progress: bool = True
    ) -> int:
        """
        Generate text samples from a specific model.
        
        Args:
            model_name: HuggingFace model name
            num_samples: Number of samples to generate
            show_progress: Whether to show progress bar
            
        Returns:
            Number of samples generated
        """
        if not self._load_model(model_name):
            logger.warning(f"Skipping {model_name} - failed to load")
            return 0
        
        initial_count = self.samples_generated
        attempts = 0
        max_attempts = num_samples * 3  # Allow for some failures
        
        progress = tqdm(total=num_samples, desc=f"Generating from {model_name}") if show_progress else None
        
        while self.samples_generated - initial_count < num_samples and attempts < max_attempts:
            attempts += 1
            
            try:
                # Get random prompt and parameters
                prompt, topic = self._get_prompt()
                params = self._get_random_params()
                
                # Generate text
                generated_text = self._generate_text(prompt, **params)
                
                if generated_text and self._is_valid_text(generated_text):
                    sample = GeneratedSample(
                        text=generated_text,
                        model=model_name,
                        prompt=prompt,
                        generation_params=params,
                        source_id=f"{model_name.replace('/', '_')}_{self.samples_generated}",
                        label=1
                    )
                    self._save_sample(sample)
                    
                    if progress:
                        progress.update(1)
                
            except Exception as e:
                logger.debug(f"Generation attempt failed: {e}")
                continue
        
        if progress:
            progress.close()
        
        samples_from_model = self.samples_generated - initial_count
        logger.info(f"Generated {samples_from_model} samples from {model_name}")
        
        return samples_from_model
    
    def generate_all(self, target_samples: Optional[int] = None) -> int:
        """
        Generate AI text from all available models.
        
        Args:
            target_samples: Target number of samples to generate
            
        Returns:
            Total number of samples generated
        """
        target = target_samples or self.config.samples_per_model * len(self.config.generation_models)
        logger.info(f"Starting AI text generation. Target: {target} samples")
        
        # Clear output file if it exists
        if self.output_path.exists():
            self.output_path.unlink()
        
        # Try models in order, use fallbacks if needed
        models_to_try = list(self.config.generation_models)
        random.shuffle(models_to_try)  # Randomize order for diversity
        
        samples_per_model = target // len(models_to_try) + 1
        
        for model_name in models_to_try:
            if self.samples_generated >= target:
                break
            
            remaining = target - self.samples_generated
            samples_to_generate = min(samples_per_model, remaining)
            
            try:
                self.generate_from_model(model_name, num_samples=samples_to_generate)
            except Exception as e:
                logger.error(f"Error with {model_name}: {e}")
                continue
            
            # Clear memory between models
            self._clear_gpu_memory()
        
        # If we didn't reach target, use fallback models
        if self.samples_generated < target:
            logger.info("Using fallback models to reach target...")
            for fallback_model in self.config.fallback_models:
                if self.samples_generated >= target:
                    break
                
                remaining = target - self.samples_generated
                try:
                    self.generate_from_model(fallback_model, num_samples=remaining)
                except Exception as e:
                    logger.error(f"Error with fallback {fallback_model}: {e}")
                    continue
                
                self._clear_gpu_memory()
        
        logger.info(f"AI text generation complete. Total samples: {self.samples_generated}")
        return self.samples_generated
    
    def generate_quick_dataset(self, num_samples: int = 5000) -> int:
        """
        Generate a quick dataset using only small, fast models.
        Good for testing and quick iterations.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Number of samples generated
        """
        logger.info(f"Quick generation mode: {num_samples} samples")
        
        quick_models = ["gpt2", "gpt2-medium", "distilgpt2"]
        samples_per_model = num_samples // len(quick_models) + 1
        
        for model_name in quick_models:
            if self.samples_generated >= num_samples:
                break
            
            remaining = num_samples - self.samples_generated
            self.generate_from_model(model_name, min(samples_per_model, remaining))
            self._clear_gpu_memory()
        
        return self.samples_generated


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for AI text generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate AI-written text using language models")
    parser.add_argument("--target", type=int, default=50000, help="Target number of samples")
    parser.add_argument("--model", type=str, default=None, help="Specific model to use")
    parser.add_argument("--quick", action="store_true", help="Quick mode using small models only")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cuda, cpu, mps)")
    
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else None
    generator = AITextGenerator(output_path=output_path, device=args.device)
    
    if args.quick:
        generator.generate_quick_dataset(num_samples=args.target)
    elif args.model:
        generator.generate_from_model(args.model, num_samples=args.target)
    else:
        generator.generate_all(target_samples=args.target)
    
    print(f"\nGeneration complete! Total samples: {generator.samples_generated}")
    print(f"Output saved to: {generator.output_path}")


if __name__ == "__main__":
    main()