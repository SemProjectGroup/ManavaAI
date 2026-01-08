"""
Model Saving Script for AI Text Humanizer
Saves LoRA adapter weights, tokenizer, and creates comprehensive model card
"""

import torch
import json
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for the trained model"""
    
    # Model information
    model_name: str
    base_model: str
    model_type: str
    task: str = "text-humanization"
    language: str = "en"
    
    # Training information
    training_date: str = None
    num_epochs: int = 0
    total_steps: int = 0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    best_val_loss: float = 0.0
    
    # LoRA configuration
    lora_r: int = 0
    lora_alpha: int = 0
    lora_dropout: float = 0.0
    target_modules: List[str] = None
    trainable_params: int = 0
    total_params: int = 0
    trainable_params_percentage: float = 0.0
    
    # Dataset information
    train_dataset_size: int = 0
    val_dataset_size: int = 0
    test_dataset_size: int = 0
    
    # Performance metrics
    perplexity_improvement: float = 0.0
    burstiness_improvement: float = 0.0
    
    # Hardware information
    device_used: str = "unknown"
    training_time_hours: float = 0.0
    
    # Version and framework
    framework_version: str = ""
    peft_version: str = ""
    transformers_version: str = ""
    
    def __post_init__(self):
        """Set default values"""
        if self.training_date is None:
            self.training_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if self.target_modules is None:
            self.target_modules = []
        
        # Get package versions
        try:
            import transformers
            self.transformers_version = transformers.__version__
        except:
            self.transformers_version = "unknown"
        
        try:
            import peft
            self.peft_version = peft.__version__
        except:
            self.peft_version = "unknown"
        
        try:
            self.framework_version = f"torch-{torch.__version__}"
        except:
            self.framework_version = "unknown"


class ModelSaver:
    """Save trained model with all necessary files and metadata"""
    
    def __init__(
        self,
        model,
        tokenizer,
        output_dir: str,
        metadata: Optional[ModelMetadata] = None
    ):
        """
        Initialize model saver
        
        Args:
            model: Trained PEFT model with LoRA adapters
            tokenizer: Tokenizer used for training
            output_dir: Directory to save model
            metadata: Model metadata (optional)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.metadata = metadata or ModelMetadata(
            model_name="humanizer-model",
            base_model="unknown",
            model_type="causal-lm"
        )
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModelSaver initialized. Output: {output_dir}")
    
    def save_lora_adapters(self) -> None:
        """Save LoRA adapter weights"""
        logger.info("Saving LoRA adapter weights...")
        
        try:
            # Save adapter weights and config
            self.model.save_pretrained(self.output_dir)
            logger.info(f"✓ LoRA adapters saved to {self.output_dir}")
            
            # List saved files
            adapter_files = list(Path(self.output_dir).glob("adapter_*"))
            for file in adapter_files:
                size_mb = os.path.getsize(file) / (1024 * 1024)
                logger.info(f"  - {file.name} ({size_mb:.2f} MB)")
                
        except Exception as e:
            logger.error(f"Error saving LoRA adapters: {str(e)}")
            raise
    
    def save_tokenizer(self) -> None:
        """Save tokenizer configuration and vocabulary"""
        logger.info("Saving tokenizer...")
        
        try:
            self.tokenizer.save_pretrained(self.output_dir)
            logger.info(f"✓ Tokenizer saved to {self.output_dir}")
            
            # List saved tokenizer files
            tokenizer_files = [
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json"
            ]
            
            for file_name in tokenizer_files:
                file_path = os.path.join(self.output_dir, file_name)
                if os.path.exists(file_path):
                    size_kb = os.path.getsize(file_path) / 1024
                    logger.info(f"  - {file_name} ({size_kb:.2f} KB)")
                    
        except Exception as e:
            logger.error(f"Error saving tokenizer: {str(e)}")
            raise
    
    def save_metadata_json(self) -> None:
        """Save metadata as JSON"""
        logger.info("Saving metadata (JSON)...")
        
        metadata_path = os.path.join(self.output_dir, "model_metadata.json")
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.metadata), f, indent=2)
            
            logger.info(f"✓ Metadata saved to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            raise
    
    def create_model_card(self) -> None:
        """Create comprehensive model card in markdown format"""
        logger.info("Creating model card...")
        
        model_card = f"""---
language: {self.metadata.language}
license: mit
tags:
- text-generation
- lora
- text-humanization
- ai-detection
- peft
base_model: {self.metadata.base_model}
---

# {self.metadata.model_name}

## Model Description

This model is a fine-tuned version of **{self.metadata.base_model}** using LoRA (Low-Rank Adaptation) adapters.
It is designed to humanize AI-generated text by learning from human-written text patterns.

**Task**: Text Humanization  
**Base Model**: {self.metadata.base_model}  
**Model Type**: {self.metadata.model_type}  
**Language**: {self.metadata.language}

## Training Information

### Training Configuration

- **Training Date**: {self.metadata.training_date}
- **Number of Epochs**: {self.metadata.num_epochs}
- **Total Training Steps**: {self.metadata.total_steps}
- **Device**: {self.metadata.device_used}
- **Training Time**: {self.metadata.training_time_hours:.2f} hours

### LoRA Configuration

- **LoRA Rank (r)**: {self.metadata.lora_r}
- **LoRA Alpha**: {self.metadata.lora_alpha}
- **LoRA Dropout**: {self.metadata.lora_dropout}
- **Target Modules**: {', '.join(self.metadata.target_modules)}
- **Trainable Parameters**: {self.metadata.trainable_params:,} ({self.metadata.trainable_params_percentage:.2f}%)
- **Total Parameters**: {self.metadata.total_params:,}

### Training Results

- **Final Training Loss**: {self.metadata.final_train_loss:.4f}
- **Final Validation Loss**: {self.metadata.final_val_loss:.4f}
- **Best Validation Loss**: {self.metadata.best_val_loss:.4f}

### Dataset

- **Training Samples**: {self.metadata.train_dataset_size:,}
- **Validation Samples**: {self.metadata.val_dataset_size:,}
- **Test Samples**: {self.metadata.test_dataset_size:,}

## Performance Metrics

- **Perplexity Improvement**: {self.metadata.perplexity_improvement:+.2f}%
- **Burstiness Improvement**: {self.metadata.burstiness_improvement:+.2f}%

## How to Use

### Installation

```bash
pip install transformers peft torch
```

### Load Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("{self.metadata.base_model}")
tokenizer = AutoTokenizer.from_pretrained("{self.metadata.base_model}")

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "{self.output_dir}")
model.eval()
```

### Generate Humanized Text

```python
# Prepare input
text = "Your AI-generated text here..."
inputs = tokenizer(text, return_tensors="pt")

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=512,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )

# Decode
humanized_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(humanized_text)
```

## Model Architecture

This model uses LoRA adapters on top of {self.metadata.base_model}. LoRA works by injecting trainable 
rank decomposition matrices into the attention layers, allowing efficient fine-tuning with minimal 
additional parameters.

## Limitations and Bias

- The model is trained primarily on English text
- Performance may vary depending on input text style and domain
- The model may inherit biases present in the training data
- Works best with text similar to the training distribution

## Framework Versions

- **Transformers**: {self.metadata.transformers_version}
- **PEFT**: {self.metadata.peft_version}
- **PyTorch**: {self.metadata.framework_version}

## Citation

If you use this model, please cite:

```bibtex
@misc{{{self.metadata.model_name.replace('-', '_')}}},
  author = {{Your Team Name}},
  title = {{{self.metadata.model_name}: AI Text Humanization with LoRA}},
  year = {{2025}},
  publisher = {{GitHub/HuggingFace}},
  howpublished = {{\\url{{https://your-repo-url}}}}
}
```

## License

This model is released under the MIT License.

## Contact

For questions or feedback, please open an issue in the repository.

---

*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        model_card_path = os.path.join(self.output_dir, "README.md")
        
        try:
            with open(model_card_path, 'w', encoding='utf-8') as f:
                f.write(model_card)
            
            logger.info(f"✓ Model card saved to {model_card_path}")
            
        except Exception as e:
            logger.error(f"Error creating model card: {str(e)}")
            raise
    
    def save_config_yaml(self) -> None:
        """Save configuration as YAML for easy reading"""
        logger.info("Saving configuration (YAML)...")
        
        config = {
            'model_info': {
                'name': self.metadata.model_name,
                'base_model': self.metadata.base_model,
                'type': self.metadata.model_type,
                'task': self.metadata.task,
                'language': self.metadata.language
            },
            'lora_config': {
                'rank': self.metadata.lora_r,
                'alpha': self.metadata.lora_alpha,
                'dropout': self.metadata.lora_dropout,
                'target_modules': self.metadata.target_modules,
                'trainable_params': self.metadata.trainable_params,
                'trainable_percentage': self.metadata.trainable_params_percentage
            },
            'training_info': {
                'date': self.metadata.training_date,
                'epochs': self.metadata.num_epochs,
                'steps': self.metadata.total_steps,
                'final_train_loss': self.metadata.final_train_loss,
                'final_val_loss': self.metadata.final_val_loss,
                'best_val_loss': self.metadata.best_val_loss,
                'device': self.metadata.device_used,
                'training_hours': self.metadata.training_time_hours
            },
            'dataset_info': {
                'train_size': self.metadata.train_dataset_size,
                'val_size': self.metadata.val_dataset_size,
                'test_size': self.metadata.test_dataset_size
            },
            'framework_versions': {
                'transformers': self.metadata.transformers_version,
                'peft': self.metadata.peft_version,
                'pytorch': self.metadata.framework_version
            }
        }
        
        config_path = os.path.join(self.output_dir, "config.yaml")
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"✓ Configuration saved to {config_path}")
            
        except ImportError:
            logger.warning("PyYAML not installed. Skipping YAML config.")
        except Exception as e:
            logger.error(f"Error saving YAML config: {str(e)}")
    
    def save_usage_example(self) -> None:
        """Save a Python script with usage example"""
        logger.info("Creating usage example script...")
        
        usage_script = f'''"""
Usage Example for {self.metadata.model_name}
This script demonstrates how to load and use the trained model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(model_path: str = "{self.output_dir}"):
    """Load the trained model with LoRA adapters"""
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "{self.metadata.base_model}",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("{self.metadata.base_model}")
    
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    print("Model loaded successfully!")
    return model, tokenizer


def humanize_text(
    model,
    tokenizer,
    text: str,
    max_length: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.9
) -> str:
    """Humanize AI-generated text"""
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {{k: v.to(model.device) for k, v in inputs.items()}}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    humanized = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return humanized


def main():
    """Main execution"""
    
    # Load model
    model, tokenizer = load_model()
    
    # Example AI text
    ai_text = """
    Artificial intelligence is transforming various industries. 
    Machine learning algorithms enable automated decision-making. 
    Companies are adopting these technologies to improve efficiency.
    """
    
    print("\\nOriginal AI Text:")
    print("-" * 60)
    print(ai_text)
    
    # Humanize
    print("\\nHumanizing text...")
    humanized = humanize_text(model, tokenizer, ai_text)
    
    print("\\nHumanized Text:")
    print("-" * 60)
    print(humanized)


if __name__ == "__main__":
    main()
'''
        
        example_path = os.path.join(self.output_dir, "usage_example.py")
        
        try:
            with open(example_path, 'w', encoding='utf-8') as f:
                f.write(usage_script)
            
            logger.info(f"✓ Usage example saved to {example_path}")
            
        except Exception as e:
            logger.error(f"Error saving usage example: {str(e)}")
    
    def create_requirements_file(self) -> None:
        """Create requirements.txt for model dependencies"""
        logger.info("Creating requirements.txt...")
        
        requirements = f"""# Model Requirements
transformers>={self.metadata.transformers_version}
peft>={self.metadata.peft_version}
torch>=2.0.0
accelerate>=0.20.0
bitsandbytes>=0.40.0
"""
        
        req_path = os.path.join(self.output_dir, "requirements.txt")
        
        try:
            with open(req_path, 'w', encoding='utf-8') as f:
                f.write(requirements)
            
            logger.info(f"✓ Requirements saved to {req_path}")
            
        except Exception as e:
            logger.error(f"Error saving requirements: {str(e)}")
    
    def save_all(self) -> None:
        """Save everything - convenience method"""
        logger.info("\n" + "="*60)
        logger.info("SAVING MODEL AND METADATA")
        logger.info("="*60 + "\n")
        
        try:
            # Save LoRA adapters
            self.save_lora_adapters()
            
            # Save tokenizer
            self.save_tokenizer()
            
            # Save metadata
            self.save_metadata_json()
            
            # Create model card
            self.create_model_card()
            
            # Save YAML config
            self.save_config_yaml()
            
            # Save usage example
            self.save_usage_example()
            
            # Create requirements file
            self.create_requirements_file()
            
            # Print summary
            self._print_summary()
            
            logger.info("\n" + "="*60)
            logger.info("MODEL SAVED SUCCESSFULLY!")
            logger.info("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"Error during saving: {str(e)}")
            raise
    
    def _print_summary(self) -> None:
        """Print summary of saved files"""
        logger.info("\n📁 Saved Files Summary:")
        logger.info("-" * 60)
        
        # Calculate total size
        total_size = 0
        file_list = []
        
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                total_size += size
                rel_path = os.path.relpath(file_path, self.output_dir)
                file_list.append((rel_path, size))
        
        # Sort by size
        file_list.sort(key=lambda x: x[1], reverse=True)
        
        # Print files
        for file_name, size in file_list:
            if size > 1024 * 1024:
                size_str = f"{size / (1024 * 1024):.2f} MB"
            else:
                size_str = f"{size / 1024:.2f} KB"
            
            logger.info(f"  {file_name:<40} {size_str:>10}")
        
        logger.info("-" * 60)
        logger.info(f"  Total Size: {total_size / (1024 * 1024):.2f} MB")
        logger.info(f"  Location: {self.output_dir}")


def main():
    """Example usage"""
    
    # This would typically be called after training
    # For demonstration, we'll show how to use it
    
    print("""
    This script should be called after training with the trained model.
    
    Example usage:
    
    from model_saving_script import ModelSaver, ModelMetadata
    
    # Create metadata
    metadata = ModelMetadata(
        model_name="humanizer-v1",
        base_model="gpt2-medium",
        model_type="causal-lm",
        num_epochs=3,
        total_steps=1500,
        final_train_loss=2.34,
        final_val_loss=2.45,
        best_val_loss=2.41,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],
        trainable_params=2_949_120,
        total_params=354_823_168,
        trainable_params_percentage=0.83,
        train_dataset_size=8000,
        val_dataset_size=1000,
        test_dataset_size=1000,
        device_used="cuda",
        training_time_hours=2.5
    )
    
    # Initialize saver (assuming model and tokenizer are already loaded)
    saver = ModelSaver(
        model=trained_model,
        tokenizer=tokenizer,
        output_dir="models/humanizer_final",
        metadata=metadata
    )
    
    # Save everything
    saver.save_all()
    """)


if __name__ == "__main__":
    main()