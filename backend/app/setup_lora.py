"""
LoRA Configuration Setup for AI Text Humanizer Project
Defines LoRA parameters and training hyperparameters for model fine-tuning.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer,
        TrainingArguments,
        get_linear_schedule_with_warmup,
        get_cosine_schedule_with_warmup
    )
    from peft import (
        LoraConfig, 
        get_peft_model, 
        TaskType,
        PeftModel,
        prepare_model_for_kbit_training
    )
    import torch
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install transformers peft torch accelerate bitsandbytes")
    exit(1)


class LoRAConfigurator:
    """Handles LoRA configuration and model setup."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.lora_config = None
        self.training_args = None
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('lora_setup_log.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_target_modules(self, model_type: str) -> List[str]:
        """Get target modules for LoRA based on model type."""
        
        # GPT-2 models
        if 'gpt2' in model_type.lower():
            return ['c_attn', 'c_proj', 'c_fc']  # Query, Key, Value projections + MLP
        
        # Mistral models
        elif 'mistral' in model_type.lower():
            return ['q_proj', 'k_proj', 'v_proj', 'o_proj']  # All attention projections
        
        # LLaMA/LLaMA-2 models
        elif 'llama' in model_type.lower():
            return ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        
        # Default: target all linear layers in attention
        else:
            self.logger.warning(f"Unknown model type: {model_type}, using default target modules")
            return ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    def create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration."""
        
        self.logger.info("Creating LoRA configuration...")
        
        # Get target modules based on model
        target_modules = self.get_target_modules(self.args.model_name)
        
        self.logger.info(f"Target modules: {target_modules}")
        self.logger.info(f"LoRA rank (r): {self.args.lora_r}")
        self.logger.info(f"LoRA alpha: {self.args.lora_alpha}")
        self.logger.info(f"LoRA dropout: {self.args.lora_dropout}")
        
        lora_config = LoraConfig(
            r=self.args.lora_r,                          # Rank of LoRA matrices
            lora_alpha=self.args.lora_alpha,            # Scaling factor
            target_modules=target_modules,               # Modules to apply LoRA
            lora_dropout=self.args.lora_dropout,        # Dropout probability
            bias=self.args.lora_bias,                   # Bias training strategy
            task_type=TaskType.CAUSAL_LM,               # Task type
            inference_mode=False,                        # Training mode
            modules_to_save=self.args.modules_to_save   # Additional modules to train
        )
        
        self.logger.info("LoRA configuration created successfully")
        return lora_config
    
    def load_base_model(self):
        """Load base model with optional quantization."""
        
        self.logger.info(f"Loading base model: {self.args.model_name}")
        
        # Setup model loading kwargs
        model_kwargs = {
            'cache_dir': self.args.cache_dir,
            'trust_remote_code': True,
        }
        
        # Add quantization config if enabled
        if self.args.use_8bit or self.args.use_4bit:
            from transformers import BitsAndBytesConfig
            
            if self.args.use_4bit:
                self.logger.info("Using 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.args.use_8bit:
                self.logger.info("Using 8-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            
            model_kwargs['quantization_config'] = quantization_config
            model_kwargs['device_map'] = 'auto'
        else:
            model_kwargs['torch_dtype'] = torch.float16 if torch.cuda.is_available() else torch.float32
        
        try:
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                **model_kwargs
            )
            
            # Prepare model for k-bit training if quantized
            if self.args.use_8bit or self.args.use_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            self.logger.info(f"Model loaded successfully")
            self.logger.info(f"Model device: {next(self.model.parameters()).device}")
            self.logger.info(f"Model dtype: {next(self.model.parameters()).dtype}")
            
            # Print model architecture summary
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logger.info(f"Total parameters: {total_params:,}")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def load_tokenizer(self):
        """Load tokenizer."""
        
        self.logger.info(f"Loading tokenizer: {self.args.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name,
                cache_dir=self.args.cache_dir,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            self.logger.info("Tokenizer loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def apply_lora(self):
        """Apply LoRA adapters to the model."""
        
        self.logger.info("Applying LoRA adapters to model...")
        
        try:
            self.model = get_peft_model(self.model, self.lora_config)
            
            # Print trainable parameters
            self.model.print_trainable_parameters()
            
            self.logger.info("LoRA adapters applied successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to apply LoRA: {e}")
            raise
    
    def create_training_args(self) -> TrainingArguments:
        """Create training arguments."""
        
        self.logger.info("Creating training arguments...")
        
        # Create output directory
        output_dir = Path(self.args.output_dir) / 'training_output'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            # Output
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            
            # Training hyperparameters
            num_train_epochs=self.args.num_epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            
            # Learning rate
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            warmup_ratio=self.args.warmup_ratio,
            lr_scheduler_type=self.args.lr_scheduler_type,
            
            # Optimization
            optim=self.args.optimizer,
            max_grad_norm=self.args.max_grad_norm,
            
            # Mixed precision
            fp16=self.args.fp16 and torch.cuda.is_available(),
            bf16=self.args.bf16 and torch.cuda.is_available(),
            
            # Logging
            logging_dir=str(output_dir / 'logs'),
            logging_steps=self.args.logging_steps,
            logging_first_step=True,
            
            # Evaluation
            evaluation_strategy=self.args.evaluation_strategy,
            eval_steps=self.args.eval_steps if self.args.evaluation_strategy == 'steps' else None,
            
            # Saving
            save_strategy=self.args.save_strategy,
            save_steps=self.args.save_steps if self.args.save_strategy == 'steps' else None,
            save_total_limit=self.args.save_total_limit,
            load_best_model_at_end=self.args.load_best_model_at_end,
            
            # Other
            seed=self.args.seed,
            dataloader_num_workers=self.args.num_workers,
            remove_unused_columns=False,
            report_to=self.args.report_to,
            
            # Gradient checkpointing for memory efficiency
            gradient_checkpointing=self.args.gradient_checkpointing,
        )
        
        self.logger.info("Training arguments created successfully")
        return training_args
    
    def save_configuration(self):
        """Save all configurations to file."""
        
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_name': self.args.model_name,
                'model_type': 'gpt2' if 'gpt2' in self.args.model_name.lower() else 'mistral',
                'quantization': {
                    'use_4bit': self.args.use_4bit,
                    'use_8bit': self.args.use_8bit,
                },
            },
            'lora_config': {
                'r': self.args.lora_r,
                'lora_alpha': self.args.lora_alpha,
                'lora_dropout': self.args.lora_dropout,
                'target_modules': self.get_target_modules(self.args.model_name),
                'bias': self.args.lora_bias,
                'task_type': 'CAUSAL_LM',
                'modules_to_save': self.args.modules_to_save
            },
            'training_hyperparameters': {
                'num_epochs': self.args.num_epochs,
                'batch_size': self.args.batch_size,
                'gradient_accumulation_steps': self.args.gradient_accumulation_steps,
                'effective_batch_size': self.args.batch_size * self.args.gradient_accumulation_steps,
                'learning_rate': self.args.learning_rate,
                'weight_decay': self.args.weight_decay,
                'warmup_ratio': self.args.warmup_ratio,
                'lr_scheduler_type': self.args.lr_scheduler_type,
                'optimizer': self.args.optimizer,
                'max_grad_norm': self.args.max_grad_norm,
                'seed': self.args.seed
            },
            'mixed_precision': {
                'fp16': self.args.fp16,
                'bf16': self.args.bf16
            },
            'logging_and_saving': {
                'logging_steps': self.args.logging_steps,
                'eval_steps': self.args.eval_steps,
                'save_steps': self.args.save_steps,
                'save_total_limit': self.args.save_total_limit,
                'evaluation_strategy': self.args.evaluation_strategy,
                'save_strategy': self.args.save_strategy
            },
            'memory_optimization': {
                'gradient_checkpointing': self.args.gradient_checkpointing
            }
        }
        
        # Save as JSON
        config_path = output_dir / 'lora_training_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Configuration saved to: {config_path}")
        
        # Save LoRA config separately
        if self.lora_config:
            lora_config_path = output_dir / 'lora_config.json'
            self.lora_config.save_pretrained(str(output_dir))
            self.logger.info(f"LoRA config saved to: {output_dir}")
        
        return config
    
    def print_configuration_summary(self, config: Dict):
        """Print configuration summary."""
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info("LORA CONFIGURATION SUMMARY")
        self.logger.info(f"{'='*70}")
        
        self.logger.info(f"\n📦 MODEL INFORMATION")
        self.logger.info(f"  Model: {config['model_info']['model_name']}")
        self.logger.info(f"  Quantization: {'4-bit' if config['model_info']['quantization']['use_4bit'] else '8-bit' if config['model_info']['quantization']['use_8bit'] else 'None'}")
        
        self.logger.info(f"\n🔧 LORA PARAMETERS")
        self.logger.info(f"  Rank (r): {config['lora_config']['r']}")
        self.logger.info(f"  Alpha: {config['lora_config']['lora_alpha']}")
        self.logger.info(f"  Dropout: {config['lora_config']['lora_dropout']}")
        self.logger.info(f"  Target modules: {', '.join(config['lora_config']['target_modules'])}")
        self.logger.info(f"  Bias: {config['lora_config']['bias']}")
        
        self.logger.info(f"\n📊 TRAINING HYPERPARAMETERS")
        self.logger.info(f"  Epochs: {config['training_hyperparameters']['num_epochs']}")
        self.logger.info(f"  Batch size: {config['training_hyperparameters']['batch_size']}")
        self.logger.info(f"  Gradient accumulation: {config['training_hyperparameters']['gradient_accumulation_steps']}")
        self.logger.info(f"  Effective batch size: {config['training_hyperparameters']['effective_batch_size']}")
        self.logger.info(f"  Learning rate: {config['training_hyperparameters']['learning_rate']}")
        self.logger.info(f"  Weight decay: {config['training_hyperparameters']['weight_decay']}")
        self.logger.info(f"  Warmup ratio: {config['training_hyperparameters']['warmup_ratio']}")
        self.logger.info(f"  LR scheduler: {config['training_hyperparameters']['lr_scheduler_type']}")
        self.logger.info(f"  Optimizer: {config['training_hyperparameters']['optimizer']}")
        
        self.logger.info(f"\n💾 LOGGING & SAVING")
        self.logger.info(f"  Logging steps: {config['logging_and_saving']['logging_steps']}")
        self.logger.info(f"  Evaluation strategy: {config['logging_and_saving']['evaluation_strategy']}")
        self.logger.info(f"  Save strategy: {config['logging_and_saving']['save_strategy']}")
        self.logger.info(f"  Save total limit: {config['logging_and_saving']['save_total_limit']}")
        
        self.logger.info(f"\n⚡ OPTIMIZATION")
        self.logger.info(f"  FP16: {config['mixed_precision']['fp16']}")
        self.logger.info(f"  BF16: {config['mixed_precision']['bf16']}")
        self.logger.info(f"  Gradient checkpointing: {config['memory_optimization']['gradient_checkpointing']}")
        
        # Calculate estimated GPU memory
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            self.logger.info(f"\n📈 MODEL STATISTICS")
            self.logger.info(f"  Total parameters: {total_params:,}")
            self.logger.info(f"  Trainable parameters: {trainable_params:,}")
            self.logger.info(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
            
            # Rough memory estimate
            param_memory = (trainable_params * 4) / (1024**3)  # 4 bytes per float32 param
            self.logger.info(f"  Est. trainable param memory: {param_memory:.2f} GB")
        
        self.logger.info(f"\n{'='*70}")
    
    def setup(self):
        """Main setup pipeline."""
        
        self.logger.info("Starting LoRA configuration setup...")
        
        # Create LoRA config
        self.lora_config = self.create_lora_config()
        
        # Load tokenizer
        self.load_tokenizer()
        
        # Load base model
        if not self.args.config_only:
            self.load_base_model()
            
            # Apply LoRA
            self.apply_lora()
        
        # Create training arguments
        self.training_args = self.create_training_args()
        
        # Save configuration
        config = self.save_configuration()
        
        # Print summary
        self.print_configuration_summary(config)
        
        # Save model if requested
        if not self.args.config_only and self.args.save_model:
            output_dir = Path(self.args.output_dir) / 'model_with_lora'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Saving model with LoRA to {output_dir}")
            self.model.save_pretrained(str(output_dir))
            self.tokenizer.save_pretrained(str(output_dir))
            self.logger.info("Model saved successfully")
        
        self.logger.info("\n✓ LoRA configuration setup completed successfully!")
        
        return self.model, self.tokenizer, self.lora_config, self.training_args


def main():
    parser = argparse.ArgumentParser(description='Setup LoRA configuration for fine-tuning')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, required=True,
                        help='Base model name (e.g., gpt2-medium, mistralai/Mistral-7B-v0.1)')
    parser.add_argument('--cache_dir', type=str, default='./model_cache',
                        help='Cache directory for model downloads')
    
    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank (r). Higher = more parameters. Recommended: 8-64')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha (scaling). Usually 2x rank. Recommended: 16-64')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout probability. Recommended: 0.05-0.1')
    parser.add_argument('--lora_bias', type=str, default='none',
                        choices=['none', 'all', 'lora_only'],
                        help='Bias training strategy')
    parser.add_argument('--modules_to_save', type=str, nargs='*', default=None,
                        help='Additional modules to train (e.g., embed_tokens, lm_head)')
    
    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size per device')
    parser.add_argument('--eval_batch_size', type=int, default=4,
                        help='Evaluation batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps (effective_batch = batch_size * this)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate. Recommended: 1e-4 to 5e-4')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for regularization')
    parser.add_argument('--warmup_ratio', type=float, default=0.03,
                        help='Warmup ratio (fraction of total steps)')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                        choices=['linear', 'cosine', 'polynomial', 'constant'],
                        help='Learning rate scheduler type')
    parser.add_argument('--optimizer', type=str, default='adamw_torch',
                        help='Optimizer type')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    
    # Quantization
    parser.add_argument('--use_4bit', action='store_true',
                        help='Use 4-bit quantization (QLoRA)')
    parser.add_argument('--use_8bit', action='store_true',
                        help='Use 8-bit quantization')
    
    # Mixed precision
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use FP16 mixed precision')
    parser.add_argument('--bf16', action='store_true',
                        help='Use BF16 mixed precision (better than FP16 if supported)')
    
    # Logging and evaluation
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--evaluation_strategy', type=str, default='steps',
                        choices=['no', 'steps', 'epoch'],
                        help='Evaluation strategy')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='Evaluate every N steps')
    
    # Saving
    parser.add_argument('--save_strategy', type=str, default='steps',
                        choices=['no', 'steps', 'epoch'],
                        help='Save strategy')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every N steps')
    parser.add_argument('--save_total_limit', type=int, default=3,
                        help='Maximum number of checkpoints to keep')
    parser.add_argument('--load_best_model_at_end', action='store_true', default=True,
                        help='Load best model at end of training')
    
    # Memory optimization
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                        help='Use gradient checkpointing to save memory')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for configs and model')
    parser.add_argument('--config_only', action='store_true',
                        help='Only create config files, do not load model')
    parser.add_argument('--save_model', action='store_true',
                        help='Save model with LoRA adapters applied')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--report_to', type=str, nargs='*', default=['tensorboard'],
                        help='Reporting tools (tensorboard, wandb, etc.)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Validate quantization options
    if args.use_4bit and args.use_8bit:
        parser.error("Cannot use both 4-bit and 8-bit quantization")
    
    # Create configurator and setup
    configurator = LoRAConfigurator(args)
    configurator.setup()


if __name__ == '__main__':
    main()