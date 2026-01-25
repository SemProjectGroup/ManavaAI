import argparse
import os
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType
from .config import TrainingConfig
from .dataset import HumanizerDataset
from .metrics import HumanizerMetrics

def train(dry_run=False):
    conf = TrainingConfig()
    
    print(f"Loading Model: {conf.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(conf.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(conf.model_name)
    
    if conf.use_peft:
        print("Applying LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, 
            inference_mode=False, 
            r=conf.lora_r, 
            lora_alpha=conf.lora_alpha, 
            lora_dropout=conf.lora_dropout
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Load Datasets
    print("Loading Datasets...")
    train_dataset = HumanizerDataset(conf.train_file, tokenizer)
    val_dataset = HumanizerDataset(conf.val_file, tokenizer)

    # If dry run, use dummy data if real data is empty
    if dry_run or len(train_dataset) == 0:
        print("Dry Run mode or Empty Dataset: Using dummy data for initialization check.")
        # TODO: Implement robust dummy data generation if needed
        # For now, we assume at least one file exists or we skip training
        if len(train_dataset) == 0:
            print("No training data found. Exiting (Setup Complete).")
            return

    # Metrics
    metrics_handler = HumanizerMetrics(device=conf.device)
    
    def compute_metrics(eval_preds):
        return metrics_handler.compute_metrics(eval_preds, tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=conf.output_dir,
        learning_rate=conf.learning_rate,
        per_device_train_batch_size=conf.batch_size,
        per_device_eval_batch_size=conf.batch_size,
        num_train_epochs=1 if dry_run else conf.num_epochs,
        weight_decay=conf.weight_decay,
        evaluation_strategy="steps",
        eval_steps=conf.eval_steps,
        save_steps=conf.save_steps,
        logging_steps=conf.logging_steps,
        predict_with_generate=True,
        fp16=(conf.mixed_precision == "fp16"),
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
    )

    print("Starting Training..." if not dry_run else "Dry Run Check Complete. Ready to Train.")
    if not dry_run:
        trainer.train()
        model.save_pretrained(conf.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Initialize only, do not train.")
    args = parser.parse_args()
    train(dry_run=args.dry_run)
