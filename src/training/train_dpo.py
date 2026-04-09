"""
DPO/IPO training script using HuggingFace TRL.

Trains a role-conditioned dialogue agent on preference pairs
generated from the LLM-as-judge evaluation pipeline.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model


@dataclass
class ScriptArguments:
    model_name: str = field(default="meta-llama/Llama-3.1-8B-Instruct")
    train_path: str = field(default="data/dpo_pairs_train.jsonl")
    eval_path: str = field(default="data/dpo_pairs_eval.jsonl")
    output_dir: str = field(default="checkpoints/role-conditioned-dpo")
    loss_type: str = field(default="sigmoid", metadata={"help": "DPO loss: sigmoid (DPO) or ipo"})
    beta: float = field(default=0.1, metadata={"help": "DPO beta parameter"})
    learning_rate: float = field(default=5e-7)
    num_epochs: int = field(default=3)
    per_device_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    max_length: int = field(default=1024)
    max_prompt_length: int = field(default=512)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    use_lora: bool = field(default=True)


def load_preference_data(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            records.append({
                "prompt": data["prompt"],
                "chosen": data["chosen"],
                "rejected": data["rejected"],
            })
    return Dataset.from_list(records)


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Load data
    print(f"Loading training data from {args.train_path}")
    train_dataset = load_preference_data(args.train_path)
    print(f"  Training examples: {len(train_dataset)}")

    eval_dataset = None
    if Path(args.eval_path).exists():
        eval_dataset = load_preference_data(args.eval_path)
        print(f"  Eval examples: {len(eval_dataset)}")

    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    # Apply LoRA if requested
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    # Configure DPO training
    training_args = DPOConfig(
        output_dir=args.output_dir,
        loss_type=args.loss_type,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=10,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=50 if eval_dataset else None,
        save_strategy="steps",
        save_steps=100,
        bf16=True,
        report_to="wandb",
        run_name="role-conditioned-dpo",
    )

    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    print("Starting DPO training...")
    trainer.train()

    # Save
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
