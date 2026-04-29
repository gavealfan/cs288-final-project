eu"""
Local SFT training entrypoint for Apple Silicon / laptop workflows.

Designed to run on Mac (MPS) without bitsandbytes/QLoRA dependencies.
Uses LoRA adapters on top of the selected base model.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from trl import SFTTrainer


@dataclass
class ScriptArguments:
    model_name: str = field(default="Qwen/Qwen2.5-3B-Instruct")
    train_path: str = field(default="data/teacher_sft_train.jsonl")
    eval_path: str = field(default="data/teacher_sft_eval.jsonl")
    output_dir: str = field(default="checkpoints/role-conditioned-sft-local")
    num_train_epochs: int = field(default=1)
    max_steps: int = field(default=200)
    learning_rate: float = field(default=2e-5)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    eval_steps: int = field(default=50)
    save_steps: int = field(default=100)
    logging_steps: int = field(default=10)
    max_train_samples: int = field(default=0)
    max_eval_samples: int = field(default=0)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)


def load_sft_data(path: str) -> Dataset:
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append({"text": d["prompt"] + "\n\n" + d["response"]})
    return Dataset.from_list(rows)


def infer_target_modules(model_name: str) -> list[str]:
    lower = model_name.lower()
    if "gpt2" in lower:
        return ["c_attn", "c_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def main() -> None:
    args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]

    train_dataset = load_sft_data(args.train_path)
    eval_dataset = load_sft_data(args.eval_path) if Path(args.eval_path).exists() else None
    if eval_dataset is not None and len(eval_dataset) == 0:
        eval_dataset = None

    if args.max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if eval_dataset is not None and args.max_eval_samples > 0:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if (use_cuda or use_mps) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    if use_mps:
        model = model.to("mps")
    elif use_cuda:
        model = model.to("cuda")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=infer_target_modules(args.model_name),
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_steps=args.save_steps,
        fp16=use_cuda,
        bf16=False,
        report_to="none",
        gradient_checkpointing=False,
        optim="adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        formatting_func=lambda x: x["text"],
    )

    print(
        f"Training local SFT | device={'mps' if use_mps else ('cuda' if use_cuda else 'cpu')} "
        f"| train={len(train_dataset)} eval={len(eval_dataset) if eval_dataset is not None else 0}"
    )
    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved SFT adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
