"""
Colab-friendly inference helper for testing the role-conditioned LoRA adapter.

Example usage in Google Colab:

    !pip install -q transformers>=4.40 peft>=0.10 accelerate>=0.28 bitsandbytes>=0.43 sentencepiece
    !python test_colab.py --adapter_zip /content/role_conditioned_dpo_lora.zip --scenario_path /content/test_scenarios.jsonl --num_examples 3

If you do not pass --scenario_path, the script runs on a built-in sample prompt.
"""

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROMPT_TEMPLATE = """You are in the following organizational role:
{role_description}

Interaction context:
{context_description}

Situation: {situation}
Background: {background}

Respond appropriately given your role and the situation."""


DEFAULT_SCENARIO = {
    "role_description": (
        "You are a C-level executive in operations, who has no direct authority "
        "over the person they are speaking with (cross-functional). You are a "
        "long-tenured employee with deep institutional knowledge."
    ),
    "context_description": (
        "You are mentoring a more junior colleague. Provide guidance, share "
        "experience, and support their development."
    ),
    "situation": (
        "Sarah Chen, a Senior Data Analyst in the Marketing department, presented "
        "a forecast to the cross-functional project team for the 'Evergreen "
        "Initiative' that was significantly more pessimistic than previous "
        "models. You offered to meet with her to understand her methodology and "
        "share insights from your years at the company."
    ),
    "background": (
        "The 'Evergreen Initiative' is a company-wide sustainability effort with "
        "a major Q4 launch. Sarah's forecast could change marketing budget "
        "allocation and project strategy. Previous forecasts were more "
        "optimistic, but Sarah incorporated external consumer-sentiment signals. "
        "You have been at the company for 18 years and have seen similar "
        "initiatives succeed or stall depending on how the data is interpreted."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model used for the adapter.",
    )
    parser.add_argument(
        "--adapter_zip",
        default=None,
        help="Path to a zipped LoRA adapter, such as role_conditioned_dpo_lora.zip.",
    )
    parser.add_argument(
        "--adapter_dir",
        default=None,
        help="Path to an extracted adapter directory containing adapter_config.json.",
    )
    parser.add_argument(
        "--scenario_path",
        default=None,
        help="Optional JSONL file with scenarios to test.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1,
        help="How many scenarios to run from the JSONL file.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=220,
        help="Max tokens to generate per response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    return parser.parse_args()


def build_prompt(scenario: dict) -> str:
    return PROMPT_TEMPLATE.format(
        role_description=scenario["role_description"],
        context_description=scenario["context_description"],
        situation=scenario["situation"],
        background=scenario["background"],
    )


def load_scenarios(path: str | None, limit: int) -> list[dict]:
    if not path:
        return [DEFAULT_SCENARIO]

    scenarios = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            scenarios.append(json.loads(line))
            if len(scenarios) >= limit:
                break

    if not scenarios:
        raise ValueError(f"No scenarios found in {path}")

    return scenarios


def find_adapter_dir(root: Path) -> Path:
    if (root / "adapter_config.json").exists():
        return root

    for candidate in root.rglob("adapter_config.json"):
        return candidate.parent

    raise FileNotFoundError(
        f"Could not find adapter_config.json under {root}. "
        "Make sure you uploaded the LoRA adapter zip or extracted directory."
    )


def resolve_adapter_dir(adapter_zip: str | None, adapter_dir: str | None) -> tuple[Path, Path | None]:
    if adapter_dir:
        path = find_adapter_dir(Path(adapter_dir))
        return path, None

    if not adapter_zip:
        raise ValueError("Pass either --adapter_zip or --adapter_dir.")

    temp_dir = Path(tempfile.mkdtemp(prefix="colab_lora_"))
    with zipfile.ZipFile(adapter_zip) as zf:
        zf.extractall(temp_dir)
    return find_adapter_dir(temp_dir), temp_dir


def load_tokenizer(base_model: str, adapter_dir: Path):
    tokenizer_source = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(base_model: str):
    quant_config = None
    torch_dtype = torch.float32

    if torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        torch_dtype = torch.bfloat16

    return AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        quantization_config=quant_config,
    )


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    adapter_dir, temp_dir = resolve_adapter_dir(args.adapter_zip, args.adapter_dir)

    print(f"Base model: {args.base_model}")
    print(f"Adapter dir: {adapter_dir}")

    tokenizer = load_tokenizer(args.base_model, adapter_dir)
    base_model = load_base_model(args.base_model)
    base_model.eval()

    scenarios = load_scenarios(args.scenario_path, args.num_examples)

    print("\nGenerating baseline responses...")
    baseline_outputs = []
    for scenario in scenarios:
        prompt = build_prompt(scenario)
        baseline_outputs.append(
            generate_response(
                base_model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        )

    tuned_model = PeftModel.from_pretrained(base_model, adapter_dir)
    tuned_model.eval()

    print("Generating adapter responses...")
    tuned_outputs = []
    for scenario in scenarios:
        prompt = build_prompt(scenario)
        tuned_outputs.append(
            generate_response(
                tuned_model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        )

    for idx, scenario in enumerate(scenarios, start=1):
        print("\n" + "=" * 80)
        print(f"Example {idx}")
        print("-" * 80)
        print("Role:", scenario["role_description"])
        print("Context:", scenario["context_description"])
        print("Situation:", scenario["situation"])
        print("Background:", scenario["background"])
        print("\nBaseline response:\n")
        print(baseline_outputs[idx - 1])
        print("\nAdapter response:\n")
        print(tuned_outputs[idx - 1])

    if temp_dir is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
