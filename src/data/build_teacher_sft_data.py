"""
Build teacher-supervised SFT data and preference pairs from role-conditioned prompts.

For each scenario, this script:
1) Builds the role-conditioned prompt
2) Generates a base response (e.g., Qwen2.5-3B)
3) Generates a teacher response (e.g., Qwen2.5-7B)
4) Writes:
   - SFT records: {"prompt": ..., "response": teacher_response}
   - Preference records: {"prompt": ..., "chosen": teacher_response, "rejected": base_response}
"""

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from transformers import HfArgumentParser


PROMPT_TEMPLATE = """You are in the following organizational role:
{role_description}

Interaction context:
{context_description}

Situation: {situation}
Background: {background}

Respond appropriately given your role and the situation."""


def strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def call_openrouter(
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def build_prompt(scenario: dict) -> str:
    return PROMPT_TEMPLATE.format(
        role_description=scenario["role_description"],
        context_description=scenario["context_description"],
        situation=scenario["situation"],
        background=scenario["background"],
    )


@dataclass
class ScriptArguments:
    input_path: str = field(default="data/evaluated.jsonl")
    output_sft_train: str = field(default="data/teacher_sft_train.jsonl")
    output_sft_eval: str = field(default="data/teacher_sft_eval.jsonl")
    output_pref_train: str = field(default="data/teacher_pref_train.jsonl")
    output_pref_eval: str = field(default="data/teacher_pref_eval.jsonl")
    base_model: str = field(default="qwen/qwen2.5-3b-instruct")
    teacher_model: str = field(default="qwen/qwen3.5-397b-a17b")
    base_system_prompt: str = field(default="")
    teacher_system_prompt: str = field(default="")
    temperature: float = field(default=0.7)
    max_scenarios: int = field(default=0, metadata={"help": "0 means all"})
    seed: int = field(default=42)


def main() -> None:
    args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Set it before running this script."
        )

    scenarios = []
    with open(args.input_path) as f:
        for line in f:
            if line.strip():
                scenarios.append(json.loads(line))

    random.seed(args.seed)
    if args.max_scenarios > 0 and len(scenarios) > args.max_scenarios:
        scenarios = random.sample(scenarios, args.max_scenarios)

    for p in [
        args.output_sft_train,
        args.output_sft_eval,
        args.output_pref_train,
        args.output_pref_eval,
    ]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    sft_train = open(args.output_sft_train, "w")
    sft_eval = open(args.output_sft_eval, "w")
    pref_train = open(args.output_pref_train, "w")
    pref_eval = open(args.output_pref_eval, "w")

    processed = 0
    for idx, scenario in enumerate(scenarios, start=1):
        try:
            prompt = build_prompt(scenario)
            split = scenario.get("split", "train")

            base_resp = call_openrouter(
                prompt=prompt,
                model=args.base_model,
                system_prompt=args.base_system_prompt or None,
                temperature=args.temperature,
            )
            teacher_resp = call_openrouter(
                prompt=prompt,
                model=args.teacher_model,
                system_prompt=args.teacher_system_prompt or None,
                temperature=args.temperature,
            )

            sft_record = {
                "prompt": prompt,
                "response": teacher_resp,
                "metadata": {
                    "split": split,
                    "base_model": args.base_model,
                    "teacher_model": args.teacher_model,
                    "source_index": idx - 1,
                },
            }
            pref_record = {
                "prompt": prompt,
                "chosen": teacher_resp,
                "rejected": base_resp,
                "metadata": {
                    "split": split,
                    "base_model": args.base_model,
                    "teacher_model": args.teacher_model,
                    "source_index": idx - 1,
                },
            }

            if split == "holdout":
                sft_eval.write(json.dumps(sft_record) + "\n")
                pref_eval.write(json.dumps(pref_record) + "\n")
            else:
                sft_train.write(json.dumps(sft_record) + "\n")
                pref_train.write(json.dumps(pref_record) + "\n")

            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed} scenarios")
        except Exception as exc:
            print(f"Error on scenario {idx}: {exc}")
            if "not a valid model ID" in str(exc):
                raise RuntimeError(
                    f"Invalid OpenRouter model id in use. "
                    f"base_model={args.base_model}, teacher_model={args.teacher_model}"
                ) from exc

    sft_train.close()
    sft_eval.close()
    pref_train.close()
    pref_eval.close()

    print(f"Done. Built data from {processed} scenarios.")
    if processed == 0:
        raise RuntimeError("No records were built. Check API key/model access and rerun.")
    print(f"SFT train/eval: {args.output_sft_train}, {args.output_sft_eval}")
    print(f"Pref train/eval: {args.output_pref_train}, {args.output_pref_eval}")


if __name__ == "__main__":
    main()
