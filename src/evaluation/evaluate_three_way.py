"""
Three-way evaluation: Base vs SFT vs SFT+DPO.

This script evaluates three systems on holdout scenarios using an LLM judge.
Outputs both per-scenario judgments and aggregate summary JSON.
"""

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from transformers import HfArgumentParser


PROMPT_TEMPLATE = """You are in the following organizational role:
{role_description}

Interaction context:
{context_description}

Situation: {situation}
Background: {background}

Respond appropriately given your role and the situation."""


JUDGE_PROMPT = """You are an expert organizational behavior evaluator.

You are given one scenario and three candidate responses from different systems.
Evaluate each response independently on a 1-5 scale:
1) appropriateness
2) role_fidelity
3) collaborative_realism

Then rank systems from best to worst overall.

Scenario:
- role_description: {role_description}
- context_description: {context_description}
- situation: {situation}
- background: {background}

Response A:
{resp_a}

Response B:
{resp_b}

Response C:
{resp_c}

Return strict JSON:
{{
  "scores": {{
    "A": {{"appropriateness": <int>, "role_fidelity": <int>, "collaborative_realism": <int>}},
    "B": {{"appropriateness": <int>, "role_fidelity": <int>, "collaborative_realism": <int>}},
    "C": {{"appropriateness": <int>, "role_fidelity": <int>, "collaborative_realism": <int>}}
  }},
  "ranking": ["<A|B|C>", "<A|B|C>", "<A|B|C>"],
  "notes": "<brief rationale>"
}}"""


@dataclass
class ScriptArguments:
    eval_data_path: str = field(default="data/evaluated.jsonl")
    output_path: str = field(default="data/evaluation_three_way.jsonl")
    summary_path: str = field(default="data/evaluation_three_way_summary.json")
    base_model: str = field(default="mistralai/mistral-7b-instruct-v0.1")
    sft_model: str = field(default="mistralai/mistral-7b-instruct-v0.1")
    dpo_model: str = field(default="mistralai/mistral-7b-instruct-v0.1")
    base_system_prompt: str = field(default="")
    sft_system_prompt: str = field(default="")
    dpo_system_prompt: str = field(default="")
    judge_model: str = field(default="google/gemini-2.0-flash-001")
    max_scenarios: int = field(default=50)
    seed: int = field(default=42)


def strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def call_openrouter(prompt: str, model: str, system_prompt: str = "", temperature: float = 0.7) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable.")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    content = resp.choices[0].message.content
    if content is None or not str(content).strip():
        raise ValueError(f"Empty completion from model={model!r}")
    return str(content).strip()


def build_prompt(scenario: dict) -> str:
    return PROMPT_TEMPLATE.format(
        role_description=scenario["role_description"],
        context_description=scenario["context_description"],
        situation=scenario["situation"],
        background=scenario["background"],
    )


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Set it before running this script."
        )
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)

    scenarios = []
    with open(args.eval_data_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("split") == "holdout":
                scenarios.append(d)

    random.seed(args.seed)
    if len(scenarios) > args.max_scenarios:
        scenarios = random.sample(scenarios, args.max_scenarios)

    systems = ["base", "sft", "dpo"]
    wins = {k: 0 for k in systems}
    pairwise = {"base>sft": 0, "base>dpo": 0, "sft>dpo": 0}
    dimension_scores = {k: defaultdict(list) for k in systems}
    scenarios_with_full_ranking = 0

    with open(args.output_path, "w") as fout:
        for i, scenario in enumerate(scenarios, start=1):
            prompt = build_prompt(scenario)
            try:
                responses = {
                    "base": call_openrouter(prompt, args.base_model, args.base_system_prompt),
                    "sft": call_openrouter(prompt, args.sft_model, args.sft_system_prompt),
                    "dpo": call_openrouter(prompt, args.dpo_model, args.dpo_system_prompt),
                }

                mapping = [("A", "base"), ("B", "sft"), ("C", "dpo")]
                random.shuffle(mapping)
                letter_to_system = {letter: system for letter, system in mapping}

                judge_prompt = JUDGE_PROMPT.format(
                    role_description=scenario["role_description"],
                    context_description=scenario["context_description"],
                    situation=scenario["situation"],
                    background=scenario["background"],
                    resp_a=responses[letter_to_system["A"]],
                    resp_b=responses[letter_to_system["B"]],
                    resp_c=responses[letter_to_system["C"]],
                )
                raw_judgment = call_openrouter(
                    prompt=judge_prompt + "\nRespond with valid JSON only.",
                    model=args.judge_model,
                    temperature=0.1,
                )
                judgment = json.loads(strip_json_fences(raw_judgment))

                ranking_letters = judgment.get("ranking", [])
                ranking_systems = [letter_to_system[x] for x in ranking_letters if x in letter_to_system]
                if len(ranking_systems) == 3 and set(ranking_systems) == {"base", "sft", "dpo"}:
                    scenarios_with_full_ranking += 1
                    wins[ranking_systems[0]] += 1
                    ib = ranking_systems.index("base")
                    isft = ranking_systems.index("sft")
                    idpo = ranking_systems.index("dpo")
                    if ib < isft:
                        pairwise["base>sft"] += 1
                    if ib < idpo:
                        pairwise["base>dpo"] += 1
                    if isft < idpo:
                        pairwise["sft>dpo"] += 1

                for letter, score_dict in judgment.get("scores", {}).items():
                    if letter not in letter_to_system:
                        continue
                    system = letter_to_system[letter]
                    for dim in ["appropriateness", "role_fidelity", "collaborative_realism"]:
                        if dim in score_dict:
                            dimension_scores[system][dim].append(score_dict[dim])

                rec = {
                    "scenario_index": i - 1,
                    "split": scenario.get("split", "holdout"),
                    "responses": responses,
                    "letter_to_system": letter_to_system,
                    "judgment": judgment,
                    "ranking_systems": ranking_systems,
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                if i % 5 == 0:
                    print(f"Evaluated {i}/{len(scenarios)}", flush=True)
            except Exception as exc:
                print(f"Error on scenario {i}: {exc}", flush=True)

    denom = max(scenarios_with_full_ranking, 1)
    summary = {
        "total_scenarios": len(scenarios),
        "scenarios_with_full_ranking": scenarios_with_full_ranking,
        "winner_counts": wins,
        "winner_rates": {k: wins[k] / denom for k in wins},
        "pairwise_preference_rates": {
            "base_over_sft": pairwise["base>sft"] / denom,
            "base_over_dpo": pairwise["base>dpo"] / denom,
            "sft_over_dpo": pairwise["sft>dpo"] / denom,
        },
        "mean_scores": {
            system: {
                dim: mean(vals) for dim, vals in dims.items()
            }
            for system, dims in dimension_scores.items()
        },
    }
    with open(args.summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results: {args.output_path}")
    print(f"Saved summary: {args.summary_path}")


if __name__ == "__main__":
    main()
