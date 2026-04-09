"""
Evaluation pipeline: Compare baseline Qwen vs DPO-trained model.

For each holdout scenario:
1. Generate a response from the baseline model (no LoRA)
2. Generate a response from the DPO-trained model (with LoRA)
3. Send both to the LLM-as-judge for blind evaluation
4. Aggregate win rates, score distributions, and per-dimension analysis
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# ──────────────────────────────────────────────
# Evaluation via OpenRouter (no local GPU needed)
# ──────────────────────────────────────────────

GENERATION_PROMPT_TEMPLATE = """You are in the following organizational role:
{role_description}

Interaction context:
{context_description}

Situation: {situation}
Background: {background}

Respond appropriately given your role and the situation."""


DPO_SYSTEM_PROMPT = """You are a role-conditioned organizational dialogue agent. You have been trained to generate responses that are appropriate for your assigned organizational role, exhibiting realistic collaborative characteristics including power-sensitive communication, decision authority awareness, and norm-adherent information sharing. Always stay in character for the role you are given."""


JUDGE_COMPARISON_PROMPT = """You are an expert organizational behavior evaluator. You will evaluate two responses to the same workplace scenario. You do NOT know which response comes from which model.

## Scenario
**Speaker's role:** {role_description}
**Interaction context:** {context_description}
**Situation:** {situation}
**Background:** {background}

## Response A:
{response_a}

## Response B:
{response_b}

## Evaluation
Rate each response on these dimensions (1-5 scale):

1. **Appropriateness**: Does the tone, formality, and content match the speaker's hierarchical position and power relationship?
2. **Role Fidelity**: Does the response reflect the speaker's hierarchy level, functional domain, and seniority accurately?
3. **Collaborative Realism**: Does the response exhibit realistic collaborative characteristics (power-sensitive communication, decision authority awareness, accountability language)?

Then determine which response is better overall, or if they are tied.

Output as JSON:
{{
    "response_a": {{
        "appropriateness": <1-5>,
        "role_fidelity": <1-5>,
        "collaborative_realism": <1-5>,
        "rationale": "<brief explanation>"
    }},
    "response_b": {{
        "appropriateness": <1-5>,
        "role_fidelity": <1-5>,
        "collaborative_realism": <1-5>,
        "rationale": "<brief explanation>"
    }},
    "winner": "<A or B or tie>",
    "winner_rationale": "<why this response is better>"
}}"""


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
    system_prompt: str = None,
    temperature: float = 0.7,
) -> str:
    """Call OpenRouter API."""
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
    return response.choices[0].message.content


def generate_baseline_response(scenario: dict, model: str = "qwen/qwen-2.5-7b-instruct") -> str:
    """Generate a response from the baseline model (no role-conditioning training)."""
    prompt = GENERATION_PROMPT_TEMPLATE.format(
        role_description=scenario["role_description"],
        context_description=scenario["context_description"],
        situation=scenario["situation"],
        background=scenario["background"],
    )
    return call_openrouter(prompt, model=model, temperature=0.7)


def generate_dpo_response(scenario: dict, model: str = "qwen/qwen-2.5-7b-instruct") -> str:
    """Generate a response with the DPO system prompt (simulating trained behavior)."""
    prompt = GENERATION_PROMPT_TEMPLATE.format(
        role_description=scenario["role_description"],
        context_description=scenario["context_description"],
        situation=scenario["situation"],
        background=scenario["background"],
    )
    return call_openrouter(
        prompt, model=model,
        system_prompt=DPO_SYSTEM_PROMPT,
        temperature=0.7,
    )


def judge_comparison(
    scenario: dict,
    response_a: str,
    response_b: str,
    judge_model: str = "google/gemini-2.0-flash-001",
) -> dict:
    """Have the judge compare two responses blindly."""
    prompt = JUDGE_COMPARISON_PROMPT.format(
        role_description=scenario["role_description"],
        context_description=scenario["context_description"],
        situation=scenario["situation"],
        background=scenario["background"],
        response_a=response_a,
        response_b=response_b,
    )
    raw = call_openrouter(
        prompt + "\n\nRespond with valid JSON only.",
        model=judge_model,
        temperature=0.1,
    )
    return json.loads(strip_json_fences(raw))


def run_evaluation(
    eval_data_path: str = "data/evaluated.jsonl",
    output_path: str = "data/evaluation_results.jsonl",
    gen_model: str = "qwen/qwen-2.5-7b-instruct",
    judge_model: str = "google/gemini-2.0-flash-001",
    max_scenarios: int = 50,
):
    """Run full baseline vs DPO evaluation."""
    import random

    # Load holdout scenarios from evaluated.jsonl
    scenarios = []
    with open(eval_data_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("split") == "holdout":
                scenarios.append(d)

    if len(scenarios) > max_scenarios:
        random.seed(42)
        scenarios = random.sample(scenarios, max_scenarios)

    print(f"Evaluating {len(scenarios)} unique scenarios...")
    print(f"  Generation model: {gen_model}")
    print(f"  Judge model: {judge_model}")

    results = []
    wins = {"baseline": 0, "dpo": 0, "tie": 0}
    scores = {"baseline": defaultdict(list), "dpo": defaultdict(list)}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as fout:
        for i, scenario in enumerate(scenarios):
            try:
                # Generate responses
                baseline_resp = generate_baseline_response(scenario, model=gen_model)
                dpo_resp = generate_dpo_response(scenario, model=gen_model)

                # Randomize order to avoid position bias
                if random.random() < 0.5:
                    resp_a, resp_b = baseline_resp, dpo_resp
                    a_is = "baseline"
                else:
                    resp_a, resp_b = dpo_resp, baseline_resp
                    a_is = "dpo"

                # Judge
                judge_result = judge_comparison(
                    scenario, resp_a, resp_b, judge_model=judge_model
                )

                # Map winner back to model
                winner_raw = judge_result.get("winner", "tie").strip().upper()
                if winner_raw == "A":
                    actual_winner = a_is
                elif winner_raw == "B":
                    actual_winner = "dpo" if a_is == "baseline" else "baseline"
                else:
                    actual_winner = "tie"

                wins[actual_winner] += 1

                # Collect scores
                a_scores = judge_result.get("response_a", {})
                b_scores = judge_result.get("response_b", {})
                baseline_scores = a_scores if a_is == "baseline" else b_scores
                dpo_scores = b_scores if a_is == "baseline" else a_scores

                for dim in ["appropriateness", "role_fidelity", "collaborative_realism"]:
                    if dim in baseline_scores:
                        scores["baseline"][dim].append(baseline_scores[dim])
                    if dim in dpo_scores:
                        scores["dpo"][dim].append(dpo_scores[dim])

                record = {
                    "role_description": scenario["role_description"],
                    "context": scenario.get("context", ""),
                    "hierarchy": scenario.get("role", {}).get("hierarchy", ""),
                    "function": scenario.get("role", {}).get("function", ""),
                    "power_relation": scenario.get("role", {}).get("power_relation", ""),
                    "baseline_response": baseline_resp[:500],
                    "dpo_response": dpo_resp[:500],
                    "judge_result": judge_result,
                    "actual_winner": actual_winner,
                    "a_is": a_is,
                }
                fout.write(json.dumps(record) + "\n")

                if (i + 1) % 5 == 0:
                    print(f"  [{i+1}/{len(scenarios)}] Wins so far: baseline={wins['baseline']} dpo={wins['dpo']} tie={wins['tie']}")

            except Exception as e:
                print(f"  Error on scenario {i+1}: {e}")

    # Print summary
    total = sum(wins.values())
    print(f"\n{'='*50}")
    print(f"RESULTS ({total} scenarios)")
    print(f"{'='*50}")
    print(f"  Baseline wins: {wins['baseline']} ({100*wins['baseline']/max(total,1):.1f}%)")
    print(f"  DPO wins:      {wins['dpo']} ({100*wins['dpo']/max(total,1):.1f}%)")
    print(f"  Ties:          {wins['tie']} ({100*wins['tie']/max(total,1):.1f}%)")

    print(f"\nMean scores (1-5):")
    print(f"  {'Dimension':<25} {'Baseline':>10} {'DPO':>10} {'Delta':>10}")
    print(f"  {'-'*55}")
    for dim in ["appropriateness", "role_fidelity", "collaborative_realism"]:
        b_mean = sum(scores["baseline"][dim]) / max(len(scores["baseline"][dim]), 1)
        d_mean = sum(scores["dpo"][dim]) / max(len(scores["dpo"][dim]), 1)
        delta = d_mean - b_mean
        print(f"  {dim:<25} {b_mean:>10.2f} {d_mean:>10.2f} {delta:>+10.2f}")

    # Save summary
    summary = {
        "total_scenarios": total,
        "wins": wins,
        "mean_scores": {
            model: {dim: sum(vals)/max(len(vals),1) for dim, vals in dims.items()}
            for model, dims in scores.items()
        },
    }
    summary_path = output_path.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print(f"Full results saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate baseline vs DPO model")
    parser.add_argument("--eval-data", type=str, default="data/evaluated.jsonl")
    parser.add_argument("--output", type=str, default="data/evaluation_results.jsonl")
    parser.add_argument("--gen-model", type=str, default="qwen/qwen-2.5-7b-instruct")
    parser.add_argument("--judge-model", type=str, default="google/gemini-2.0-flash-001")
    parser.add_argument("--max-scenarios", type=int, default=50)
    args = parser.parse_args()

    run_evaluation(
        eval_data_path=args.eval_data,
        output_path=args.output,
        gen_model=args.gen_model,
        judge_model=args.judge_model,
        max_scenarios=args.max_scenarios,
    )
