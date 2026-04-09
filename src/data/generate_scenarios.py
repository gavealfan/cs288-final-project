"""
Synthetic organizational dialogue generation pipeline.

For each (role, context) combination:
1. Generate a realistic scenario prompt
2. Generate multiple candidate responses (varied quality/appropriateness)
3. Output scenarios ready for LLM-as-judge evaluation
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from role_taxonomy import (
    OrganizationalRole,
    InteractionContext,
    CONTEXT_DESCRIPTIONS,
    Hierarchy,
    Function,
    PowerRelation,
    Seniority,
    sample_combinations,
    is_valid_combination,
)

# Number of candidate responses per scenario for pairwise comparison
NUM_CANDIDATES = 4


SCENARIO_GENERATION_PROMPT = """You are generating realistic workplace dialogue scenarios for training AI agents to simulate organizational roles.

{role_description}

Interaction context: {context_description}

Generate a realistic workplace scenario that fits this role and context. Include:
1. A brief situation description (2-3 sentences) establishing what happened and why this interaction is occurring.
2. Any relevant background (e.g., project name, deadline, prior events) to make the scenario concrete.
3. The message or dialogue turn that the person in this role would write/say in this situation.

The scenario should feel authentic to a real workplace. Use realistic but fictional names, projects, and details.

Output as JSON:
{{
    "situation": "<situation description>",
    "background": "<relevant background details>",
    "message": "<the role-appropriate message/dialogue turn>"
}}"""


CANDIDATE_GENERATION_PROMPT = """You are generating candidate responses for evaluating organizational role fidelity in workplace dialogue.

Here is the scenario:
- Role: {role_description}
- Context: {context_description}
- Situation: {situation}
- Background: {background}

Generate {num_candidates} different responses that this person might produce in this situation. The responses should vary in quality along these dimensions:
- **Appropriateness**: Does the tone match the role and power dynamic?
- **Role fidelity**: Does the response reflect the hierarchy, seniority, and function accurately?
- **Collaborative realism**: Does it sound like something a real person in this role would say?

Make sure the responses span a range from clearly inappropriate/unrealistic to highly appropriate/realistic. Do NOT label which is better -- just produce varied responses.

Output as JSON:
{{
    "responses": [
        "<response_1>",
        "<response_2>",
        "<response_3>",
        "<response_4>"
    ]
}}"""


@dataclass
class Scenario:
    role: OrganizationalRole
    context: InteractionContext
    situation: str
    background: str
    message: str
    candidate_responses: List[str]

    def to_dict(self) -> dict:
        return {
            "role": self.role.to_dict(),
            "context": self.context.value,
            "role_description": self.role.to_prompt_description(),
            "context_description": CONTEXT_DESCRIPTIONS[self.context],
            "situation": self.situation,
            "background": self.background,
            "message": self.message,
            "candidate_responses": self.candidate_responses,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(
            role=OrganizationalRole.from_dict(d["role"]),
            context=InteractionContext(d["context"]),
            situation=d["situation"],
            background=d["background"],
            message=d["message"],
            candidate_responses=d["candidate_responses"],
        )


def build_scenario_prompt(role: OrganizationalRole, context: InteractionContext) -> str:
    return SCENARIO_GENERATION_PROMPT.format(
        role_description=role.to_prompt_description(),
        context_description=CONTEXT_DESCRIPTIONS[context],
    )


def build_candidate_prompt(
    role: OrganizationalRole,
    context: InteractionContext,
    situation: str,
    background: str,
    num_candidates: int = NUM_CANDIDATES,
) -> str:
    return CANDIDATE_GENERATION_PROMPT.format(
        role_description=role.to_prompt_description(),
        context_description=CONTEXT_DESCRIPTIONS[context],
        situation=situation,
        background=background,
        num_candidates=num_candidates,
    )


def strip_json_fences(text: str) -> str:
    """Strip markdown code fences from LLM JSON responses."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def call_llm(prompt: str, model: str = "google/gemini-2.0-flash-001") -> str:
    """Call an LLM via OpenRouter (OpenAI-compatible API)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable.")

    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt + "\n\nRespond with valid JSON only."}],
        temperature=0.9,
    )
    return strip_json_fences(response.choices[0].message.content)


def generate_scenario(
    role: OrganizationalRole,
    context: InteractionContext,
    model: str = "google/gemini-2.0-flash-001",
) -> Scenario:
    """Generate a complete scenario with candidate responses."""
    # Step 1: Generate the scenario
    scenario_prompt = build_scenario_prompt(role, context)
    scenario_raw = call_llm(scenario_prompt, model=model)
    scenario_data = json.loads(scenario_raw)

    # Step 2: Generate varied candidate responses
    candidate_prompt = build_candidate_prompt(
        role, context,
        scenario_data["situation"],
        scenario_data["background"],
    )
    candidates_raw = call_llm(candidate_prompt, model=model)
    candidates_data = json.loads(candidates_raw)

    return Scenario(
        role=role,
        context=context,
        situation=scenario_data["situation"],
        background=scenario_data["background"],
        message=scenario_data["message"],
        candidate_responses=candidates_data["responses"],
    )


def generate_dataset(
    n_scenarios: int = 200,
    n_holdout: int = 50,
    output_path: str = "data/scenarios.jsonl",
    model: str = "google/gemini-2.0-flash-001",
    seed: int = 42,
):
    """Generate the full synthetic scenario dataset."""
    train_combos, holdout_combos = sample_combinations(n_scenarios, seed=seed)
    # Cap holdout to a reasonable size
    rng = __import__('random').Random(seed)
    if len(holdout_combos) > n_holdout:
        holdout_combos = rng.sample(holdout_combos, n_holdout)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate training scenarios
    print(f"Generating {len(train_combos)} training scenarios...")
    with open(output_path, "w") as f:
        for i, (role, ctx) in enumerate(train_combos):
            try:
                scenario = generate_scenario(role, ctx, model=model)
                record = scenario.to_dict()
                record["split"] = "train"
                f.write(json.dumps(record) + "\n")
                if (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{len(train_combos)} training scenarios")
            except Exception as e:
                print(f"  Error generating scenario {i + 1}: {e}")

    # Generate holdout scenarios
    holdout_path = output_path.replace(".jsonl", "_holdout.jsonl")
    print(f"\nGenerating {len(holdout_combos)} holdout scenarios...")
    with open(holdout_path, "w") as f:
        for i, (role, ctx) in enumerate(holdout_combos):
            try:
                scenario = generate_scenario(role, ctx, model=model)
                record = scenario.to_dict()
                record["split"] = "holdout"
                f.write(json.dumps(record) + "\n")
                if (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{len(holdout_combos)} holdout scenarios")
            except Exception as e:
                print(f"  Error generating scenario {i + 1}: {e}")

    print(f"\nDone. Training: {output_path}, Holdout: {holdout_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic organizational dialogue scenarios")
    parser.add_argument("--n", type=int, default=200, help="Number of training scenarios")
    parser.add_argument("--n-holdout", type=int, default=50, help="Number of holdout scenarios")
    parser.add_argument("--output", type=str, default="data/scenarios.jsonl", help="Output file path")
    parser.add_argument("--model", type=str, default="google/gemini-2.0-flash-001", help="LLM model to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_dataset(
        n_scenarios=args.n,
        n_holdout=args.n_holdout,
        output_path=args.output,
        model=args.model,
        seed=args.seed,
    )
