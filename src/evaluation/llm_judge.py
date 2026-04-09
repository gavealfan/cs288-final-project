"""
LLM-as-Judge evaluation pipeline for organizational role fidelity.

Given a scenario and candidate responses, the judge produces structured
evaluations on three dimensions:
1. Appropriateness: Does the tone/content match the organizational context?
2. Role fidelity: Does the response reflect the role's hierarchy, function, and power?
3. Collaborative realism: Does it sound like a real person in this role?

Outputs pairwise preference rankings for DPO/IPO training.
"""

import json
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional


JUDGE_PROMPT = """You are an expert organizational behavior evaluator. Your task is to evaluate candidate responses in a workplace interaction scenario for how well they reflect the speaker's organizational role.

## Scenario
**Speaker's role:** {role_description}
**Interaction context:** {context_description}
**Situation:** {situation}
**Background:** {background}

## Candidate Responses
{candidates_formatted}

## Evaluation Criteria
Evaluate each response on these dimensions (1-5 scale):

1. **Appropriateness** (1-5): Does the tone, formality, and content match what is expected given the speaker's hierarchical position and power relationship? A score of 1 means completely inappropriate for this role/context; 5 means perfectly calibrated.

2. **Role Fidelity** (1-5): Does the response accurately reflect the speaker's hierarchy level, functional domain expertise, and seniority? Does it use the right level of technical detail, strategic framing, or operational specificity? 1 = completely breaks character; 5 = indistinguishable from a real person in this role.

3. **Collaborative Realism** (1-5): Does the response exhibit realistic collaborative characteristics? Consider:
   - Power-sensitive communication (appropriate deference or authority)
   - Decision authority awareness (not overstepping or under-asserting)
   - Information granularity (strategic vs. tactical, appropriate for the audience)
   - Accountability language (ownership, delegation, follow-up)
   1 = clearly artificial/unrealistic; 5 = convincingly human-like for this organizational role.

## Output Format
Output your evaluation as JSON:
{{
    "evaluations": [
        {{
            "response_index": 0,
            "appropriateness": <1-5>,
            "role_fidelity": <1-5>,
            "collaborative_realism": <1-5>,
            "rationale": "<brief explanation of scores>"
        }},
        ...
    ],
    "ranking": [<indices from best to worst overall>],
    "confidence": <0.0-1.0, your confidence in the ranking>,
    "notes": "<any observations about systematic issues, e.g., all responses miss power dynamics>"
}}"""


@dataclass
class JudgeEvaluation:
    response_index: int
    appropriateness: int
    role_fidelity: int
    collaborative_realism: int
    rationale: str

    @property
    def overall_score(self) -> float:
        return (self.appropriateness + self.role_fidelity + self.collaborative_realism) / 3.0

    def to_dict(self) -> dict:
        return {
            "response_index": self.response_index,
            "appropriateness": self.appropriateness,
            "role_fidelity": self.role_fidelity,
            "collaborative_realism": self.collaborative_realism,
            "overall_score": self.overall_score,
            "rationale": self.rationale,
        }


@dataclass
class JudgeResult:
    evaluations: List[JudgeEvaluation]
    ranking: List[int]
    confidence: float
    notes: str

    def to_preference_pairs(self) -> List[Dict]:
        """Convert ranking into pairwise preferences for DPO/IPO training."""
        pairs = []
        for i in range(len(self.ranking)):
            for j in range(i + 1, len(self.ranking)):
                winner_idx = self.ranking[i]
                loser_idx = self.ranking[j]
                winner_eval = self.evaluations[winner_idx]
                loser_eval = self.evaluations[loser_idx]
                margin = winner_eval.overall_score - loser_eval.overall_score

                pairs.append({
                    "winner_index": winner_idx,
                    "loser_index": loser_idx,
                    "margin": margin,
                    "confidence": self.confidence,
                    "winner_scores": winner_eval.to_dict(),
                    "loser_scores": loser_eval.to_dict(),
                })
        return pairs

    def to_dict(self) -> dict:
        return {
            "evaluations": [e.to_dict() for e in self.evaluations],
            "ranking": self.ranking,
            "confidence": self.confidence,
            "notes": self.notes,
            "preference_pairs": self.to_preference_pairs(),
        }


def format_candidates(responses: List[str]) -> str:
    parts = []
    for i, resp in enumerate(responses):
        parts.append(f"**Response {i}:**\n{resp}")
    return "\n\n".join(parts)


def build_judge_prompt(scenario: dict) -> str:
    return JUDGE_PROMPT.format(
        role_description=scenario["role_description"],
        context_description=scenario["context_description"],
        situation=scenario["situation"],
        background=scenario["background"],
        candidates_formatted=format_candidates(scenario["candidate_responses"]),
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


def call_judge(prompt: str, model: str = "google/gemini-2.0-flash-001") -> str:
    """Call the judge LLM via OpenRouter."""
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
        temperature=0.1,  # low temperature for consistent evaluation
    )
    return strip_json_fences(response.choices[0].message.content)


def parse_judge_output(raw: str) -> JudgeResult:
    data = json.loads(raw)
    evaluations = []
    for e in data["evaluations"]:
        evaluations.append(JudgeEvaluation(
            response_index=e["response_index"],
            appropriateness=e["appropriateness"],
            role_fidelity=e["role_fidelity"],
            collaborative_realism=e["collaborative_realism"],
            rationale=e["rationale"],
        ))
    return JudgeResult(
        evaluations=evaluations,
        ranking=data["ranking"],
        confidence=data.get("confidence", 0.5),
        notes=data.get("notes", ""),
    )


def evaluate_scenario(scenario: dict, model: str = "google/gemini-2.0-flash-001") -> JudgeResult:
    """Run the LLM judge on a single scenario."""
    prompt = build_judge_prompt(scenario)
    raw = call_judge(prompt, model=model)
    return parse_judge_output(raw)


def evaluate_dataset(
    input_path: str = "data/scenarios.jsonl",
    output_path: str = "data/evaluated.jsonl",
    judge_model: str = "google/gemini-2.0-flash-001",
):
    """Run the judge over all scenarios and output evaluated results."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = []
    with open(input_path) as f:
        for line in f:
            scenarios.append(json.loads(line))

    print(f"Evaluating {len(scenarios)} scenarios with {judge_model}...")
    with open(output_path, "w") as f:
        for i, scenario in enumerate(scenarios):
            try:
                result = evaluate_scenario(scenario, model=judge_model)
                record = {**scenario, "judge_result": result.to_dict()}
                f.write(json.dumps(record) + "\n")
                if (i + 1) % 10 == 0:
                    print(f"  Evaluated {i + 1}/{len(scenarios)}")
            except Exception as e:
                print(f"  Error evaluating scenario {i + 1}: {e}")

    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM-as-judge evaluation")
    parser.add_argument("--input", type=str, default="data/scenarios.jsonl")
    parser.add_argument("--output", type=str, default="data/evaluated.jsonl")
    parser.add_argument("--model", type=str, default="google/gemini-2.0-flash-001")
    args = parser.parse_args()

    evaluate_dataset(
        input_path=args.input,
        output_path=args.output,
        judge_model=args.model,
    )
