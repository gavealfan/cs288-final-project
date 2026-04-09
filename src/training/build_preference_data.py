"""
Convert LLM-judge evaluated scenarios into DPO/IPO preference pairs.

Takes evaluated.jsonl (scenarios + judge results) and produces a dataset
in the format expected by TRL's DPOTrainer:
  {"prompt": ..., "chosen": ..., "rejected": ...}
"""

import json
from pathlib import Path
from typing import Optional


def build_dpo_prompt(scenario: dict) -> str:
    """Construct the conditioning prompt for DPO training."""
    return (
        f"You are in the following organizational role:\n"
        f"{scenario['role_description']}\n\n"
        f"Interaction context:\n"
        f"{scenario['context_description']}\n\n"
        f"Situation: {scenario['situation']}\n"
        f"Background: {scenario['background']}\n\n"
        f"Respond appropriately given your role and the situation."
    )


def extract_preference_pairs(
    input_path: str = "data/evaluated.jsonl",
    output_path: str = "data/dpo_pairs.jsonl",
    min_margin: float = 0.5,
    min_confidence: float = 0.5,
):
    """
    Extract DPO preference pairs from judge evaluations.

    Args:
        input_path: Path to evaluated scenarios
        output_path: Path to write DPO pairs
        min_margin: Minimum score margin to include a pair (filters ambiguous pairs)
        min_confidence: Minimum judge confidence to include
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    total_scenarios = 0
    total_pairs = 0
    filtered_margin = 0
    filtered_confidence = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            scenario = json.loads(line)
            total_scenarios += 1

            judge = scenario.get("judge_result", {})
            confidence = judge.get("confidence", 0.0)

            if confidence < min_confidence:
                filtered_confidence += 1
                continue

            prompt = build_dpo_prompt(scenario)
            responses = scenario["candidate_responses"]

            for pair in judge.get("preference_pairs", []):
                if pair["margin"] < min_margin:
                    filtered_margin += 1
                    continue

                winner_idx = pair["winner_index"]
                loser_idx = pair["loser_index"]

                dpo_record = {
                    "prompt": prompt,
                    "chosen": responses[winner_idx],
                    "rejected": responses[loser_idx],
                    "metadata": {
                        "role": scenario["role"],
                        "context": scenario["context"],
                        "margin": pair["margin"],
                        "confidence": confidence,
                        "winner_scores": pair["winner_scores"],
                        "loser_scores": pair["loser_scores"],
                        "split": scenario.get("split", "train"),
                    },
                }
                fout.write(json.dumps(dpo_record) + "\n")
                total_pairs += 1

    print(f"Processed {total_scenarios} scenarios")
    print(f"Generated {total_pairs} DPO preference pairs")
    print(f"Filtered: {filtered_margin} (low margin), {filtered_confidence} (low confidence)")
    print(f"Output: {output_path}")


def split_train_eval(
    input_path: str = "data/dpo_pairs.jsonl",
    train_path: str = "data/dpo_train.jsonl",
    eval_path: str = "data/dpo_eval.jsonl",
):
    """Split DPO pairs by the train/holdout split from scenario generation."""
    train_count = 0
    eval_count = 0

    with open(input_path) as fin:
        with open(train_path, "w") as ftrain, open(eval_path, "w") as feval:
            for line in fin:
                record = json.loads(line)
                split = record.get("metadata", {}).get("split", "train")
                if split == "holdout":
                    feval.write(line)
                    eval_count += 1
                else:
                    ftrain.write(line)
                    train_count += 1

    print(f"Train: {train_count} pairs -> {train_path}")
    print(f"Eval: {eval_count} pairs -> {eval_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build DPO preference pairs from judge evaluations")
    parser.add_argument("--input", type=str, default="data/evaluated.jsonl")
    parser.add_argument("--output", type=str, default="data/dpo_pairs.jsonl")
    parser.add_argument("--min-margin", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()

    extract_preference_pairs(
        input_path=args.input,
        output_path=args.output,
        min_margin=args.min_margin,
        min_confidence=args.min_confidence,
    )

    split_train_eval(
        input_path=args.output,
        train_path=args.output.replace(".jsonl", "_train.jsonl"),
        eval_path=args.output.replace(".jsonl", "_eval.jsonl"),
    )
