"""
Single-command experiment harness for reproducible runs.

Reads a JSON config and runs selected stages:
- teacher data build
- local SFT training
- DPO/IPO training
- three-way evaluation
- report table rendering
"""

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    config: str = field(default="configs/experiment_local_qwen3b.json")


def run(cmd: str) -> None:
    print(f"\n[run] {cmd}")
    subprocess.run(shlex.split(cmd), check=True)


def resolve_sft_profile(cfg: dict) -> dict:
    profile_name = cfg.get("student_profile", "qwen3b")
    profiles = cfg.get("student_profiles", {})
    profile = profiles.get(profile_name)
    if profile is None:
        return cfg["sft_local"]

    base = dict(cfg["sft_local"])
    base.update(profile)
    print(f"[profile] student_profile={profile_name} model={base.get('model_name')}")
    return base


def main() -> None:
    args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    with open(args.config) as f:
        cfg = json.load(f)

    needs_openrouter = cfg.get("build_teacher_data", True) or cfg.get("run_three_way_eval", True)
    if needs_openrouter and not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is required for teacher generation/evaluation.\n"
            "Set it first, e.g.:\n"
            "  export OPENROUTER_API_KEY='your_key_here'"
        )

    if cfg.get("build_teacher_data", True):
        t = cfg["teacher_data"]
        run(
            "python src/data/build_teacher_sft_data.py "
            f"--input-path {t['input_path']} "
            f"--output-sft-train {t['output_sft_train']} "
            f"--output-sft-eval {t['output_sft_eval']} "
            f"--output-pref-train {t['output_pref_train']} "
            f"--output-pref-eval {t['output_pref_eval']} "
            f"--base-model {t['base_model']} "
            f"--teacher-model {t['teacher_model']} "
            f"--max-scenarios {t.get('max_scenarios', 0)} "
            f"--seed {t.get('seed', 42)}"
        )

    if cfg.get("run_sft_local", True):
        s = resolve_sft_profile(cfg)
        run(
            "python src/training/train_sft_local.py "
            f"--model-name {s['model_name']} "
            f"--train-path {s['train_path']} "
            f"--eval-path {s['eval_path']} "
            f"--output-dir {s['output_dir']} "
            f"--num-train-epochs {s.get('num_train_epochs', 1)} "
            f"--max-steps {s.get('max_steps', 200)} "
            f"--max-train-samples {s.get('max_train_samples', 0)} "
            f"--max-eval-samples {s.get('max_eval_samples', 0)}"
        )

    if cfg.get("run_dpo", False):
        d = cfg["dpo"]
        run(
            "python src/training/train_dpo.py "
            f"--model-name {d['model_name']} "
            f"--train-path {d['train_path']} "
            f"--eval-path {d['eval_path']} "
            f"--output-dir {d['output_dir']} "
            f"--loss-type {d.get('loss_type', 'sigmoid')} "
            f"--num-epochs {d.get('num_epochs', 1)} "
            f"--per-device-batch-size {d.get('per_device_batch_size', 1)} "
            f"--gradient-accumulation-steps {d.get('gradient_accumulation_steps', 1)}"
        )

    if cfg.get("run_three_way_eval", True):
        e = cfg["eval"]
        run(
            "python src/evaluation/evaluate_three_way.py "
            f"--eval-data-path {e['eval_data_path']} "
            f"--output-path {e['output_path']} "
            f"--summary-path {e['summary_path']} "
            f"--base-model {e['base_model']} "
            f"--sft-model {e['sft_model']} "
            f"--dpo-model {e['dpo_model']} "
            f"--judge-model {e['judge_model']} "
            f"--max-scenarios {e.get('max_scenarios', 50)} "
            f"--seed {e.get('seed', 42)}"
        )

    if cfg.get("render_report_tables", True):
        r = cfg["report"]
        run(
            "python src/report/render_report_tables.py "
            f"--summary-path {r['summary_path']} "
            f"--overall-table-tex {r['overall_table_tex']}"
        )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
