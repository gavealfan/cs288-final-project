"""
Render report-ready LaTeX tables from evaluation summary JSON.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    summary_path: str = field(default="data/evaluation_three_way_summary.json")
    overall_table_tex: str = field(default="artifacts/report/overall_table.tex")


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}"


def main() -> None:
    args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    Path(args.overall_table_tex).parent.mkdir(parents=True, exist_ok=True)

    with open(args.summary_path) as f:
        s = json.load(f)

    wr = s.get("winner_rates", {})
    ms = s.get("mean_scores", {})

    base = ms.get("base", {})
    sft = ms.get("sft", {})
    dpo = ms.get("dpo", {})

    tex = f"""\\begin{{table}}[t]
\\caption{{Three-way held-out evaluation (Base vs SFT vs SFT+DPO).}}
\\label{{tab:overall}}
\\begin{{center}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Base}} & \\textbf{{SFT}} & \\textbf{{SFT+DPO}} \\\\
\\midrule
Winner Rate (\\%) & {pct(wr.get('base', 0.0))} & {pct(wr.get('sft', 0.0))} & {pct(wr.get('dpo', 0.0))} \\\\
Appropriateness & {base.get('appropriateness', 0.0):.2f} & {sft.get('appropriateness', 0.0):.2f} & {dpo.get('appropriateness', 0.0):.2f} \\\\
Role Fidelity & {base.get('role_fidelity', 0.0):.2f} & {sft.get('role_fidelity', 0.0):.2f} & {dpo.get('role_fidelity', 0.0):.2f} \\\\
Collaborative Realism & {base.get('collaborative_realism', 0.0):.2f} & {sft.get('collaborative_realism', 0.0):.2f} & {dpo.get('collaborative_realism', 0.0):.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{center}}
\\end{{table}}
"""

    with open(args.overall_table_tex, "w") as f:
        f.write(tex)

    print(f"Wrote {args.overall_table_tex}")


if __name__ == "__main__":
    main()
