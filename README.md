# CS288 Final Project

## Local Pipeline (Qwen2.5-3B Student, Qwen3.5-397B-A17B Teacher)

This repo includes a reproducible local pipeline to:
- build teacher-supervised SFT data,
- train a local SFT adapter,
- run three-way evaluation (Base vs SFT vs SFT+DPO),
- render LaTeX report tables.

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install openai peft trl datasets transformers accelerate
```

### 2) Set API key

```bash
export OPENROUTER_API_KEY="your_key_here"
```

### 3) Run the full local harness

```bash
python src/pipeline/run_experiment.py --config configs/experiment_local_qwen3b.json
```

### Expected outputs

- Teacher data files:
  - `data/teacher_sft_train.jsonl`
  - `data/teacher_sft_eval.jsonl`
  - `data/teacher_pref_train.jsonl`
  - `data/teacher_pref_eval.jsonl`
- Local SFT checkpoint:
  - `checkpoints/role-conditioned-sft-local/`
- Three-way eval outputs:
  - `data/evaluation_three_way.jsonl`
  - `data/evaluation_three_way_summary.json`
- Report table:
  - `artifacts/report/overall_table.tex`

### Notes

- Teacher model is configured as `qwen/qwen3.5-397b-a17b`.
- The pipeline config is in `configs/experiment_local_qwen3b.json`.
