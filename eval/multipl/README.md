# MultiPL-E Code Generation Evaluation

Evaluate code generation models on MultiPL-E benchmark (HumanEval in multiple languages).

## Quick Start

```bash
bash multipl.sh
```

## Configuration

Edit `multipl.sh` to customize:

```bash
python multipl_eval.py \
  --name "Qwen/Qwen3-4B-Thinking-2507" \  # Model name
  --lang jl \                              # Language: r, rkt, ml, jl, lua
  --output-dir ./multipl_e/jl/qwen3 \     # Output directory
  --completion-limit 4 \                   # Completions per problem
  --batch-size 64 \                        # Batch size
  --first-turn-max-new-tokens 1024 \      # First turn tokens
  --second-turn-max-new-tokens 1024 \     # Second turn tokens
  --evaluate \                             # Run evaluation
  --notes "multipl_e jl qwen3" \          # CSV notes
  --passk-path results.csv \              # Pass@k CSV file
  --dataset-index 0 1 2 3 4               # Problem subset (optional)
```

## Arguments

- `--name`: HuggingFace model path
- `--lang`: Target language (r/rkt/ml/jl/lua)
- `--output-dir`: Results directory
- `--completion-limit`: Completions per problem (default: 200)
- `--batch-size`: Generation batch size (default: 16)
- `--first-turn-max-new-tokens`: Max tokens for reasoning (default: 1024)
- `--second-turn-max-new-tokens`: Max tokens for completion (default: 1024)
- `--evaluate`: Enable code evaluation
- `--passk-path`: CSV file for pass@k results
- `--notes`: Additional CSV metadata
- `--eval-url`: Sandbox API URL (default: http://147.46.15.142:8080/run_code)
- `--dataset-index`: Filter specific problems (e.g., `0 1 2 3`)

## Pipeline Stages

1. **Get Ready** (1_get_ready): Load dataset and model
2. **Generate** (2_generate): Batch code generation
3. **Evaluate** (3_evaluate): Async code execution
4. **Wrapup** (4_wrapup): Calculate pass@k metrics

## Output Files

- `{output_dir}/*.json.gz`: Per-problem results
- `{output_dir}/timing_multipl_*.json`: Timing breakdown
- `results.csv`: Pass@k metrics across runs

## Timing Analysis

```bash
# View timing for single run
cat multipl_e/jl/qwen3/timing_multipl_*.json
```
