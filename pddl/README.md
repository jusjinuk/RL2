# PDDL Problem Generation & Evaluation

Generate PDDL problems using LLMs and evaluate them with equivalence checking.

## Quick Start

```bash
python evaluate_pddl.py \
  --name "Qwen/Qwen3-4B-Instruct-2507" \
  --output-dir ./outputs/experiment1 \
  --completion-limit 200 \
  --batch-size 16 \
  --evaluate \
  --passk-path results.csv \
  --notes "baseline experiment"
```

## Required Arguments

- `--name`: Model name/path (e.g., `Qwen/Qwen3-4B-Thinking-2507`, `openai/gpt-oss-120b`)
- `--output-dir`: Directory to save results

## Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `BatsResearch/planetarium` | Dataset to use |
| `--dataset-split` | `test` | Dataset split |
| `--dataset-index` | All | Specific problem indices (e.g., `--dataset-index 1 2 3`) |
| `--completion-limit` | 200 | Number of completions per problem |
| `--batch-size` | 128 | Batch size for generation |
| `--first-turn-max-new-tokens` | 1024 | Max tokens for first turn |
| `--second-turn-max-new-tokens` | 1024 | Max tokens for second turn (thinking models) |
| `--evaluate` | False | Run PDDL equivalence evaluation |
| `--passk-path` | None | CSV file to save pass@k results |
| `--notes` | "" | Additional notes for CSV |
| `--domain-path` | `~/RL2/pddl` | Path to PDDL domain files |
| `--lora-path` | None | LoRA checkpoint path (warning only) |

## Supported Models

- **Qwen3 Think**: `Qwen/Qwen3-30B-A3B-Thinking-2507` (reasoning model)
- **Qwen3 Instruct**: `Qwen/Qwen3-4B-Instruct-2507` (standard model)
- **GPT-OSS**: `openai/gpt-oss-20b` (reasoning model)
- **Gemma**: `google/gemma*` (standard model)

## Output Format

### JSON Results
Each problem generates `{problem_id}.json.gz` containing:
```json
{
  "id": 0,
  "domain": "blocksworld",
  "natural_language": "...",
  "problem_pddl": "...",
  "is_placeholder": false,
  "prompt": "...",
  "completions": ["..."],
  "eval_results": [{"parseable": true, "valid": true, "equivalent": true}]
}
```

### CSV Results (with `--passk-path`)
```csv
timestamp,model,dataset,pass@1,num_problems,min_completions,max_completions,notes
2025-01-15T10:30:00,Qwen/Qwen3-Instruct-8B,experiment1,0.6234,164,200,200,baseline experiment
```

## Examples

**Basic generation (no evaluation):**
```bash
python evaluate_pddl.py \
  --name "Qwen/Qwen3-4B-Instruct-2507" \
  --output-dir ./outputs/baseline
```

**With evaluation and tracking:**
```bash
python evaluate_pddl.py \
  --name "Qwen/Qwen3-4B-Thinking-2507" \
  --output-dir ./outputs/thinking \
  --evaluate \
  --passk-path pass_k.csv \
  --notes "thinking model with temp=0.6"
```

**Specific problems only:**
```bash
python evaluate_pddl.py \
  --name "openai/gpt-oss-20b" \
  --output-dir ./outputs/test \
  --dataset-index 0 1 2 3 4 \
  --completion-limit 50
```

## Evaluation Metrics

- **Parseable**: PDDL syntax is valid
- **Valid**: PDDL problem is semantically correct
- **Equivalent**: Generated problem is equivalent to ground truth
- **Pass@1**: Probability of getting equivalent solution in 1 try
