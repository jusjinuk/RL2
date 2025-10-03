# AutoModel Instruct - Code Generation & Evaluation

Generate code completions using LLMs and evaluate them with pass@k metrics.

## Quick Start

```bash
python automodel_instruct.py \
  --name "Qwen/Qwen3-Instruct-8B" \
  --lang jl \
  --output-dir ./outputs/experiment1 \
  --completion-limit 200 \
  --batch-size 16 \
  --evaluate \
  --passk-path results.csv \
  --notes "baseline experiment"
```

## Required Arguments

- `--name`: Model name/path (e.g., `Qwen/Qwen3-Instruct-8B`, `gpt-oss`)
- `--lang`: Programming language (`r`, `rkt`, `ml`, `jl`, `lua`)
- `--output-dir`: Directory to save results

## Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--completion-limit` | 200 | Number of completions per problem |
| `--batch-size` | 16 | Batch size for generation |
| `--first-turn-max-new-tokens` | 1024 | Max tokens for first turn |
| `--second-turn-max-new-tokens` | 1024 | Max tokens for second turn (thinking models) |
| `--dataset-index` | All | Specific problem indices (e.g., `--dataset-index 1 2 3`) |
| `--evaluate` | False | Run code evaluation |
| `--eval-url` | `http://ipaddress:port/run_code` | Sandbox API URL (tao ip address is now default) |
| `--passk-path` | None | CSV file to save pass@k results |
| `--notes` | "" | Additional notes for CSV |
| `--lora-path` | None | LoRA checkpoint path (warning only) |

## Supported Models

- **Qwen3 Think**: `Qwen/Qwen3-30B-A3B-Thinking-2507` (reasoning model)
- **Qwen3 Instruct**: `Qwen/Qwen3-4B-Instruct-2507` (standard model)
- **GPT-OSS**: `openai/gpt-oss-20b` (reasoning model)

## Output Format

### JSON Results
Each problem generates `{problem_name}.json.gz` containing:
```json
{
  "name": "HumanEval_0_has_close_elements",
  "language": "Julia",
  "prompt": "...",
  "tests": "...",
  "completions": ["..."],
  "eval_results": [{"status": "Finished", "execution_time": 0.241116762161255, "return_code": 0, "stdout": "Test Summary: | Pass  Total\ntest set      |    3      3\n", "stderr": ""}]
}
```

### CSV Results (with `--passk-path`)
```csv
timestamp,model,language,dataset,pass@1,num_problems,min_completions,max_completions,notes
2025-01-15T10:30:00,Qwen/Qwen3-Instruct-8B,jl,experiment1,0.6234,164,200,200,baseline experiment
```

## Examples

**Basic generation (no evaluation):**
```bash
python automodel_instruct.py \
  --name "Qwen/Qwen3-4B-Instruct-2507" \
  --lang lua \
  --output-dir ./outputs/lua_baseline
```

**With evaluation and tracking:**
```bash
python automodel_instruct.py \
  --name "Qwen/Qwen3-4B-Thinking-2507" \
  --lang jl \
  --output-dir ./outputs/julia_think \
  --evaluate \
  --passk-path pass_k.csv \
  --notes "thinking model with temp=0.6"
```

**Specific problems only:**
```bash
python automodel_instruct.py \
  --name "openai/gpt-oss-20b" \
  --lang r \
  --output-dir ./outputs/r_test \
  --dataset-index 0 1 2 3 4 \
  --completion-limit 50
```

## Environment Variables

- `SANDBOX_URL`: Override default evaluation sandbox URL
