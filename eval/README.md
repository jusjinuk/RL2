# LLM Generation & Evaluation for Code and PDDL

This framework uses Large Language Models (LLMs) to generate and evaluate outputs for two primary tasks:
1.  **Code Generation (MultiPL-E):** Generates code in Julia, Lua, OCaml, R, and Racket, evaluated using `pass@k` metrics in a sandbox.
2.  **PDDL Generation:** Generates Planning Domain Definition Language (PDDL) problems, evaluated via equivalence checking.

***

## Quick Start

### 1. Code Generation (MultiPL-E)

```bash
python multipl_eval.py \
  --name "Qwen/Qwen3-4B-Instruct-2507" \
  --lang jl \
  --output-dir ./outputs/code_experiment \
  --evaluate \
  --passk-path results.csv
````

### 2\. PDDL Generation

```bash
python pddl_eval.py \
  --name "Qwen/Qwen3-4B-Instruct-2507" \
  --output-dir ./outputs/pddl_experiment \
  --evaluate \
  --passk-path results.csv
```

-----

## Arguments

### Common Arguments

| Argument | Default | Description |
|---|---|---|
| `--name` | **Required** | Model name or path (e.g., `Qwen/Qwen3-4B-Instruct-2507`) |
| `--output-dir` | **Required** | Directory to save generation and evaluation results |
| `--completion-limit`| 200 | Number of completions to generate per problem |
| `--batch-size` | 16 | Batch size for generation |
| `--dataset-index` | All | Specific problem indices (e.g., `--dataset-index 1 2 3`) |
| `--evaluate` | False | Flag to run evaluation after generation |
| `--passk-path` | None | Path to save `pass@k` results in a CSV file |
| `--first-turn-max-new-tokens`| 1024 | Max new tokens for the first generation turn |
| `--second-turn-max-new-tokens`| 1024 | Max new tokens for the second turn (used by thinking models) |
| `--notes` | "" | Additional notes to include in the results CSV |
| `--lora-path` | None | Path to a LoRA checkpoint (Warning: may not be fully supported) |

### Task-Specific Arguments

**For Code Generation (`multipl_eval.py`):**

  * `--lang`: **(Required)** Programming language. Options: `jl`, `lua`, `ml`, `r`, `rkt`.
  * `--eval-url`: URL for the code execution sandbox API. Defaults to the value of the `SANDBOX_URL` environment variable if set.

**For PDDL Generation (`pddl_eval.py`):**

  * `--dataset`: Dataset to use. Defaults to `BatsResearch/planetarium`.
  * `--dataset-split`: Dataset split. Defaults to `test`.
  * `--domain-path`: Path to PDDL domain files. Defaults to `~/RL2/pddl`.

-----

## Supported Models

This framework supports a variety of standard and reasoning-specialized ("thinking") models, including:

  * **Qwen3 Think**: `Qwen/Qwen3-30B-A3B-Thinking-2507`
  * **Qwen3 Instruct**: `Qwen/Qwen3-4B-Instruct-2507`
  * **GPT-OSS**: `openai/gpt-oss-20b`

-----

## Output Format

Results are saved in two formats: detailed JSON files for each problem and an aggregated CSV for `pass@k` scores.

### JSON Results (`.json.gz`)

A gzipped JSON file is created for each problem, containing the prompt, completions, and evaluation results.

**Code Generation Example:**

```json
{
  "name": "HumanEval_0_has_close_elements",
  "language": "Julia",
  "completions": ["..."],
  "eval_results": [{
      "status": "Finished", 
      "return_code": 0, 
      "stdout": "Test Summary: | Pass  Total\ntest set      |    3      3\n", 
      "stderr": ""
  }]
}
```

**PDDL Generation Example:**

```json
{
  "id": 0,
  "domain": "blocksworld",
  "completions": ["..."],
  "eval_results": [{
      "parseable": true, 
      "valid": true, 
      "equivalent": true
  }]
}
```

### CSV Results (`--passk-path`)

If a path is provided, a CSV file will be created to track `pass@k` scores across experiments.

**Example Row:**

```csv
timestamp,model,language,dataset,pass@1,num_problems,min_completions,max_completions,notes
2025-01-15T10:30:00,Qwen/Qwen3-Instruct-8B,jl,experiment1,0.6234,164,200,200,baseline experiment
```
