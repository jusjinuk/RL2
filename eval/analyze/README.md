# Evaluation Analysis Tools

Analysis utilities for MultiPL-E and PDDL evaluation results.

## Tools

### 1. Timing Tracker (`timing_tracker.py`)

Automatic timestamp tracking for evaluation pipeline stages.

**Pipeline Stages:**
1. **Get Ready**: Model/dataset loading
2. **Generate**: Batch generation
3. **Evaluate**: Code/PDDL evaluation
4. **Wrapup**: Pass@k calculation

**Usage:**
Already integrated into `multipl_eval.py` and `pddl_eval.py` - runs automatically.

**Output:**
Timing data saved to: `{output_dir}/timing_{experiment_name}.json`

**Example Output:**
```
============================================================
TIMING SUMMARY
============================================================
Experiment: multipl_jl_Qwen_Qwen3_4B_Instruct_2507
Start: 2025-10-18T10:30:00
End: 2025-10-18T11:45:30
Total Duration: 1h 15m 30.25s

Stage Breakdown:
------------------------------------------------------------
  1_get_ready         : 2m 15.43s
  2_generate          : 1h 10m 5.12s
  3_evaluate          : 2m 45.67s
  4_wrapup            : 24.03s
============================================================
```

### 2. Pass@1 Calculator (`pass_1.py`)

Calculate pass@1 metrics for evaluation results.

**Usage:**
```bash
# MultiPL-E evaluation
python pass_1.py ./multipl_e/jl/qwen3

# PDDL evaluation (default: equivalent)
python pass_1.py ./pddl/qwen3

# PDDL with different metrics
python pass_1.py ./pddl/qwen3 --metric parseable
python pass_1.py ./pddl/qwen3 --metric valid
python pass_1.py ./pddl/qwen3 --metric equivalent

# Multiple directories
python pass_1.py ./multipl_e/*/qwen3 ./pddl/qwen3

# CSV output without header
python pass_1.py ./results/* --suppress-header
```

**Arguments:**
- `dirs`: Directories containing result files
- `--metric`: PDDL metric (parseable/valid/equivalent)
- `--suppress-header`: Omit CSV header

**Output Format:**
```
Dataset,Pass@1,NumProblems,MinCompletions,MaxCompletions
qwen3,0.7450,164,4,4
```

## File Formats

### Timing JSON Structure
```json
{
  "experiment": "multipl_jl_Qwen_Qwen3_4B_Instruct_2507",
  "start_time": "2025-10-18T10:30:00",
  "end_time": "2025-10-18T11:45:30",
  "total_duration_seconds": 4530.25,
  "total_duration_formatted": "1h 15m 30.25s",
  "stages": {
    "1_get_ready": {
      "start_time": "2025-10-18T10:30:00",
      "end_time": "2025-10-18T10:32:15",
      "duration_seconds": 135.43,
      "duration_formatted": "2m 15.43s"
    },
    "2_generate": { ... },
    "3_evaluate": { ... },
    "4_wrapup": { ... }
  },
  "metadata": {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "lang": "jl",
    "batch_size": 64,
    "completion_limit": 4
  }
}
```

### Result JSON Structure

**MultiPL-E:**
```json
{
  "name": "HumanEval_0_has_close_elements",
  "language": "Julia",
  "results": [
    {"status": "OK", "exit_code": 0},
    {"status": "FAIL", "exit_code": 1}
  ]
}
```

**PDDL:**
```json
{
  "id": 1,
  "domain": "blocksworld",
  "eval_results": [
    {"parseable": true, "valid": true, "equivalent": true},
    {"parseable": true, "valid": false, "equivalent": false}
  ]
}
```

## Workflow Example

### Complete Analysis Pipeline

```bash
# 1. Run evaluations
cd ../multipl && bash multipl.sh
cd ../pddl && bash pddl.sh

# 2. Calculate pass@1
cd ../analyze
python pass_1.py ../multipl/multipl_e/jl/qwen3
python pass_1.py ../pddl/pddl/qwen3 --metric equivalent

# 3. View detailed timing
cat ../multipl/multipl_e/jl/qwen3/timing_*.json | jq .
```

