"""PDDL Problem Generation and Evaluation using SGLang."""
import warnings
import os

# Suppress FutureWarnings from transformers
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
# Set HF_HOME to suppress TRANSFORMERS_CACHE deprecation warning
os.environ.setdefault('HF_HOME', os.environ.get('TRANSFORMERS_CACHE', os.path.expanduser('~/.cache/huggingface')))

from transformers import AutoTokenizer
import torch
from typing import List
import re
import sglang as sgl
import datasets
import argparse
import gzip
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import signal
import csv
from datetime import datetime
import sys
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.pddl import equivalence, extract_pddl_from_response

# Model-specific patterns for reasoning extraction
MODEL_PATTERNS = {
    'qwen3_think': {
        'end_reasoning': r'</think>',
        'thinking_suffix': "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n",
        'thinking_check': lambda text: '</think>' in text
    },
    'gpt_oss': {
        'end_reasoning': r'<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>',
        'thinking_suffix': "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.<|end|><|start|>assistant<|channel|>final<|message|>",
        'thinking_check': lambda text: bool(re.search(r"assistantfinal.*?```pddl.*?```", text, re.DOTALL))
    }
}

def arg_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PDDL Generation and Evaluation")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="BatsResearch/planetarium")
    parser.add_argument("--dataset-split", type=str, default="test")
    parser.add_argument("--dataset-index", nargs='+', type=int)
    parser.add_argument("--completion-limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--first-turn-max-new-tokens", type=int, default=1024)
    parser.add_argument("--second-turn-max-new-tokens", type=int, default=1024)
    parser.add_argument("--name", type=str, required=True, help="Model name/path")
    parser.add_argument("--lora-path", type=str)
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--passk-path", type=str)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--domain-path", type=str, default="~/RL2/pddl")
    return parser

@lru_cache(maxsize=8)
def get_params(model_name):
    """Get model-specific parameters (cached for performance)."""
    name_lower = model_name.lower()
    if "qwen3" in name_lower:
        return {
            "name": "qwen3_think" if "think" in name_lower else "qwen3_instruct",
            "temperature": 0.6 if "think" in name_lower else 0.7,
            "top_p": 0.95,
            "dtype": torch.bfloat16,
            "enable_thinking": "think" in name_lower
        }
    elif "oss" in name_lower:
        return {"name": "gpt_oss", "temperature": 1.0, "top_p": 1.0,
                "dtype": "auto", "enable_thinking": True}
    elif "gemma" in name_lower:
        return {"name": "gemma", "temperature": 1.0, "top_p": 0.95,
                "dtype": torch.bfloat16, "enable_thinking": False}
    raise ValueError(f"Unsupported model: {model_name}")

def initialize_engine(dtype, name):
    """Initialize SGLang engine with optimal GPU configuration."""
    gpu_count = torch.cuda.device_count()
    optimal_tp_size = 1 << (gpu_count.bit_length() - 1)
    return sgl.Engine(model_path=name, dtype=dtype, tp_size=optimal_tp_size,
                      mem_fraction_static=0.7)

def load_domains(domain_path: str) -> dict[str, str]:
    """Load PDDL domain files from directory."""
    domain_dir = Path(domain_path).expanduser()
    return {f.stem: f.read_text() for f in domain_dir.glob("*.pddl")}

def read_completions(exp_dir, model_name, first_turn_max_new_tokens,
                     second_turn_max_new_tokens, problem):
    """Read or initialize completion data for a problem."""
    problem_filename = exp_dir / f"{problem['id']}.json.gz"

    if problem_filename.exists():
        with gzip.open(problem_filename, "rt") as f:
            existing = json.load(f)
            return (existing["id"], existing)

    params = get_params(model_name)
    return (problem["id"], {
        "id": problem["id"],
        "domain": problem["domain"],
        "natural_language": problem["natural_language"],
        "problem_pddl": problem["problem_pddl"],
        "is_placeholder": problem["is_placeholder"],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "first_turn_max_new_tokens": first_turn_max_new_tokens,
        "second_turn_max_new_tokens": second_turn_max_new_tokens,
        "prompt": problem["prompt"],
        "prompt_and_reasoning": [],
        "final_response": [],
        "completions": [],
        "eval_results": []
    })

# Signal handler for timeout (planetarium compatibility)
def signal_handler(signum, frame):
    raise TimeoutError("Timed out")

signal.signal(signal.SIGALRM, signal_handler)

def _evaluate_single_pddl(args):
    """Evaluate single PDDL (multiprocessing worker)."""
    gt_pddl, llm_pddl, domains, is_placeholder = args
    try:
        signal.alarm(600)  # 10-minute timeout
        parseable, valid, equivalent = equivalence(gt_pddl, llm_pddl, domains, is_placeholder)
        signal.alarm(0)
        return {'parseable': parseable, 'valid': valid, 'equivalent': equivalent}
    except TimeoutError:
        return {'parseable': None, 'valid': None, 'equivalent': None, 'error': 'TIMEOUT'}
    except Exception as e:
        return {'parseable': False, 'valid': False, 'equivalent': None, 'error': str(e)}

def evaluate_pddl_batch(llm_pddls, ground_truth_pddls, domains_list, is_placeholder_list):
    """Batch evaluation using multiprocessing.Pool with optimized chunking."""
    eval_args = list(zip(ground_truth_pddls, llm_pddls, domains_list, is_placeholder_list))
    num_processes = max(1, min(mp.cpu_count(), len(eval_args)))
    chunksize = max(1, len(eval_args) // (num_processes * 2))

    print(f"  Using {num_processes} worker processes (chunksize={chunksize})...")

    with mp.Pool(processes=num_processes) as pool:
        return list(pool.imap_unordered(_evaluate_single_pddl, eval_args, chunksize=chunksize))

def format_pddl_prompt(natural_language: str, domain_pddl: str) -> str:
    """Format prompt for PDDL problem generation."""
    instruction = "Provide me with the complete, valid problem PDDL file " + \
                  "that describes the following planning problem in ```pddl markdown blocks:\n\n"
    return f"{instruction}{natural_language}\n\nThe domain for the planning problem is:\n{domain_pddl}\n"

def process_final_response(final_response: str, model_name: str) -> str:
    """Extract PDDL code from model response."""
    params = get_params(model_name)
    pattern_config = MODEL_PATTERNS.get(params["name"])

    # Remove reasoning prefix if present
    if pattern_config:
        match = re.search(pattern_config['end_reasoning'], final_response)
        if match:
            final_response = final_response[match.end():]

    return extract_pddl_from_response(final_response)

def make_main(args, model_name, gen_completions, domains):
    """Main execution loop for PDDL generation and evaluation."""
    exp_dir = Path(args.output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load and filter dataset
    problems = datasets.load_dataset(args.dataset, split=args.dataset_split)
    if args.dataset_index:
        problems = problems.filter(lambda x: x["id"] in args.dataset_index)

    # Add prompts
    problems = problems.map(lambda ex: {
        **ex,
        "prompt": format_pddl_prompt(ex["natural_language"], domains[ex["domain"]])
    })

    # Load existing completions
    all_completions = dict(
        read_completions(exp_dir, args.name, args.first_turn_max_new_tokens,
                        args.second_turn_max_new_tokens, p) for p in problems
    )

    # Build problem list with needed completions
    problem_list = []
    for comp in all_completions.values():
        if len(comp["final_response"]) < args.completion_limit:
            needed = args.completion_limit - len(comp["final_response"])
            item = {k: comp[k] for k in ["prompt", "id", "problem_pddl", "domain", "is_placeholder"]}
            problem_list.extend([item] * needed)

    batches = [problem_list[i:i+args.batch_size] for i in range(0, len(problem_list), args.batch_size)]

    for batch_idx, batch in enumerate(tqdm(batches, unit="batch", desc="Processing")):
        # Generate completions
        new_completions = gen_completions(
            prompts=[item["prompt"] for item in batch],
            first_turn_max_new_tokens=args.first_turn_max_new_tokens,
            second_turn_max_new_tokens=args.second_turn_max_new_tokens,
            temperature=None, top_p=None
        )

        modified = set()
        pddl_to_eval = []

        for item, (reasoning, response) in zip(batch, new_completions):
            comp = all_completions[item["id"]]
            comp["prompt_and_reasoning"].append(reasoning)
            comp["final_response"].append(response)

            pddl_code = process_final_response(response, model_name)
            comp["completions"].append(pddl_code)

            if args.evaluate:
                pddl_to_eval.append({
                    "id": item["id"], "llm_pddl": pddl_code,
                    "ground_truth": item["problem_pddl"],
                    "domain": item["domain"], "is_placeholder": item["is_placeholder"]
                })
            modified.add(item["id"])

        # Batch evaluate
        if pddl_to_eval:
            print(f"\nEvaluating {len(pddl_to_eval)} PDDL problems (batch {batch_idx+1}/{len(batches)})...")
            eval_results = evaluate_pddl_batch(
                [item["llm_pddl"] for item in pddl_to_eval],
                [item["ground_truth"] for item in pddl_to_eval],
                [{item["domain"]: domains[item["domain"]]} for item in pddl_to_eval],
                [item["is_placeholder"] for item in pddl_to_eval]
            )

            # Stats
            counts = {k: sum(1 for r in eval_results if r.get(k, False)) for k in ['parseable', 'valid', 'equivalent']}
            print(f"  Parseable: {counts['parseable']}/{len(eval_results)}, "
                  f"Valid: {counts['valid']}/{len(eval_results)}, "
                  f"Equivalent: {counts['equivalent']}/{len(eval_results)}")

            for eval_item, res in zip(pddl_to_eval, eval_results):
                all_completions[eval_item["id"]]["eval_results"].append(res)

        # Save (optimized JSON serialization)
        for problem_id in modified:
            with gzip.open(exp_dir / f"{problem_id}.json.gz", "wt") as f:
                # Use separators for faster JSON serialization
                json.dump(all_completions[problem_id], f, separators=(',', ':'))

class Model:
    """Model wrapper for PDDL generation using SGLang."""

    def __init__(self, model_name, lora_path=None):
        params = get_params(model_name)
        self.name, self.temperature, self.top_p = params["name"], params["temperature"], params["top_p"]
        self.enable_thinking = params["enable_thinking"]
        self.llm = initialize_engine(params["dtype"], model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
        self.pattern_config = MODEL_PATTERNS.get(self.name)
        if lora_path:
            print(f"Warning: LoRA not directly supported with SGLang. Path: {lora_path}")

    def _generate(self, prompts, max_new_tokens, temperature, top_p):
        return self.llm.generate(prompts, {
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p,
            "max_new_tokens": max_new_tokens
        })

    def _format_prompt(self, prompt):
        """Format prompt with chat template."""
        if self.name == "gpt_oss":
            messages, kwargs = [{"role": "user", "content": prompt}], {"reasoning_effort": "medium"}
        elif self.name == "gemma":
            messages, kwargs = [{"role": "user", "content": prompt}], {}
        else:
            messages, kwargs = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ], {}
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)

    def completions(self, prompts: List[str], first_turn_max_new_tokens: int,
                   second_turn_max_new_tokens: int, temperature: float, top_p: float):
        """Generate completions with optional two-turn reasoning."""
        formatted_prompts = [self._format_prompt(p.strip()) for p in prompts]
        outputs = self._generate(formatted_prompts, first_turn_max_new_tokens, temperature, top_p)

        initial_results, continuation_prompts, continuation_indices = [], [], []

        for idx, (formatted_prompt, output) in enumerate(zip(formatted_prompts, outputs)):
            output_text = output['text']
            if self.enable_thinking and not self.pattern_config['thinking_check'](output_text):
                continuation_indices.append(idx)
                continuation_prompts.append(formatted_prompt + output_text + self.pattern_config['thinking_suffix'])
                initial_results.append((formatted_prompt, output_text, True))
            else:
                initial_results.append((formatted_prompt, output_text, False))

        continuation_results = {}
        if continuation_prompts:
            batch_outputs = self._generate(continuation_prompts, second_turn_max_new_tokens, temperature, top_p)
            continuation_results = {idx: (prompt, output['text'])
                                   for idx, prompt, output in zip(continuation_indices, continuation_prompts, batch_outputs)}

        return [continuation_results[idx] if needs_cont and idx in continuation_results else (pre, comp)
                for idx, (pre, comp, needs_cont) in enumerate(initial_results)]

def estimator(num_samples, num_correct, k):
    """Calculate pass@k using unbiased estimator."""
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))

def pass_k(output_dir, model_name, csv_path=None, notes=""):
    """Calculate and report pass@k metrics."""
    # Optimized file loading with early filtering
    results = []
    for p in Path(output_dir).glob("*.json*"):
        try:
            with (gzip.open(p, 'rt') if str(p).endswith('.gz') else open(p, 'rt')) as f:
                r = json.load(f)
                if "eval_results" in r and r["eval_results"]:
                    results.append(r)
        except (json.JSONDecodeError, OSError):
            continue

    if not results:
        print(f"No evaluation results found in {output_dir}")
        return

    # Vectorized computation for better performance
    problem_results = [{
        "pass@1": estimator(len(r["eval_results"]), sum(1 for e in r["eval_results"] if e.get('equivalent', False)), 1),
        "num_samples": len(r["eval_results"]),
        "num_correct": sum(1 for e in r["eval_results"] if e.get('equivalent', False))
    } for r in results]

    num_problems = len(problem_results)
    min_completions = min(r["num_samples"] for r in problem_results)
    max_completions = max(r["num_samples"] for r in problem_results)
    pass_1 = np.mean([r["pass@1"] for r in problem_results])

    print(f"\n{'='*60}\nPDDL Evaluation Results: {Path(output_dir).name}\n{'='*60}")
    print(f"Model: {model_name}\nProblems: {num_problems}\nCompletions: {min_completions}-{max_completions}")
    print(f"Pass@1: {pass_1:.4f}\n{'='*60}\n")

    if csv_path:
        csv_file = Path(csv_path)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not csv_file.exists():
                writer.writerow(['timestamp', 'model', 'dataset', 'pass@1',
                               'num_problems', 'min_completions', 'max_completions', 'notes'])
            writer.writerow([datetime.now().isoformat(), model_name, Path(output_dir).name,
                           f"{pass_1:.4f}", num_problems, min_completions, max_completions, notes])

if __name__ == "__main__":
    args = arg_parser().parse_args()

    domains = load_domains(args.domain_path)
    print(f"Loaded {len(domains)} PDDL domains: {list(domains.keys())}")

    model = Model(args.name, lora_path=args.lora_path)
    make_main(args, args.name.replace("/", "_").replace("-", "_"), model.completions, domains)

    if args.passk_path:
        pass_k(args.output_dir, args.name, args.passk_path, args.notes)
