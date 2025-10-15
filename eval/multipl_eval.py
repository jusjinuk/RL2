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
import asyncio
import aiohttp
import numpy as np
import os

LANG_MAP = {"r": "R", "rkt": "Racket", "ml": "OCaml", "jl": "Julia", "lua": "Lua"}

COMMENT_CONFIG = {
    'ml': {'multiline_prefix': ['(*'], 'multiline_postfix': ['*)']},
    'jl': {'multiline_prefix': ['"""'], 'multiline_postfix': ['"""']},
    'lua': {'multiline_prefix': [], 'multiline_postfix': []},
    'r': {'multiline_prefix': [], 'multiline_postfix': []},
    'rkt': {'multiline_prefix': [], 'multiline_postfix': []}
}

MODEL_PATTERNS = {
    'qwen3_think': {
        'end_reasoning': r'</think>',
        'thinking_suffix': "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n",
        'thinking_check': lambda text: '</think>' in text
    },
    'gpt_oss': {
        'end_reasoning': r'<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>',
        'thinking_suffix': "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.<|end|><|start|>assistant<|channel|>final<|message|>",
        'thinking_check': lambda text: bool(re.search(r"assistantfinal.*?'''.*?'''", text, re.DOTALL))
    }
}

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True, choices=list(LANG_MAP.keys()))
    parser.add_argument("--dataset-index", nargs='+', type=int)
    parser.add_argument("--completion-limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--first-turn-max-new-tokens", type=int, default=1024)
    parser.add_argument("--second-turn-max-new-tokens", type=int, default=1024)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--lora-path", type=str)
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--passk-path", type=str, help="CSV file to save pass@k results")
    parser.add_argument("--notes", type=str, default="", help="Additional notes to save in CSV")
    parser.add_argument("--eval-url", type=str, default='http://147.46.15.142:8081/run_code')
    return parser

def get_params(model_name):
    name_lower = model_name.lower()
    if "qwen3" in name_lower:
        if "think" in name_lower:
            return {"name": "qwen3_think", "temperature": 0.6, "top_p": 0.95, "dtype": torch.bfloat16, "enable_thinking": True}
        elif "instruct" in name_lower:
            return {"name": "qwen3_instruct", "temperature": 0.7, "top_p": 0.95, "dtype": torch.bfloat16, "enable_thinking": False}
    elif "gpt-oss" in name_lower:
        return {"name": "gpt_oss", "temperature": 1.0, "top_p": 1.0, "dtype": "auto", "enable_thinking": True}
    raise ValueError(f"Unsupported model: {model_name}")

def initialize_engine(dtype, name):
    gpu_count = torch.cuda.device_count()
    optimal_tp_size = 1 << (gpu_count.bit_length() - 1)
    return sgl.Engine(model_path=name, dtype=dtype, tp_size=optimal_tp_size, mem_fraction_static=0.7)

def read_completions(exp_dir, model_name, first_turn_max_new_tokens, second_turn_max_new_tokens, problem):
    problem_filename = exp_dir / f"{problem['name']}.json.gz"

    if problem_filename.exists():
        with gzip.open(problem_filename, "rt") as f:
            existing = json.loads(f.read())
            return (existing["name"], existing)

    params = get_params(model_name)
    new_completions = {
        "name": problem["name"],
        "language": problem["language"],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "first_turn_max_new_tokens": first_turn_max_new_tokens,
        "second_turn_max_new_tokens": second_turn_max_new_tokens,
        "prompt": problem["prompt"],
        "tests": problem["tests"],
        "prompt_and_reasoning": [],
        "final_response": [],
        "completions": [],
        "eval_results": []
    }
    return (new_completions["name"], new_completions)

async def evaluate_batch_async(codes, url, lang):
    run_url = os.environ.get('SANDBOX_URL', url)
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for code in codes:
            payload = {'code': code, 'language': LANG_MAP[lang].lower()}
            tasks.append(session.post(run_url, json=payload, timeout=aiohttp.ClientTimeout(total=30)))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for resp in responses:
            if isinstance(resp, Exception):
                results.append({'run_result': {'stdout': '', 'stderr': 'Evaluation failed', 'return_code': -1}})
            else:
                async with resp:
                    results.append(await resp.json())

    return results

def evaluate_batch(codes, url, lang):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(evaluate_batch_async(codes, url, lang))

def make_main(args, model_name, gen_completions):
    exp_dir = Path(args.output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    problems = datasets.load_dataset("jsbyun121/MultiPL-E-fixed", f"humaneval-{args.lang}", split="test")

    if args.dataset_index:
        extract_num = lambda name: int(re.search(r'HumanEval_(\d+)_', name).group(1)) if re.search(r'HumanEval_(\d+)_', name) else None
        problems = problems.filter(lambda x: extract_num(x["name"]) in args.dataset_index)

    all_completions = dict(read_completions(exp_dir, args.name, args.first_turn_max_new_tokens, args.second_turn_max_new_tokens, p) for p in problems)

    problem_list = []
    for comp in all_completions.values():
        if len(comp["final_response"]) < args.completion_limit:
            item = {"prompt": comp["prompt"], "name": comp["name"]}
            problem_list.extend([item] * (args.completion_limit - len(comp["final_response"])))

    batches = [problem_list[i:i+args.batch_size] for i in range(0, len(problem_list), args.batch_size)]

    for batch in tqdm(batches, unit="batch"):
        new_completions = gen_completions(
            prompts=[item["prompt"] for item in batch],
            first_turn_max_new_tokens=args.first_turn_max_new_tokens,
            second_turn_max_new_tokens=args.second_turn_max_new_tokens,
            temperature=None,
            top_p=None,
        )

        modified = set()
        codes_to_eval = []

        for item, (reasoning, response) in zip(batch, new_completions):
            comp = all_completions[item["name"]]
            comp["prompt_and_reasoning"].append(reasoning)
            comp["final_response"].append(response)

            code = process_final_response(response, model_name, args.lang)
            comp["completions"].append(code)

            if args.evaluate:
                full_code = code + '\n\n' + comp["tests"]
                codes_to_eval.append((item["name"], full_code))

            modified.add(item["name"])

        if codes_to_eval:
            eval_results = evaluate_batch(
                [code for _, code in codes_to_eval],
                args.eval_url,
                args.lang
            )

            for (name, _), res in zip(codes_to_eval, eval_results):
                all_completions[name]["eval_results"].append(res.get('run_result', {}))

        for name in modified:
            with gzip.open(exp_dir / f"{name}.json.gz", "wt") as f:
                f.write(json.dumps(all_completions[name]))

def _remove_until_end_reasoning(final_response, model_name):
    pattern_config = MODEL_PATTERNS.get(get_params(model_name)["name"])
    if not pattern_config:
        return final_response
    match = re.search(pattern_config['end_reasoning'], final_response)
    return final_response[match.start():] if match else final_response

def _clean_code(completion, language):
    config = COMMENT_CONFIG.get(language, {})
    prefixes = config.get('multiline_prefix', [])
    postfixes = config.get('multiline_postfix', [])

    # Remove multiline comments
    if prefixes and postfixes:
        patterns = [f'{re.escape(start)}.*?{re.escape(end)}' for start in prefixes for end in postfixes]
        if patterns:
            completion = re.sub('|'.join(patterns), '', completion, flags=re.DOTALL)

    # Extract code from markdown blocks
    code_match = re.search(r'```(?:\S+)?\s*\n(.*?)\n?```', completion, re.DOTALL)
    if not code_match:
        return ""

    lines = code_match.group(1).strip().split('\n')

    # Remove leading empty lines only
    while lines and not lines[0].strip():
        lines = lines[1:]

    # Remove language shebang lines
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith(('#lang', '#julia', '#r', '#rkt', '#ocaml', '#lua')):
            lines.pop(idx)
            break

    return '\n'.join(lines).rstrip()
    
def process_final_response(final_response, model_name, lang):
    final_response = _remove_until_end_reasoning(final_response, model_name)
    final_response = _clean_code(final_response, lang)
    return final_response


class Model:
    def __init__(self, model_name, lang, lora_path=None):
        params = get_params(model_name)
        self.name = params["name"]
        self.temperature = params["temperature"]
        self.top_p = params["top_p"]
        self.enable_thinking = params["enable_thinking"]
        self.llm = initialize_engine(params["dtype"], model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
        self.language = LANG_MAP[lang.lower()]
        self.pattern_config = MODEL_PATTERNS.get(self.name)

        if lora_path:
            print(f"Warning: LoRA checkpoint loading not directly supported with SGLang Engine. Path: {lora_path}")

    def _generate(self, prompts, max_new_tokens, temperature, top_p):
        return self.llm.generate(prompts, {
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p,
            "max_new_tokens": max_new_tokens
        })

    def _format_prompt(self, prompt):
        user_msg = f"Using given examples and the signature, generate the missing implementation in {self.language} by wrapping your code in ```{self.language.lower()} markdown blocks:\n\n{prompt}\n\n"

        if self.name == "gpt_oss":
            messages = [{"role": "user", "content": user_msg}]
            kwargs = {"reasoning_effort": "medium"}
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_msg}
            ]
            kwargs = {}

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)

    def _is_thinking_complete(self, output_text):
        if not self.pattern_config:
            raise ValueError(f"Unsupported model: {self.name}")
        return self.pattern_config['thinking_check'](output_text)

    def _get_thinking_suffix(self):
        if not self.pattern_config:
            raise ValueError(f"Unsupported model: {self.name}")
        return self.pattern_config['thinking_suffix']

    def completions(self, prompts: List[str], first_turn_max_new_tokens: int, second_turn_max_new_tokens: int, temperature: float, top_p: float):
        formatted_prompts = [self._format_prompt(prompt.strip()) for prompt in prompts]
        outputs = self._generate(formatted_prompts, first_turn_max_new_tokens, temperature, top_p)

        initial_results = []
        continuation_prompts = []
        continuation_indices = []

        for idx, (formatted_prompt, output) in enumerate(zip(formatted_prompts, outputs)):
            output_text = output['text']
            if self.enable_thinking and not self._is_thinking_complete(output_text):
                continuation_indices.append(idx)
                continuation_prompts.append(formatted_prompt + output_text + self._get_thinking_suffix())
                initial_results.append((formatted_prompt, output_text, True))
            else:
                initial_results.append((formatted_prompt, output_text, False))

        continuation_results = {}
        if continuation_prompts:
            batch_outputs = self._generate(continuation_prompts, second_turn_max_new_tokens, temperature, top_p)
            for idx, prompt, output in zip(continuation_indices, continuation_prompts, batch_outputs):
                continuation_results[idx] = (prompt, output['text'])

        results = []
        for idx, (pre_completion, completion_result, needs_continuation) in enumerate(initial_results):
            if needs_continuation and idx in continuation_results:
                results.append(continuation_results[idx])
            else:
                results.append((pre_completion, completion_result))

        return results

def estimator(num_samples, num_correct, k):
    """Calculates 1 - comb(num_samples - num_correct, k) / comb(num_samples, k)."""
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))

def load_result(file_path):
    open_fn = gzip.open if str(file_path).endswith('.gz') else open
    with open_fn(file_path, 'rt') as f:
        return json.load(f)

def pass_k(output_dir, lang, model_name, csv_path=None, notes=""):
    results = [result for result in [load_result(path) for path in Path(output_dir).glob("*.json*")] if "eval_results" in result and result["eval_results"]]

    if not results:
        print(f"No evaluation results found in {output_dir}")
        return

    problem_results = []
    for result in results:
        num_samples = len(result["eval_results"])
        num_correct = sum(1 for eval_res in result["eval_results"] if eval_res.get('return_code', 1) == 0)
        problem_results.append({"pass@1": estimator(num_samples, num_correct, 1), "pass@10": estimator(num_samples, num_correct, 10), "pass@100": estimator(num_samples, num_correct, 100), "num_samples": num_samples, "num_correct": num_correct})

    num_problems = len(problem_results)
    min_completions = min(result["num_samples"] for result in problem_results)
    max_completions = max(result["num_samples"] for result in problem_results)
    pass_1 = np.mean([result["pass@1"] for result in problem_results])

    name = f"{Path(output_dir).name}_{lang}"
    print(f"{name},1,{pass_1:.4f},{num_problems},{min_completions},{max_completions}")

    if csv_path:
        import csv
        from datetime import datetime
        csv_file = Path(csv_path)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not csv_file.exists():
                writer.writerow(['timestamp', 'model', 'language', 'dataset', 'pass@1', 'num_problems', 'min_completions', 'max_completions', 'notes'])
            writer.writerow([datetime.now().isoformat(), model_name, lang, Path(output_dir).name, f"{pass_1:.4f}", num_problems, min_completions, max_completions, notes])

if __name__ == "__main__":
    args = arg_parser().parse_args()
    model = Model(args.name, args.lang, lora_path=args.lora_path)
    make_main(args, args.name.replace("/", "_").replace("-", "_"), model.completions)
    if args.passk_path:
        pass_k(args.output_dir, args.lang, args.name, args.passk_path, args.notes)
