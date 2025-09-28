import os
import re
import asyncio
import aiohttp

lang2Language = {"r": "R", "rkt": "Racket", "ml": "OCaml", "jl": "Julia", "lua": "Lua"}
Language2lang = {"R": "r", "Racket": "rkt", "OCaml": "ml", "Julia": "jl", "Lua": "lua"}
lang2executor = {"r": "R", "rkt": "racket", "ml": "ocaml", "jl": "julia", "lua": "lua"}

def _ml_prefix(): 
    return {'ml': ['(*'], 'jl': ['"""'], 'lua': [], 'r': [], 'rkt': []}

def _ml_postfix(): 
    return {'ml': ['*)'], 'jl': ['"""'], 'lua': [], 'r': [], 'rkt': []}

def _remove_reasoning_prefix(s: str) -> str:
    # Only supports for qwen
    m = re.search(r'</think>', s)
    return s[m.end():] if m else s

def _clean_code(s: str, lang: str) -> str:
    starts = [re.escape(x) for x in _ml_prefix().get(lang, [])]
    ends   = [re.escape(x) for x in _ml_postfix().get(lang, [])]
    pats = [f'{a}.*?{b}' for a in starts for b in ends]
    s = re.sub('|'.join(pats), '', s, flags=re.DOTALL) if pats else s
    
    m = re.search(r'```(?:\S+)?\s*\n(.*?)\n?```', s, re.DOTALL)
    code = m.group(1).strip() if m else s.strip()
    lines = [ln for ln in code.split('\n')]
    while lines and not lines[0].strip():
        lines = lines[1:]
    for i, ln in enumerate(lines):
        t = ln.strip().lower()
        if t.startswith(('#lang', '#julia', '#r', '#rkt', '#ocaml', '#lua')):
            lines.pop(i)
            break
    return '\n'.join(lines).rstrip()

def _passed(rc: int) -> bool:
    return rc == 0

async def step(state: str, action: str, extra_info: dict):
    if '</think>' not in state + action:
        return {
            'next_state': state + action + "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n",
            'reward': 0.0, 'score': 0.0, 'done': False, 'extra_info': extra_info
        }

    if '</think>' in action:
        action = action.split('</think>', 1)[1]

    lang = Language2lang[extra_info["language"]]
    tests = extra_info["tests"]
    run_url = os.environ.get('SANDBOX_URL', 'http://localhost:8080/run_code')

    action = _remove_reasoning_prefix(action)
    code_body = _clean_code(action, lang)

    parts = [code_body, tests]
    final_code = '\n'.join(p for p in parts if p is not None)

    payload = {'code': final_code, 'language': lang2executor[lang]}
    async with aiohttp.ClientSession() as sess:
        while True:
            try:
                async with sess.post(run_url, json=payload) as resp:
                    res = await resp.json()
                    break
            except Exception:
                await asyncio.sleep(1)

    rr = res.get('run_result', {})
    rc = rr.get('return_code', 1)
    stdout = rr.get('stdout', '')
    stderr = rr.get('stderr', '')

    success = _passed(rc)

    extra_info["stdout"] = stdout
    extra_info["stderr"] = stderr
    extra_info["clean_code"] = code_body

    if success:
        return {
            'next_state': None,
            'reward': 1.0, 'score': 1.0, 'done': True, 'extra_info': extra_info
        }
    else:
        return {
            'next_state': None,
            'reward': 0.0, 'score': 0.0, 'done': True, 'extra_info': extra_info
        }

    # # Useful for multi-turn tool usage
    # exec_block = (
    #     f"\n<exec>\nreturn_code={rc}\n\n<stdout>\n{_cap(stdout)}\n</stdout>\n\n"
    #     f"<stderr>\n{_cap(stderr)}\n</stderr>\n</exec>\n"
    # )
    # return {
    #     'next_state': state + action + exec_block + "The execution failed or tests did not pass. I should fix the code and try again.\n",
    #     'reward': 0.0, 'score': 0.0, 'done': False, 'extra_info': extra_info
    # }
