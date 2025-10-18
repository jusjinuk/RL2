"""
PDDL Planning Environment for RL2

Integrates the Planetarium PDDL evaluation system with RL2's reinforcement learning framework.
This environment evaluates generated PDDL problem descriptions against ground truth problems.
"""

import os
import re
import signal
from typing import Any, Callable

from lark.exceptions import LarkError
from pddl.core import Problem
from pddl.formatter import problem_to_string
from pddl.parser.problem import ProblemParser

from planetarium import builder, downward, graph, metric, oracle

# Environment variables for validation
VALIDATE = os.getenv("VALIDATE", "Validate")
DOWNWARD = os.getenv("DOWNWARD", "downward")

def signal_handler(signum, frame):
    """Handle timeout signals."""
    raise TimeoutError("Timed out")

signal.signal(signal.SIGALRM, signal_handler)

def timeout_and_retry(
    func: Callable,
    *args,
    timeout: int = 30,
    retries: int = 5,
    **kwargs,
) -> Any:
    """Run a function with timeout and retries.

    Args:
        func: Function to run
        timeout: Timeout in seconds per attempt
        retries: Number of retry attempts

    Returns:
        Function return value

    Raises:
        TimeoutError: If all retries timeout
    """
    for _ in range(retries):
        try:
            signal.alarm(timeout)
            return func(*args, **kwargs)
        except TimeoutError:
            continue
        finally:
            signal.alarm(0)
    raise TimeoutError(f"Timed out after {retries} retries")


def clean_pddl(pddl_str: str) -> str:
    """Clean and normalize PDDL string.

    Args:
        pddl_str: Raw PDDL string

    Returns:
        Cleaned PDDL string
    """
    problem: Problem = ProblemParser()(pddl_str)
    return problem_to_string(problem)

def extract_pddl_from_response(response: str) -> str:
    """Extract PDDL code from model response.

    Handles markdown code block formats:

    Args:
        response: Model generated response

    Returns:
        Extracted PDDL code
    """
    m = re.search(r'```(?:\S+)?\s*\n(.*?)\n?```', response, re.DOTALL)
    code = m.group(1).strip() if m else response.strip()
    return code

def fast_equivalence(
    problem_pddl: str,
    llm_problem_pddl: str,
) -> tuple[bool, tuple[bool, bool, bool], dict[str, graph.ProblemGraph]]:
    """Evaluate a PDDL problem quickly (if possible).

    Args:
        problem_pddl (str): The ground truth PDDL.
        llm_problem_pddl (str): The PDDL output from the LLM.

    Returns:
        tuple[bool, dict[str, bool], dict[str, graph.ProblemGraph]]: A tuple
            with a boolean indicating if the problem was resolved, a tuple
            containing whether the PDDL is parseable, valid, and equivalent,
            and a dictionary containing the problem graphs.
    """
    # initialize variables
    parseable = False
    valid = False
    equivalent = False

    problem_graph = None
    llm_problem_graph = None

    resolved = False

    def result():
        return (
            resolved,
            (
                parseable,
                valid,
                equivalent,
            ),
            {
                "problem_graph": problem_graph,
                "llm_problem_graph": llm_problem_graph,
            },
        )

    try:
        # try to parse the LLM output
        llm_problem_graph = builder.build(llm_problem_pddl)
        parseable = True

        # reduce and further validate the LLM output
        oracle.reduce(llm_problem_graph.init())
        oracle.reduce(llm_problem_graph.goal())
        valid = True

        problem_graph = builder.build(problem_pddl)
        init, _ = problem_graph.decompose()

        if len(llm_problem_graph.constants) != len(problem_graph.constants):
            resolved = True
            return result()

        llm_init, _ = llm_problem_graph.decompose()

        if not timeout_and_retry(
            metric.equals,
            init,
            llm_init,
            is_placeholder=False,
            timeout=30,
            retries=5,
        ):
            # If the initial states are not equal, then the problems cannot be equivalent
            resolved = True
            return result()

    except LarkError:
        resolved = True
    except AttributeError:
        resolved = True
    except ValueError:
        resolved = True
    except TimeoutError:
        pass

    return result()


def full_equivalence(
    source: graph.ProblemGraph,
    target: graph.ProblemGraph,
    is_placeholder: bool = False,
) -> bool:
    """Checks if two scene graphs are equivalent.

    Args:
        source (graph.ProblemGraph): The source scene graph.
        target (graph.ProblemGraph): The target scene graph.

    Returns:
        bool: True if the scene graphs are equivalent, False otherwise.
    """
    return metric.equals(
        oracle.fully_specify(source, return_reduced=True),
        oracle.fully_specify(target, return_reduced=True),
        is_placeholder=is_placeholder,
    )


def validate_pddl(
    pddl_str: str,
    domain_str: str,
    fast_downward: str = DOWNWARD,
    **downward_args,
) -> bool:
    """Validate PDDL as solvable.

    Args:
        pddl_str: PDDL problem
        domain_str: PDDL domain
        fast_downward: Path to Fast Downward planner

    Returns:
        True if valid and solvable
    """
    valid = False
    pddl_str = clean_pddl(pddl_str)
    try:
        problem_graph = builder.build(pddl_str)
        plan = oracle.plan_to_string(oracle.plan(problem_graph))
        valid = downward.validate(domain_str, pddl_str, plan, VALIDATE)
    except (LarkError, AttributeError, ValueError):
        pass
    except (oracle.DomainNotSupportedError, NotImplementedError):
        try:
            plan_str, _ = downward.plan(
                domain_str,
                pddl_str,
                downward=fast_downward,
                **downward_args,
            )
            valid = downward.validate(domain_str, pddl_str, plan_str, VALIDATE)
        except Exception:
            pass

    return valid


def equivalence(
    problem_pddl: str,
    llm_problem_pddl: str,
    domains: dict[str, str],
    is_placeholder: bool = False,
) -> tuple[bool, bool, bool]:
    """Evaluate a PDDL problem and save the results.

    Args:
        problem_pddl (str): The ground truth PDDL.
        llm_problem_pddl (str): The PDDL output from the LLM.
        domains (dict[str, str]): The domains to use.
        is_placeholder (bool, optional): Whether the LLM output is a
            placeholder. Defaults to False.

    Returns:
        tuple[bool, bool, bool]: A tuple containing whether the PDDL is
            parseable, valid, and equivalent.
    """

    # fast equivalence check
    resolved, (parseable, valid, equivalent), graphs = fast_equivalence(
        problem_pddl, llm_problem_pddl
    )
    if resolved:
        return parseable, valid, equivalent

    return (
        parseable,
        validate_pddl(
            llm_problem_pddl,
            domains[graphs["llm_problem_graph"].domain],
            alias="lama-first",
        ),
        full_equivalence(
            graphs["problem_graph"],
            graphs["llm_problem_graph"],
            is_placeholder=is_placeholder,
        ),
    )

def _remove_reasoning_prefix(s: str) -> str:
    # Only supports for qwen
    m = re.search(r'</think>', s)
    return s[m.end():] if m else s

async def step(state: str, action: str, extra_info: dict):
    """PDDL environment step function for RL2.

    Evaluates generated PDDL against ground truth problem.

    Args:
        state: Current state (prompt/partial response)
        action: Model's action (generated PDDL)
        extra_info: Extra information dict with:
            - problem_pddl: Ground truth PDDL (required)
            - domain: Domain name (required)
            - domain_pddl: Domain PDDL string (required)
            - is_placeholder: Whether to allow placeholders (default: False)
            - timeout: Evaluation timeout in seconds (default: 600)

    Returns:
        dict with:
            - next_state: None (terminal state)
            - reward: 1.0 if equivalent, 0.0 otherwise
            - score: Same as reward
            - done: True
            - extra_info: Updated with evaluation results
    """
    if '</think>' not in state + action:
        return {
            'next_state': state + action + "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n",
            'reward': 0.0, 'score': 0.0, 'done': False, 'extra_info': extra_info
        }

    if '</think>' in action:
        action = action.split('</think>', 1)[1]

    action = _remove_reasoning_prefix(action)
    llm_problem_pddl = extract_pddl_from_response(action)

    # Get ground truth and domain
    problem_pddl = extra_info.get("problem_pddl")
    domain_name = extra_info.get("domain")
    domain_pddl = extra_info.get("domain_pddl")
    domains = {domain_name: domain_pddl}
    is_placeholder = extra_info.get("is_placeholder")
    timeout = 600

    # Evaluate with timeout
    try:
        signal.alarm(timeout)
        parseable, valid, equivalent = equivalence(
            problem_pddl,
            llm_problem_pddl,
            domains,
            is_placeholder,
        )
        signal.alarm(0)
    except TimeoutError:
        extra_info["error"] = "Evaluation timeout"
        extra_info["parseable"] = False
        extra_info["valid"] = False
        extra_info["equivalent"] = False
        return {
            "next_state": None,
            "reward": 0.0,
            "score": 0.0,
            "done": True,
            "extra_info": extra_info,
        }
    except Exception as e:
        extra_info["error"] = f"Evaluation error: {str(e)}"
        extra_info["parseable"] = False
        extra_info["valid"] = False
        extra_info["equivalent"] = False
        return {
            "next_state": None,
            "reward": 0.0,
            "score": 0.0,
            "done": True,
            "extra_info": extra_info,
        }

    # Update extra_info with results
    extra_info["parseable"] = parseable
    extra_info["valid"] = valid
    extra_info["equivalent"] = equivalent
    extra_info["generated_pddl"] = llm_problem_pddl

    # Compute reward (1.0 for equivalent, 0.0 otherwise)
    reward = 1.0 if equivalent else 0.0

    return {
        "next_state": None,
        "reward": reward,
        "score": reward,
        "done": True,
        "extra_info": extra_info,
    }
